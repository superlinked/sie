"""Conservative identity of an immutable model's local execution profile.

The identity is a versioned digest, not a numerical equivalence claim. Both
sides must compute the same descriptor; unknown weights or execution settings
fail closed. Hybrid routing additionally checks the upstream's current record.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import sie_sdk
import torch
from huggingface_hub.utils import HFValidationError, validate_repo_id
from threadpoolctl import threadpool_info

import sie_server
from sie_server.config.engine import EngineConfig
from sie_server.config.model import ModelConfig, is_immutable_revision, is_remote_adapter_path

_INFERENCE_DISTRIBUTIONS = (
    "torch",
    "transformers",
    "tokenizers",
    "numpy",
    "threadpoolctl",
    "sentence-transformers",
    "FlagEmbedding",
    "safetensors",
    "onnxruntime",
    "mlx",
    "sglang",
    "tensorrt-llm",
)

_RUNTIME_NONCE = uuid4().hex


def runtime_instance_id() -> str:
    """Opaque process identity; restarts and forks require a fresh local probe."""
    return hashlib.sha256(f"{os.getpid()}:{_RUNTIME_NONCE}".encode()).hexdigest()


# These adapters execute in this Python process and consume the loader's
# revision-pinned model source. Child/vendor engines require effective runtime
# evidence before they can claim the same execution identity.
_PINNED_PROCESS_ADAPTERS = frozenset(
    {
        "sie_server.adapters.bge_m3:BGEM3Adapter",
    }
)

_HARDWARE_TEXT_MAX_BYTES = 1 << 20
_CPU_FACTS = frozenset(
    {
        "vendor_id",
        "model name",
        "cpu family",
        "model",
        "stepping",
        "microcode",
        "flags",
        "features",
        "cpu implementer",
        "cpu architecture",
        "cpu variant",
        "cpu part",
        "cpu revision",
    }
)
_NUMERICAL_ENVIRONMENT = (
    "ATEN_CPU_CAPABILITY",
    "NVIDIA_TF32_OVERRIDE",
    "CUBLAS_WORKSPACE_CONFIG",
    "MKL_CBWR",
    "MKL_ENABLE_INSTRUCTIONS",
    "MKL_DYNAMIC",
    "OPENBLAS_CORETYPE",
    "ONEDNN_MAX_CPU_ISA",
    "DNNL_MAX_CPU_ISA",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _hardware_text(path: str) -> str:
    with Path(path).open("rb") as stream:
        data = stream.read(_HARDWARE_TEXT_MAX_BYTES + 1)
    if not data or len(data) > _HARDWARE_TEXT_MAX_BYTES:
        raise ValueError("execution hardware facts are unavailable")
    text = data.decode("utf-8").strip()
    if not text:
        raise ValueError("execution hardware facts are unavailable")
    return text


def _cpu_hardware() -> dict[str, Any]:
    system = platform.system()
    if system == "Linux":
        facts: dict[str, set[str]] = {}
        for line in _hardware_text("/proc/cpuinfo").splitlines():
            key, separator, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if separator and key in _CPU_FACTS and value:
                facts.setdefault(key, set()).add(value)
        if not (facts.get("model name") or facts.get("cpu part")) or not (facts.get("flags") or facts.get("features")):
            raise ValueError("CPU execution hardware cannot be identified")
        return {key: sorted(values) for key, values in facts.items()}
    if system == "Darwin":
        try:
            result = subprocess.run(
                [
                    "/usr/sbin/sysctl",
                    "-n",
                    "machdep.cpu.brand_string",
                    "hw.model",
                    "hw.machine",
                    "hw.ncpu",
                    "kern.osversion",
                ],
                capture_output=True,
                check=True,
                timeout=1.0,
            )
        except (OSError, subprocess.SubprocessError):
            raise ValueError("CPU execution hardware cannot be identified") from None
        values = result.stdout.decode("utf-8").strip().splitlines()
        if len(values) != 5 or any(not value for value in values):
            raise ValueError("CPU execution hardware cannot be identified")
        return dict(zip(("brand", "model", "machine", "cores", "os_build"), values, strict=True))
    raise ValueError("CPU execution hardware cannot be identified")


def _execution_hardware(device: str) -> dict[str, Any]:
    """Identify observed hardware, never infer it from a configured device label."""
    resolved = torch.device(device)
    context: dict[str, Any] = {
        "kernel": platform.release(),
        "cpu": _cpu_hardware(),
        "capability": torch.backends.cpu.get_cpu_capability(),
    }
    if resolved.type == "cpu":
        return context
    if resolved.type == "cuda" and platform.system() == "Linux" and torch.version.cuda is not None:
        properties = torch.cuda.get_device_properties(resolved)
        context.update(
            cuda={
                "name": properties.name,
                "capability": [properties.major, properties.minor],
                "total_memory": properties.total_memory,
                "multiprocessors": properties.multi_processor_count,
                "driver": _hardware_text("/proc/driver/nvidia/version"),
            }
        )
        return context
    raise ValueError("execution hardware cannot be identified")


def _json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 64:
        raise ValueError("profile settings exceed the identity nesting limit")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("profile settings must use string keys")
        return {key: _json_value(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item, depth=depth + 1) for item in value]
    if value is None or isinstance(value, str | bool | int | float):
        return value
    raise ValueError("profile settings cannot be represented in an identity")


@lru_cache(maxsize=1)
def _execution_code() -> dict[str, Any]:
    """Hash shipped serving sources and identify the inference dependency stack."""
    digest = hashlib.sha256()
    for module in (sie_server, sie_sdk):
        if module.__file__ is None:
            raise ValueError("serving sources cannot be identified")
        root = Path(module.__file__).parent
        sources = sorted(root.rglob("*.py"))
        if not sources:
            raise ValueError("serving sources cannot be identified")
        for source in sources:
            name = f"{module.__name__}/{source.relative_to(root).as_posix()}".encode()
            content = source.read_bytes()
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    dependencies: dict[str, str | None] = {}
    for distribution in _INFERENCE_DISTRIBUTIONS:
        try:
            dependencies[distribution] = version(distribution)
        except PackageNotFoundError:
            dependencies[distribution] = None
    return {
        "sources": digest.hexdigest(),
        "dependencies": dependencies,
        "python": sys.implementation.cache_tag,
        "platform": [platform.system(), platform.machine()],
    }


@lru_cache(maxsize=256)
def _adapter_source_available(adapter_path: str) -> bool:
    """Refuse custom/unidentifiable adapters without importing provider code."""
    module_name, separator, class_name = adapter_path.partition(":")
    if not separator or not class_name.isidentifier() or not module_name.startswith("sie_server.adapters."):
        return False
    parts = module_name.split(".")[1:]
    if not all(part.isidentifier() for part in parts) or sie_server.__file__ is None:
        return False
    root = Path(sie_server.__file__).parent.resolve()
    module_path = root.joinpath(*parts)
    source = module_path.with_suffix(".py")
    if not source.is_file():
        source = module_path / "__init__.py"
    if not source.is_file() or not source.resolve().is_relative_to(root):
        return False
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return False
    return any(isinstance(node, ast.ClassDef) and node.name == class_name for node in tree.body)


def _numerical_libraries() -> dict[str, Any]:
    """Observe loaded BLAS kernels and threads without exporting host paths."""
    keys = ("user_api", "internal_api", "prefix", "version", "num_threads", "threading_layer", "architecture")
    libraries = [{key: library.get(key) for key in keys} for library in threadpool_info()]
    blas = [library for library in libraries if library["user_api"] == "blas"]
    if any(
        library["internal_api"] not in {"openblas", "blis"}
        or not library["version"]
        or not library["architecture"]
        or not library["threading_layer"]
        or not isinstance(library["num_threads"], int)
        or library["num_threads"] < 1
        for library in blas
    ):
        raise ValueError("numerical execution libraries cannot be identified")
    # NumPy's system BLAS remains relevant even alongside another loaded BLAS.
    build = np.show_config(mode="dicts")
    numpy_blas = build.get("Build Dependencies", {}).get("blas", {}).get("name")
    uses_accelerate = numpy_blas == "accelerate"
    accelerate = None
    if uses_accelerate:
        if platform.system() != "Darwin":
            raise ValueError("numerical execution libraries cannot be identified")
        accelerate = platform.mac_ver()[0]
        if not accelerate:
            raise ValueError("numerical execution libraries cannot be identified")
    else:
        supported_builds = {
            "openblas": "openblas",
            "openblas64": "openblas",
            "scipy-openblas": "openblas",
            "scipy-openblas64": "openblas",
            "blis": "blis",
        }
        required_api = supported_builds.get(numpy_blas)
        if required_api is None or not any(library["internal_api"] == required_api for library in blas):
            raise ValueError("numerical execution libraries cannot be identified")
    return {
        "loaded": sorted(libraries, key=lambda library: json.dumps(library, sort_keys=True)),
        "accelerate_os_version": accelerate,
    }


def _execution_runtime() -> dict[str, Any]:
    """Capture mutable precision/determinism settings on every observation."""
    return {
        "default_dtype": str(torch.get_default_dtype()),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        "fp16_reduced_precision": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduced_precision": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_build": torch.__config__.show(),
        "numpy_build": np.show_config(mode="dicts"),
        "numerical_libraries": _numerical_libraries(),
        "numerical_environment": {name: os.environ.get(name) for name in _NUMERICAL_ENVIRONMENT},
    }


def local_profile_identity(
    config: ModelConfig,
    profile_name: str,
    *,
    device: str,
    engine_config: EngineConfig | None = None,
) -> str | None:
    """Identify a pinned local profile without importing its adapter or weights.

    Aliases and profile inheritance do not change identity when their resolved
    settings match. Every resolved setting is included conservatively, even
    batching or admission settings that may be operational rather than semantic.
    Remote profiles and mutable/local/package weights have no identity.
    """
    if (
        config.weights_path is not None
        or config.package_backed
        or not config.hf_id
        or not is_immutable_revision(config.hf_revision)
        or not isinstance(device, str)
        or not device
    ):
        return None
    if any(not is_immutable_revision(revision) for revision in config.hf_tokenizer_dependencies.values()):
        return None
    try:
        validate_repo_id(config.hf_id)
        if Path(config.hf_id).exists():
            return None
        profile = config.resolve_profile(profile_name)
        if (
            is_remote_adapter_path(profile.adapter_path)
            or profile.adapter_path not in _PINNED_PROCESS_ADAPTERS
            or not _adapter_source_available(profile.adapter_path)
        ):
            return None
        if config.lora_revisions():
            return None
        # The loader applies loadtime options after its model/revision kwargs.
        # Nested HF config kwargs can override the library's revision and trust
        # settings too. Refuse those alternate authorities until verified.
        if any(key in profile.loadtime for key in ("model_name_or_path", "revision", "serving_artifact")):
            return None
        if profile.loadtime.get("config_kwargs"):
            return None
        if profile.loadtime.get("trust_remote_code") not in (None, False):
            return None
        if engine_config is not None and not isinstance(engine_config, EngineConfig):
            return None
        if profile.compute_precision is None and engine_config is None:
            return None
        descriptor = {
            "version": 1,
            "hf_id": config.hf_id,
            "hf_revision": config.hf_revision,
            "tokenizer_dependencies": config.hf_tokenizer_dependencies,
            "inputs": config.inputs.model_dump(),
            "tasks": config.tasks.model_dump(),
            "max_sequence_length": config.max_sequence_length,
            "profile": _json_value(profile.model_dump()),
            "execution": _execution_code(),
            "runtime": _execution_runtime(),
            "device": device,
            "hardware": _execution_hardware(device),
            "engine": engine_config.model_dump(mode="json") if engine_config is not None else None,
        }
        encoded = json.dumps(
            descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    except (
        OSError,
        TypeError,
        ValueError,
        HFValidationError,
        RecursionError,
        RuntimeError,
        AssertionError,
        AttributeError,
    ):
        return None
    return "v1:sha256:" + hashlib.sha256(encoded).hexdigest()
