from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file, save_model
from sie_server.adapters.tensorrt_llm import compat

_SITECUSTOMIZE_DIR = Path(compat.__file__).parent / "_compat"
_PINNED_MADLAD_EMBEDDING_SHAPE = (256_000, 1_024)


def _compat_environment(prepend: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["SIE_TRTLLM_RC24_COMPAT"] = "1"
    python_paths = (str(_SITECUSTOMIZE_DIR), str(prepend), environment.get("PYTHONPATH", ""))
    environment["PYTHONPATH"] = os.pathsep.join(path for path in python_paths if path)
    return environment


def _write_fake_tensorrt_llm(root: Path, *, version: str) -> None:
    package = root / "tensorrt_llm"
    sources = {
        "__init__.py": "",
        "_torch/__init__.py": "",
        "_torch/models/__init__.py": "",
        "_torch/models/modeling_t5.py": "class T5ForConditionalGeneration:\n    pass\n",
        "_torch/models/checkpoints/__init__.py": "",
        "_torch/models/checkpoints/hf/__init__.py": "",
        "_torch/models/checkpoints/hf/weight_loader.py": "class HfWeightLoader:\n    pass\n",
    }
    for relative, contents in sources.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    metadata_dir = root / f"tensorrt_llm-{version}.dist-info"
    metadata_dir.mkdir()
    (metadata_dir / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: tensorrt-llm\nVersion: {version}\n",
        encoding="utf-8",
    )


def _load_t5_weights(
    weights: dict[str, object], *, tie_word_embeddings: bool = True
) -> tuple[dict[str, object], object]:
    observed: dict[str, object] = {}
    sentinel = object()

    class Model:
        def __init__(self) -> None:
            self.model_config = SimpleNamespace(
                pretrained_config=SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
            )

        def load_weights(self, incoming: dict[str, object], **_kwargs: object) -> object:
            observed.update(incoming)
            return sentinel

    module = SimpleNamespace(T5ForConditionalGeneration=Model)
    compat._install_t5_weight_patch(module)
    compat._install_t5_weight_patch(module)
    result = Model().load_weights(weights)
    assert result is sentinel
    return observed, result


def _save_model_t5(checkpoint: Path, *, metadata: dict[str, str] | None = None) -> None:
    class SaveModelT5(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = torch.nn.Embedding(4, 3)
            self.encoder = torch.nn.Module()
            self.encoder.embed_tokens = self.shared
            self.decoder = torch.nn.Module()
            self.decoder.embed_tokens = self.shared
            self.lm_head = torch.nn.Linear(3, 4, bias=False)

    save_model(SaveModelT5(), checkpoint, metadata=metadata)


def test_t5_weight_patch_restores_legitimate_save_model_t5_alias(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    _save_model_t5(checkpoint)
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        assert handle.metadata() == {
            "encoder.embed_tokens.weight": "decoder.embed_tokens.weight",
            "shared.weight": "decoder.embed_tokens.weight",
        }
    weights = load_file(checkpoint)
    assert set(weights) == {"decoder.embed_tokens.weight", "lm_head.weight"}

    observed, _result = _load_t5_weights(weights)

    assert observed["shared.weight"] is weights["decoder.embed_tokens.weight"]
    assert observed["lm_head.weight"] is weights["lm_head.weight"]


def test_t5_weight_patch_leaves_legitimate_non_t5_save_model_alias_opaque(tmp_path: Path) -> None:
    class TopLevelAlias(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))
            self.alias = self.weight

    checkpoint = tmp_path / "model.safetensors"
    save_model(TopLevelAlias(), checkpoint)
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        assert handle.metadata() == {"weight": "alias"}
    weights = load_file(checkpoint)
    assert set(weights) == {"alias"}

    observed, _result = _load_t5_weights(weights)

    assert observed == weights
    assert "weight" not in observed


def test_t5_weight_patch_ignores_user_metadata_collision_with_real_tensor_name(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    _save_model_t5(checkpoint, metadata={"shared.weight": "lm_head.weight"})
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        assert handle.metadata()["shared.weight"] == "lm_head.weight"
    weights = load_file(checkpoint)
    assert set(weights) == {"decoder.embed_tokens.weight", "lm_head.weight"}

    observed, _result = _load_t5_weights(weights, tie_word_embeddings=False)

    assert observed["shared.weight"] is weights["decoder.embed_tokens.weight"]
    assert observed["shared.weight"] is not weights["lm_head.weight"]


def test_t5_weight_patch_restores_pinned_madlad_retained_embedding_shape() -> None:
    decoder_embedding = torch.empty(_PINNED_MADLAD_EMBEDDING_SHAPE, device="meta")
    lm_head = torch.empty(_PINNED_MADLAD_EMBEDDING_SHAPE, device="meta")

    observed, _result = _load_t5_weights(
        {
            "decoder.embed_tokens.weight": decoder_embedding,
            "lm_head.weight": lm_head,
        },
        tie_word_embeddings=False,
    )

    assert observed["shared.weight"] is decoder_embedding
    assert isinstance(observed["shared.weight"], torch.Tensor)
    assert tuple(observed["shared.weight"].shape) == _PINNED_MADLAD_EMBEDDING_SHAPE
    assert observed["lm_head.weight"] is lm_head


def test_t5_weight_patch_ignores_arbitrary_dotted_metadata_chains_and_conflicts(tmp_path: Path) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    save_file(
        {"decoder.embed_tokens.weight": torch.zeros(2, 2)},
        first,
        metadata={
            "shared.weight": "unrelated.actual.weight",
            "attacker.chain.destination": "attacker.chain.source",
            "attacker.cycle.left": "attacker.cycle.right",
        },
    )
    save_file(
        {"unrelated.actual.weight": torch.ones(2, 2)},
        second,
        metadata={
            "shared.weight": "decoder.embed_tokens.weight",
            "attacker.chain.source": "decoder.embed_tokens.weight",
            "attacker.cycle.right": "attacker.cycle.left",
        },
    )
    with safe_open(first, framework="pt", device="cpu") as handle:
        assert handle.metadata()["shared.weight"] == "unrelated.actual.weight"
    with safe_open(second, framework="pt", device="cpu") as handle:
        assert handle.metadata()["shared.weight"] == "decoder.embed_tokens.weight"
    weights = {**load_file(first), **load_file(second)}

    observed, _result = _load_t5_weights(weights)

    assert observed["shared.weight"] is weights["decoder.embed_tokens.weight"]
    assert "attacker.chain.destination" not in observed
    assert "attacker.chain.source" not in observed
    assert "attacker.cycle.left" not in observed
    assert "attacker.cycle.right" not in observed


@pytest.mark.parametrize(
    "source_key",
    ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"],
)
def test_t5_weight_patch_accepts_each_known_top_level_input_embedding(source_key: str) -> None:
    tensor = object()

    observed, _result = _load_t5_weights({source_key: tensor})

    assert observed["shared.weight"] is tensor


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_t5_weight_patch_never_uses_lm_head_as_missing_shared_embedding(
    tie_word_embeddings: bool,
) -> None:
    lm_head = object()

    observed, _result = _load_t5_weights(
        {"lm_head.weight": lm_head},
        tie_word_embeddings=tie_word_embeddings,
    )

    assert observed == {"lm_head.weight": lm_head}


def test_t5_weight_patch_accepts_identical_known_embedding_candidates() -> None:
    tensor = object()
    observed, _result = _load_t5_weights(
        {
            "encoder.embed_tokens.weight": tensor,
            "decoder.embed_tokens.weight": tensor,
        }
    )

    assert observed["shared.weight"] is tensor


def test_t5_weight_patch_preserves_existing_shared_embedding() -> None:
    shared = object()
    decoder = object()

    observed, _result = _load_t5_weights(
        {
            "shared.weight": shared,
            "decoder.embed_tokens.weight": decoder,
        }
    )

    assert observed["shared.weight"] is shared


def test_t5_weight_patch_rejects_distinct_known_embedding_candidates() -> None:
    with pytest.raises(RuntimeError, match="ambiguous T5 shared embedding tensors"):
        _load_t5_weights(
            {
                "encoder.embed_tokens.weight": object(),
                "decoder.embed_tokens.weight": object(),
            }
        )


def test_rc24_install_leaves_generic_weight_loader_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    class Loader:
        def _prefetch_and_load(self, weight_files: tuple[str, ...]) -> tuple[str, ...]:
            return weight_files

    class Model:
        def __init__(self, _config: object) -> None:
            pass

        def load_weights(self, _weights: dict[str, object]) -> None:
            return None

    original_loader = Loader._prefetch_and_load
    weight_loader = SimpleNamespace(HfWeightLoader=Loader)
    modeling_t5 = SimpleNamespace(T5ForConditionalGeneration=Model)
    monkeypatch.setattr(compat, "verify_exact_rc24_sources", lambda *_args: None)
    monkeypatch.setattr(compat, "install_transformers_5_t5_tokenizer_compatibility", lambda: None)
    monkeypatch.setattr(
        compat.importlib,
        "import_module",
        lambda name: modeling_t5 if name.endswith("modeling_t5") else weight_loader,
    )

    compat.install_rc24_compatibility()

    assert Loader._prefetch_and_load is original_loader


@pytest.mark.parametrize(
    ("scale_decoder_outputs", "tie_word_embeddings", "expected"),
    [(True, False, True), (False, True, False), (None, True, True), (None, False, False)],
)
def test_t5_patch_uses_saved_scaling_then_legacy_fallback(
    scale_decoder_outputs: bool | None,
    tie_word_embeddings: bool,
    expected: bool,
) -> None:
    class Model:
        def __init__(self, config: Any) -> None:
            _ = config
            self.rescale_before_lm_head = True

    compat._install_t5_scaling_patch(SimpleNamespace(T5ForConditionalGeneration=Model))
    fields: dict[str, Any] = {"tie_word_embeddings": tie_word_embeddings}
    if scale_decoder_outputs is not None:
        fields["scale_decoder_outputs"] = scale_decoder_outputs

    model = Model(SimpleNamespace(pretrained_config=SimpleNamespace(**fields)))

    assert model.rescale_before_lm_head is expected


def test_t5_patch_rejects_missing_pretrained_config() -> None:
    class Model:
        def __init__(self, model_config: Any) -> None:
            _ = model_config
            self.rescale_before_lm_head = True

    compat._install_t5_scaling_patch(SimpleNamespace(T5ForConditionalGeneration=Model))

    with pytest.raises(RuntimeError, match="missing pretrained_config"):
        Model(SimpleNamespace(scale_decoder_outputs=False))


def test_t5_tokenizer_patch_preserves_checkpoint_serialization(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}\n", encoding="utf-8")
    sentinel = object()
    calls: list[str] = []

    class Backend:
        @classmethod
        def convert_to_native_format(cls, trust_remote_code: bool = False, **kwargs: Any) -> dict[str, Any]:
            return {"fallback": cls.__name__, "trust_remote_code": trust_remote_code, **kwargs}

    class T5Tokenizer(Backend):
        pass

    class TokenizerFast:
        @staticmethod
        def from_file(path: str) -> object:
            calls.append(path)
            return sentinel

    module = SimpleNamespace(T5Tokenizer=T5Tokenizer)
    backend = SimpleNamespace(TokenizerFast=TokenizerFast)
    compat._install_t5_tokenizer_serialization_patch(module, backend)
    compat._install_t5_tokenizer_serialization_patch(module, backend)

    converted = T5Tokenizer.convert_to_native_format(
        tokenizer_file=str(tokenizer_path),
        trust_remote_code=False,
        revision="a" * 40,
    )

    assert converted == {"revision": "a" * 40, "tokenizer_object": sentinel}
    assert calls == [str(tokenizer_path)]


def test_t5_tokenizer_patch_delegates_without_a_trusted_local_serialization(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    class Backend:
        @classmethod
        def convert_to_native_format(cls, trust_remote_code: bool = False, **kwargs: Any) -> dict[str, Any]:
            return {"fallback": cls.__name__, "trust_remote_code": trust_remote_code, **kwargs}

    class T5Tokenizer(Backend):
        pass

    module = SimpleNamespace(T5Tokenizer=T5Tokenizer)
    backend = SimpleNamespace(TokenizerFast=SimpleNamespace())
    compat._install_t5_tokenizer_serialization_patch(module, backend)

    assert T5Tokenizer.convert_to_native_format(
        tokenizer_file=str(missing),
        trust_remote_code=False,
    ) == {
        "fallback": "T5Tokenizer",
        "tokenizer_file": str(missing),
        "trust_remote_code": False,
    }
    assert T5Tokenizer.convert_to_native_format(
        tokenizer_file=str(missing),
        trust_remote_code=True,
    ) == {
        "fallback": "T5Tokenizer",
        "tokenizer_file": str(missing),
        "trust_remote_code": True,
    }


def test_exact_source_guard_rejects_version_and_hash_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    weight_path = tmp_path / "weight_loader.py"
    t5_path = tmp_path / "modeling_t5.py"
    weight_path.write_text("weight source\n")
    t5_path.write_text("t5 source\n")
    weight_module = ModuleType("weight_loader")
    weight_module.__file__ = str(weight_path)
    t5_module = ModuleType("modeling_t5")
    t5_module.__file__ = str(t5_path)

    monkeypatch.setattr(compat.importlib.metadata, "version", lambda _name: "1.3.0rc25")
    with pytest.raises(RuntimeError, match=r"requires 1\.3\.0rc24"):
        compat.verify_exact_rc24_sources(weight_module, t5_module)

    monkeypatch.setattr(compat.importlib.metadata, "version", lambda _name: "1.3.0rc24")
    monkeypatch.setattr(compat, "HF_WEIGHT_LOADER_SHA256", hashlib.sha256(weight_path.read_bytes()).hexdigest())
    monkeypatch.setattr(compat, "MODELING_T5_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="changed source modeling_t5"):
        compat.verify_exact_rc24_sources(weight_module, t5_module)


def test_exact_transformers_source_guard_rejects_version_and_hash_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    t5_tokenization_path = tmp_path / "tokenization_t5.py"
    backend_path = tmp_path / "tokenization_utils_tokenizers.py"
    t5_tokenization_path.write_text("t5 tokenizer source\n")
    backend_path.write_text("backend source\n")
    t5_tokenization_module = ModuleType("tokenization_t5")
    t5_tokenization_module.__file__ = str(t5_tokenization_path)
    backend_module = ModuleType("tokenization_utils_tokenizers")
    backend_module.__file__ = str(backend_path)

    monkeypatch.setattr(compat.importlib.metadata, "version", lambda _name: "5.5.5")
    with pytest.raises(RuntimeError, match=r"requires Transformers 5\.5\.4"):
        compat.verify_exact_transformers_5_sources(t5_tokenization_module, backend_module)

    monkeypatch.setattr(compat.importlib.metadata, "version", lambda _name: "5.5.4")
    monkeypatch.setattr(
        compat,
        "T5_TOKENIZATION_SHA256",
        hashlib.sha256(t5_tokenization_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(compat, "TOKENIZERS_BACKEND_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="changed source tokenization_utils_tokenizers"):
        compat.verify_exact_transformers_5_sources(t5_tokenization_module, backend_module)


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("missing", "synthetic missing TensorRT-LLM"),
        ("version", "requires 1.3.0rc24, found 1.3.0rc25"),
        ("hash", "changed source"),
    ],
)
def test_sitecustomize_aborts_interpreter_on_compatibility_failure(
    tmp_path: Path,
    failure: str,
    expected_error: str,
) -> None:
    if failure == "missing":
        (tmp_path / "tensorrt_llm.py").write_text(
            'raise ModuleNotFoundError("synthetic missing TensorRT-LLM")\n',
            encoding="utf-8",
        )
    else:
        _write_fake_tensorrt_llm(
            tmp_path,
            version="1.3.0rc25" if failure == "version" else "1.3.0rc24",
        )

    completed = subprocess.run(
        [sys.executable, "-c", 'print("INTERPRETER_CONTINUED")'],
        check=False,
        capture_output=True,
        env=_compat_environment(tmp_path),
        text=True,
    )

    assert completed.returncode != 0
    assert "INTERPRETER_CONTINUED" not in completed.stdout
    assert "TensorRT-LLM rc24 compatibility startup failed" in completed.stderr
    assert expected_error in completed.stderr


def test_sitecustomize_allows_startup_after_successful_install(tmp_path: Path) -> None:
    package = tmp_path / "sie_server/adapters/tensorrt_llm"
    for relative in (
        "sie_server/__init__.py",
        "sie_server/adapters/__init__.py",
        "sie_server/adapters/tensorrt_llm/__init__.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (package / "compat.py").write_text(
        "import builtins\n\ndef install_rc24_compatibility():\n    builtins.SIE_TRTLLM_COMPAT_INSTALLED = True\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import builtins; assert builtins.SIE_TRTLLM_COMPAT_INSTALLED; print('PATCHED_STARTUP')",
        ],
        check=False,
        capture_output=True,
        env=_compat_environment(tmp_path),
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "PATCHED_STARTUP\n"
    assert "Error in sitecustomize" not in completed.stderr
