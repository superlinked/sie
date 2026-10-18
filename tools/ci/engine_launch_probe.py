"""Parse every routed profile's engine launch with the engine's own argument parser.

Runs inside a built server image, with the image's interpreter, catalog and
bundles. Each profile routed to the bundle is rendered into the exact argv its
adapter would launch, and that argv is handed to the parser of the engine the
image installed. The report is one JSON document on stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import yaml
from sie_server.adapters.sglang import _memory, _server, generation
from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.config.model import ModelConfig
from sie_server.core.loader import load_adapter, load_model_configs

SGLANG_LAUNCHER = ("-m", "sglang.launch_server")
SGLANG_MODULE_PREFIX = "sie_server.adapters.sglang"
REPORTED_DISTRIBUTIONS = ("sglang", "torch", "transformers", "xgrammar", "flashinfer-python")
RENDER_DEVICE = "cuda:0"


class _Rendered(Exception):  # noqa: N818
    def __init__(self, argv: list[str]) -> None:
        super().__init__("rendered")
        self.argv = argv


class EngineArgumentError(ValueError):
    pass


class _EngineParser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:
        raise EngineArgumentError(message)


def bundle_adapter_modules(bundles_dir: Path, bundle: str) -> frozenset[str]:
    data = yaml.safe_load((bundles_dir / f"{bundle}.yaml").read_text(encoding="utf-8")) or {}
    return frozenset(str(module) for module in data.get("adapters") or [])


def routed_profiles(models_dir: Path, adapter_modules: frozenset[str]) -> dict[str, ModelConfig]:
    """Every routable model and profile variant whose adapter module the bundle carries."""
    configs = load_model_configs(models_dir)
    return {
        route: config
        for route, config in sorted(configs.items())
        if config.resolve_profile("default").adapter_path.partition(":")[0] in adapter_modules
    }


def is_sglang_route(config: ModelConfig) -> bool:
    return config.resolve_profile("default").adapter_path.startswith(SGLANG_MODULE_PREFIX)


def render_sglang_argv(config: ModelConfig, models_dir: Path) -> list[str]:
    """The arguments the adapter passes after ``python -m sglang.launch_server``."""
    adapter = load_adapter(config, models_dir, device=RENDER_DEVICE)

    def capture(command: list[str], **_: object) -> None:
        raise _Rendered([str(part) for part in command])

    with (
        mock.patch.object(_server, "launch_sglang_server", capture),
        mock.patch.object(generation, "_reasoning_control_token_ids", return_value={}),
        mock.patch.object(_memory, "device_memory", return_value=(80 * _memory.GIB, 80 * _memory.GIB)),
        mock.patch.object(_memory.SGLangMemoryBudget, "static_bytes", 8 * _memory.GIB),
    ):
        try:
            if isinstance(adapter, SGLangGenerationAdapter):
                # Subclass loads add work before the launch, such as fetching a
                # processor; the argv itself is built by the generation load.
                SGLangGenerationAdapter.load(adapter, RENDER_DEVICE)
            elif isinstance(adapter, SGLangEmbeddingAdapter):
                with mock.patch.object(adapter, "_resolve_eos_token", return_value="</s>"):
                    adapter.load(RENDER_DEVICE)
            else:
                msg = f"{type(adapter).__name__} is not an SGLang launch adapter"
                raise TypeError(msg)
        except _Rendered as rendered:
            command = rendered.argv
        else:
            msg = f"{config.sie_id} loaded without launching an engine"
            raise RuntimeError(msg)
    if tuple(command[1:3]) != SGLANG_LAUNCHER:
        msg = f"{config.sie_id} launches {command[1:3]!r}, not {SGLANG_LAUNCHER!r}"
        raise RuntimeError(msg)
    return command[3:]


def undeclared_flags(argv: list[str], declared: frozenset[str]) -> list[str]:
    """Option tokens the parser would only accept through prefix matching, or not at all."""
    return [
        token.partition("=")[0] for token in argv if token.startswith("--") and token.partition("=")[0] not in declared
    ]


def check_argv(parser: argparse.ArgumentParser, argv: list[str]) -> str | None:
    declared = frozenset(parser._option_string_actions)
    missing = undeclared_flags(argv, declared)
    if missing:
        return f"not declared by the engine: {', '.join(missing)}"
    try:
        parser.parse_args(argv)
    except EngineArgumentError as error:
        return str(error)
    return None


def sglang_parser() -> argparse.ArgumentParser:
    from sglang.srt.server_args import ServerArgs

    parser = _EngineParser()
    ServerArgs.add_cli_args(parser)
    return parser


def distribution_versions() -> dict[str, str]:
    versions = {}
    for name in REPORTED_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def flag_set_sha256(flags: list[str]) -> str:
    return hashlib.sha256("".join(f"{flag}\n" for flag in flags).encode()).hexdigest()


def probe(bundle: str, *, models_dir: Path, bundles_dir: Path) -> dict[str, Any]:
    routes = routed_profiles(models_dir, bundle_adapter_modules(bundles_dir, bundle))
    sglang_routes = {route: config for route, config in routes.items() if is_sglang_route(config)}
    report: dict[str, Any] = {
        "schema": 1,
        "bundle": bundle,
        "engine": None,
        "engine_version": None,
        "distributions": distribution_versions(),
        "flags": [],
        "flags_sha256": None,
        "routes": [],
    }
    if not sglang_routes:
        return report
    parser = sglang_parser()
    flags = sorted(parser._option_string_actions)
    report.update(
        engine="sglang",
        engine_version=importlib.metadata.version("sglang"),
        flags=flags,
        flags_sha256=flag_set_sha256(flags),
    )
    for route, config in sglang_routes.items():
        try:
            argv = render_sglang_argv(config, models_dir)
        except Exception as error:  # noqa: BLE001
            report["routes"].append({"route": route, "argv": None, "error": f"render failed: {error}"})
            continue
        report["routes"].append({"route": route, "argv": argv, "error": check_argv(parser, argv)})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--models-dir", type=Path, default=Path("/app/models"))
    parser.add_argument("--bundles-dir", type=Path, default=Path("/app/bundles"))
    args = parser.parse_args()
    json.dump(probe(args.bundle, models_dir=args.models_dir, bundles_dir=args.bundles_dir), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
