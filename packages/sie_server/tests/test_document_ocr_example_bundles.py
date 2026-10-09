from __future__ import annotations

import re
from pathlib import Path

import yaml
from sie_sdk.bundle_utils import match_bundle_models

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE = _REPO_ROOT / "examples" / "document-ocr"
_COMPOSE_FILES = (_EXAMPLE / "compose.yml", _EXAMPLE / "compose.gpu.yml")
_EXPECTED_PRELOADS = {
    "compose.yml": [
        "naver-clova-ix/donut-base-finetuned-cord-v2",
        "naver-clova-ix/donut-base-finetuned-docvqa",
        "urchade/gliner_multi-v2.1",
        "PaddlePaddle/PaddleOCR-VL-1.5:transformers",
    ],
    "compose.gpu.yml": ["lightonai/LightOnOCR-2-1B"],
}
_CONFIG_TS = _EXAMPLE / "src" / "config.ts"
_BUNDLES_DIR = _REPO_ROOT / "packages" / "sie_server" / "bundles"
_MODELS_DIR = _REPO_ROOT / "packages" / "sie_server" / "models"

# Longest suffix first so a tag cannot match a shorter sibling by accident.
_IMAGE_BUNDLE_SUFFIXES = (
    ("-sglang-vision-extract", "sglang-vision-extract.yaml"),
    ("-transformers5", "transformers5.yaml"),
    ("-default", "default.yaml"),
)

# Quoted org/name ids, optionally ``model:profile``. A path like "data/samples"
# has neither a digit nor a hyphen/underscore, which every catalog id here does.
_QUOTED_MODEL_ID = re.compile(
    r"""["']([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*(?::[A-Za-z0-9][A-Za-z0-9_.-]*)?)["']"""
)
_MODEL_ID_MARK = re.compile(r"[-_0-9]")


def _models_by_id() -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for path in sorted(_MODELS_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            continue
        sie_id = data.get("sie_id")
        if isinstance(sie_id, str):
            found[sie_id] = data
    return found


def _bundle_filename_for_image(image: str) -> str:
    tag = image.rsplit(":", 1)[-1]
    for suffix, filename in _IMAGE_BUNDLE_SUFFIXES:
        if tag.endswith(suffix):
            return filename
    msg = f"no bundle mapping for image {image!r}"
    raise AssertionError(msg)


def _preload_ids(command: object) -> list[str]:
    if isinstance(command, str):
        parts = command.split()
    elif isinstance(command, list):
        parts = [str(item) for item in command]
    else:
        return []
    found: list[str] = []
    for index, part in enumerate(parts):
        if part == "--preload" and index + 1 < len(parts):
            found.extend(item.strip() for item in parts[index + 1].split(",") if item.strip())
        elif part.startswith("--preload="):
            found.extend(item.strip() for item in part.split("=", 1)[1].split(",") if item.strip())
    return found


def _bundle_model_ids(bundle_filename: str) -> set[str]:
    return set(match_bundle_models(_BUNDLES_DIR / bundle_filename, _MODELS_DIR))


def _compose_image_and_preloads(path: Path) -> tuple[str, list[str]]:
    data = yaml.safe_load(path.read_text()) or {}
    services = data.get("services") if isinstance(data, dict) else None
    service = services.get("sie") if isinstance(services, dict) else None
    if not isinstance(service, dict):
        msg = f"{path} has no services.sie"
        raise AssertionError(msg)
    image = service.get("image")
    if not isinstance(image, str) or not image:
        msg = f"{path} has no image"
        raise AssertionError(msg)
    preloads = _preload_ids(service.get("command"))
    if not preloads:
        msg = f"{path} preloads no models"
        raise AssertionError(msg)
    return image, preloads


def test_document_ocr_preloaded_models_match_image_bundles() -> None:
    assert _CONFIG_TS.is_file()
    config_text = _CONFIG_TS.read_text()

    problems: list[str] = []
    for compose in _COMPOSE_FILES:
        assert compose.is_file(), compose
        image, preloads = _compose_image_and_preloads(compose)
        expected = _EXPECTED_PRELOADS[compose.name]
        if preloads != expected:
            problems.append(f"{compose.name} preloads {preloads!r}, expected {expected!r}")
        bundle_name = _bundle_filename_for_image(image)
        matched = _bundle_model_ids(bundle_name)
        assert matched, f"{bundle_name} matches no models"
        for preload_id in preloads:
            if preload_id not in config_text:
                problems.append(f"{compose.name} preloads {preload_id!r}, which src/config.ts does not list")
            if preload_id not in matched:
                problems.append(
                    f"{compose.name} image {image} ({bundle_name}) preloads {preload_id!r}, "
                    "which match_bundle_models does not return for that bundle"
                )

    assert not problems, "document-ocr preloads models its image cannot load:\n  " + "\n  ".join(problems)


def test_document_ocr_config_model_ids_exist() -> None:
    models = _models_by_id()
    assert models, f"No model YAML files found in {_MODELS_DIR}"
    served: set[str] = set()
    for compose in _COMPOSE_FILES:
        image, _preloads = _compose_image_and_preloads(compose)
        served.update(_bundle_model_ids(_bundle_filename_for_image(image)))
    text = _CONFIG_TS.read_text()
    config_ids = [model_id for model_id in _QUOTED_MODEL_ID.findall(text) if _MODEL_ID_MARK.search(model_id)]
    assert len(config_ids) >= 5, "config.ts scanner found too few model ids — the pattern has rotted"

    unknown: list[str] = []
    for model_id in sorted(set(config_ids)):
        base, separator, profile = model_id.partition(":")
        if separator and profile and profile != "default":
            if base not in models or model_id not in served:
                unknown.append(model_id)
        elif model_id not in models:
            unknown.append(model_id)
    assert not unknown, (
        "document-ocr config.ts ids that are not catalog sie_ids "
        "or bundle-matched profile ids:\n  " + "\n  ".join(unknown)
    )
