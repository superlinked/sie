"""document-ocr must only preload models the image it starts can load.

``sie-server serve --preload`` checks the selected profile (the default
profile for a bare id) against the image bundle. This test parses the example
compose files and ``src/config.ts`` and fails when a preloaded adapter is not
in that image's bundle, when a compose preload list drifts from the documented
set, or when a listed model id is not a catalog ``sie_id``.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE = _REPO_ROOT / "examples" / "document-ocr"
_COMPOSE_FILES = (_EXAMPLE / "compose.yml", _EXAMPLE / "compose.gpu.yml")
# GPU compose preloads only the recognition default. GLM-OCR is about 9B, so
# GLM-OCR and PaddleOCR-VL-1.5 are not preloaded with LightOnOCR; they stay in
# the UI and load on demand. The CPU compose is unchanged.
_EXPECTED_PRELOADS = {
    "compose.yml": [
        "naver-clova-ix/donut-base-finetuned-cord-v2",
        "naver-clova-ix/donut-base-finetuned-docvqa",
        "urchade/gliner_multi-v2.1",
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

# Quoted org/name strings. A path like "data/samples" has neither a digit nor
# a hyphen/underscore, which every catalog id in this example does.
_QUOTED_MODEL_ID = re.compile(r"""["']([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)["']""")
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


def _bundle_adapters(filename: str) -> set[str]:
    path = _BUNDLES_DIR / filename
    data = yaml.safe_load(path.read_text()) or {}
    adapters = data.get("adapters") or []
    if not isinstance(adapters, list):
        msg = f"{filename} adapters is not a list"
        raise AssertionError(msg)
    return {item for item in adapters if isinstance(item, str)}


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


def _bare_id_and_profile(preload_id: str) -> tuple[str, str]:
    base, separator, profile = preload_id.partition(":")
    if separator and profile and profile != "default":
        return base, profile
    return base, "default"


def _profile_adapter_module(model: dict[str, object], profile: str) -> str:
    profiles = model.get("profiles")
    if not isinstance(profiles, dict) or profile not in profiles:
        msg = f"profile {profile!r} missing from model profiles"
        raise AssertionError(msg)
    body = profiles[profile] or {}
    if not isinstance(body, dict):
        msg = f"profile {profile!r} is not a mapping"
        raise AssertionError(msg)
    adapter_path = body.get("adapter_path")
    if not isinstance(adapter_path, str) or not adapter_path:
        msg = f"profile {profile!r} has no adapter_path"
        raise AssertionError(msg)
    return adapter_path.split(":", 1)[0]


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
    models = _models_by_id()
    assert models, f"No model YAML files found in {_MODELS_DIR}"
    assert _CONFIG_TS.is_file()

    problems: list[str] = []
    for compose in _COMPOSE_FILES:
        assert compose.is_file(), compose
        image, preloads = _compose_image_and_preloads(compose)
        expected = _EXPECTED_PRELOADS[compose.name]
        if preloads != expected:
            problems.append(f"{compose.name} preloads {preloads!r}, expected {expected!r}")
        bundle_name = _bundle_filename_for_image(image)
        adapters = _bundle_adapters(bundle_name)
        assert adapters, f"{bundle_name} declares no adapters"
        for preload_id in preloads:
            bare_id, profile = _bare_id_and_profile(preload_id)
            model = models.get(bare_id)
            if model is None:
                problems.append(f"{compose.name} preloads {preload_id!r}, which has no model yaml")
                continue
            module = _profile_adapter_module(model, profile)
            if module not in adapters:
                problems.append(
                    f"{compose.name} image {image} ({bundle_name}) preloads {preload_id!r} "
                    f"via {module}, which that bundle does not list"
                )

    assert not problems, "document-ocr preloads models its image cannot load:\n  " + "\n  ".join(problems)


def test_document_ocr_config_model_ids_exist() -> None:
    models = _models_by_id()
    assert models, f"No model YAML files found in {_MODELS_DIR}"
    text = _CONFIG_TS.read_text()
    config_ids = [model_id for model_id in _QUOTED_MODEL_ID.findall(text) if _MODEL_ID_MARK.search(model_id)]
    assert len(config_ids) >= 5, "config.ts scanner found too few model ids — the pattern has rotted"

    unknown = sorted({model_id for model_id in config_ids if model_id not in models})
    assert not unknown, "document-ocr config.ts ids that are not catalog sie_ids:\n  " + "\n  ".join(unknown)
