# Regression guard: every model.yaml's adapter_path must be declared in some bundle.
#
# If this ever fails again we get a clear CI error instead of a silent
# "Adapter(s) not in any known bundle" warning at gateway bootstrap time
# and a stuck sie-config epoch.

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.core.loader import load_model_config

SIE_SERVER_ROOT = Path(__file__).resolve().parents[2]
BUNDLES_DIR = SIE_SERVER_ROOT / "bundles"
MODELS_DIR = SIE_SERVER_ROOT / "models"


def _load_bundle_adapters() -> dict[str, set[str]]:
    # Fail loudly on duplicate bundle names rather than silently letting the
    # later file's adapter list clobber the earlier one: a collision would
    # otherwise corrupt the union `all_declared` below and mask real drift.
    out: dict[str, set[str]] = {}
    source_file: dict[str, Path] = {}
    for bf in sorted(BUNDLES_DIR.glob("*.yaml")):
        data = yaml.safe_load(bf.read_text()) or {}
        name = data.get("name", bf.stem)
        if name in out:
            raise AssertionError(
                f"Duplicate bundle name '{name}' in {source_file[name].name} and "
                f"{bf.name}. Bundle names must be unique across packages/sie_server/bundles/*.yaml."
            )
        out[name] = set(data.get("adapters", []) or [])
        source_file[name] = bf
    return out


def _load_model_adapter_refs() -> dict[str, dict[str, str]]:
    """Return {model_file: {profile_name: adapter_module}}."""
    out: dict[str, dict[str, str]] = {}
    for mf in sorted(MODELS_DIR.glob("*.yaml")):
        data = yaml.safe_load(mf.read_text()) or {}
        profs = data.get("profiles") or {}
        per_profile: dict[str, str] = {}
        for pname, pval in profs.items():
            ap = (pval or {}).get("adapter_path") or ""
            if not ap:
                continue
            per_profile[pname] = ap.split(":", 1)[0]
        if per_profile:
            out[mf.name] = per_profile
    return out


def _load_model_generation_flags() -> dict[str, bool]:
    """Return whether each base model declares the generation primitive."""
    out: dict[str, bool] = {}
    for model_yaml in sorted(MODELS_DIR.glob("*.yaml")):
        data = yaml.safe_load(model_yaml.read_text()) or {}
        model_id = data.get("sie_id")
        if model_id:
            out[model_id] = (data.get("tasks") or {}).get("generate") is not None
    return out


def test_every_model_adapter_is_declared_in_some_bundle() -> None:
    # Guard against vacuous success: if the bundle/model directories ever
    # move (packaging-layout drift, rename, wrong pytest rootdir) the globs
    # would silently return nothing and the regression guard would stop
    # guarding. Asserting non-empty discovery up-front turns a layout break
    # into an immediate, obvious CI failure.
    assert sorted(BUNDLES_DIR.glob("*.yaml")), f"No bundle YAML files found in {BUNDLES_DIR}"
    assert sorted(MODELS_DIR.glob("*.yaml")), f"No model YAML files found in {MODELS_DIR}"

    bundles = _load_bundle_adapters()
    all_declared = {a for adapters in bundles.values() for a in adapters}

    model_refs = _load_model_adapter_refs()
    assert model_refs, (
        f"No model YAMLs declared any adapter_path under {MODELS_DIR}; the coverage "
        "guard cannot validate anything and would pass vacuously."
    )
    missing: list[str] = []
    for mfile, profiles in model_refs.items():
        for pname, module in profiles.items():
            if module not in all_declared:
                missing.append(f"{mfile}::{pname} -> {module}")

    assert not missing, (
        "Model YAMLs reference adapter modules not declared in any bundle. "
        "Either add the module to the appropriate bundle (default/sglang/"
        "transformers5) or remove the model YAML. Without a matching bundle "
        "sie-config cannot route the model and sie-gateway bootstrap stays "
        "stuck.\n\nUnrouteable references:\n  " + "\n  ".join(missing)
    )


def test_load_bundle_adapters_raises_on_duplicate_bundle_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Two YAMLs declaring the same `name:` would previously have silently
    # overwritten each other in `out[name]`, corrupting the union check.
    # Now it must fail loudly with both filenames in the error message so
    # operators can see exactly which bundles collided.
    tmp_bundles = tmp_path / "bundles"
    tmp_bundles.mkdir()
    (tmp_bundles / "a.yaml").write_text("name: shared\npriority: 10\nadapters:\n  - pkg.adapters.one\n")
    (tmp_bundles / "b.yaml").write_text("name: shared\npriority: 20\nadapters:\n  - pkg.adapters.two\n")
    import sys

    this_module = sys.modules[__name__]
    monkeypatch.setattr(this_module, "BUNDLES_DIR", tmp_bundles)
    with pytest.raises(AssertionError, match=r"Duplicate bundle name 'shared'.*a\.yaml.*b\.yaml"):
        _load_bundle_adapters()


def test_bundle_and_model_dirs_are_non_empty() -> None:
    # Companion guard for the parametrized suite below: an empty bundles/
    # directory would cause `pytest.mark.parametrize` to collect zero
    # test cases, which is a silent pass. Asserting here turns that into
    # a loud failure regardless of what the parametrized test does.
    assert sorted(BUNDLES_DIR.glob("*.yaml")), f"No bundle YAML files found in {BUNDLES_DIR}"
    assert sorted(MODELS_DIR.glob("*.yaml")), f"No model YAML files found in {MODELS_DIR}"


def test_candle_bundle_only_exposes_profile_variants() -> None:
    matches = match_bundle_models(BUNDLES_DIR / "candle.yaml", MODELS_DIR)
    assert matches, "candle bundle should expose at least one model profile variant"

    bare_model_matches = [model for model in matches if ":" not in model]
    formatted_matches = "\n  ".join(bare_model_matches)
    assert not bare_model_matches, (
        "Candle must be selected through explicit model profile variants, not bare model ids. "
        "For overlap models, keep the bare default profile on a Python adapter; for Candle-only "
        f"models, omit the default profile and expose only `model:candle`:\n  {formatted_matches}"
    )


@pytest.mark.parametrize(
    ("model_id", "transformers_bundle"),
    [
        ("lightonai/LightOnOCR-2-1B", "transformers5.yaml"),
        ("PaddlePaddle/PaddleOCR-VL-1.5", "default.yaml"),
        ("zai-org/GLM-OCR", "transformers5.yaml"),
    ],
)
def test_generative_ocr_defaults_to_isolated_sglang_extract_and_keeps_transformers_fallback(
    model_id: str,
    transformers_bundle: str,
) -> None:
    extract_models = match_bundle_models(BUNDLES_DIR / "sglang-vision-extract.yaml", MODELS_DIR)
    assert model_id in extract_models
    assert f"{model_id}:transformers" not in extract_models

    generation_models = match_bundle_models(BUNDLES_DIR / "sglang.yaml", MODELS_DIR)
    assert model_id not in generation_models

    transformers_models = match_bundle_models(BUNDLES_DIR / transformers_bundle, MODELS_DIR)
    assert model_id not in transformers_models
    assert f"{model_id}:transformers" in transformers_models


@pytest.mark.parametrize("bundle_yaml", sorted(BUNDLES_DIR.glob("*.yaml")))
def test_bundle_does_not_mix_generation_and_non_generation_models(bundle_yaml: Path) -> None:
    generation_flags = _load_model_generation_flags()
    matched = match_bundle_models(bundle_yaml, MODELS_DIR)
    generation = sorted(model_id for model_id in matched if generation_flags[model_id.split(":", 1)[0]])
    non_generation = sorted(model_id for model_id in matched if not generation_flags[model_id.split(":", 1)[0]])

    assert not (generation and non_generation), (
        f"{bundle_yaml.name} mixes generation and non-generation worker task classes; "
        f"generation={generation}, non_generation={non_generation}"
    )


def test_gte_multilingual_candle_profile_targets_rust_fp16() -> None:
    model_yaml = MODELS_DIR / "Alibaba-NLP__gte-multilingual-base.yaml"
    data = yaml.safe_load(model_yaml.read_text()) or {}
    profiles = data["profiles"]

    default = profiles["default"]
    candle = profiles["candle"]

    assert default["adapter_path"] == "sie_server.adapters.rope_flash:RoPEFlashAdapter"
    assert candle["extends"] == "default"
    assert candle["adapter_path"] == "sie_server_rust.adapters.candle:CandleEmbeddingAdapter"
    assert candle["compute_precision"] == "float16"
    assert "Alibaba-NLP/gte-multilingual-base:candle" in match_bundle_models(
        BUNDLES_DIR / "candle.yaml",
        MODELS_DIR,
    )


def test_snowflake_arctic_candle_profile_targets_rust_fp16() -> None:
    model_yaml = MODELS_DIR / "Snowflake__snowflake-arctic-embed-l-v2.0.yaml"
    data = yaml.safe_load(model_yaml.read_text()) or {}
    profiles = data["profiles"]

    default = profiles["default"]
    candle = profiles["candle"]

    assert default["adapter_path"] == "sie_server.adapters.xlm_roberta_flash:XLMRobertaFlashAdapter"
    assert default["compute_precision"] is None
    assert candle["extends"] == "default"
    assert candle["adapter_path"] == "sie_server_rust.adapters.candle:CandleEmbeddingAdapter"
    assert candle["compute_precision"] == "float16"
    assert "Snowflake/snowflake-arctic-embed-l-v2.0:candle" in match_bundle_models(
        BUNDLES_DIR / "candle.yaml",
        MODELS_DIR,
    )


def test_splade_pp_exposes_checkpoint_sequence_capacity() -> None:
    model_yaml = MODELS_DIR / "prithivida__Splade_PP_en_v2.yaml"
    data = yaml.safe_load(model_yaml.read_text()) or {}
    config = load_model_config(model_yaml)

    assert data["max_sequence_length"] == 512
    assert data["profiles"]["default"]["adapter_options"]["runtime"]["max_seq_length"] == 128
    assert data["profiles"]["sentence_transformer"]["adapter_options"]["loadtime"]["max_seq_length"] == 128
    assert data["profiles"]["candle"]["adapter_options"]["loadtime"]["max_seq_length"] == 128
    assert data["profiles"]["candle"]["adapter_options"]["runtime"]["max_seq_length"] == 128
    assert config.resolve_profile("candle").loadtime["max_seq_length"] == 128
    assert config.resolve_profile("candle").runtime["max_seq_length"] == 128


@pytest.mark.parametrize(
    "model_file",
    [
        "Snowflake__snowflake-arctic-embed-m-v2.0.yaml",
        "ibm-granite__granite-embedding-english-r2.yaml",
    ],
)
def test_cosqa_precision_sensitive_defaults_use_reference_adapter(model_file: str) -> None:
    data = yaml.safe_load((MODELS_DIR / model_file).read_text()) or {}

    assert data["profiles"]["default"]["compute_precision"] is None
    assert data["profiles"]["default"]["adapter_path"] == (
        "sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter"
    )


def test_granite_candle_profile_keeps_explicit_bfloat16_precision() -> None:
    model_yaml = MODELS_DIR / "ibm-granite__granite-embedding-english-r2.yaml"
    config = load_model_config(model_yaml)

    assert config.resolve_profile("candle").compute_precision == "bfloat16"


def test_multilingual_e5_instruct_profiles_keep_catalog_instruction() -> None:
    model_yaml = MODELS_DIR / "intfloat__multilingual-e5-large-instruct.yaml"
    data = yaml.safe_load(model_yaml.read_text()) or {}
    expected = "Given a web search query, retrieve relevant passages that answer the query"

    for profile_name in ("default", "candle"):
        runtime = data["profiles"][profile_name]["adapter_options"]["runtime"]
        assert runtime["default_instruction"] == expected
        assert "instruction" not in runtime


@pytest.mark.parametrize("bundle_yaml", sorted(BUNDLES_DIR.glob("*.yaml")))
def test_bundle_yaml_has_required_fields(bundle_yaml: Path) -> None:
    data = yaml.safe_load(bundle_yaml.read_text()) or {}
    assert data.get("name"), f"{bundle_yaml.name}: missing 'name'"
    assert isinstance(data.get("priority"), int), f"{bundle_yaml.name}: 'priority' must be int"
    adapters = data.get("adapters")
    assert isinstance(adapters, list), f"{bundle_yaml.name}: 'adapters' must be a list"
    assert adapters, f"{bundle_yaml.name}: 'adapters' must be non-empty"


# NOTE: Intentionally no "adapter appears in exactly one bundle" test.
#
# sie-config (packages/sie_config/src/sie_config/model_registry.py) and
# sie-gateway (packages/sie_gateway/src/state/model_registry.rs) both
# treat multi-bundle adapter membership as a supported state: they
# collect every bundle whose `adapters` list overlaps the model and
# pick the default route by priority. Existing registry tests (e.g.
# test_model_multiple_profiles_different_adapters in test_model_registry.py)
# rely on the same model being compatible with both `default` and
# `sglang`.
#
# Forbidding duplicates here would block legitimate future setups where
# the same adapter is intentionally available in multiple dependency
# stacks (e.g. a CPU bundle and a CUDA bundle that both ship the same
# sentence-transformer adapter). If equal-priority resolution ever
# needs to be fully deterministic in sie-config too, the fix is a
# registry-level secondary sort key, not a CI lint on bundle contents.
