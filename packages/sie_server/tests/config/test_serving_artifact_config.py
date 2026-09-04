from __future__ import annotations

import pytest
from pydantic import ValidationError
from sie_server.config.engine import ComputePrecision
from sie_server.config.model import AdapterOptions, ModelConfig, ProfileConfig, Tasks
from sie_server.core.loader import expand_profile_variants

SOURCE_REVISION = "a" * 40


def _declaration(
    digest: str,
    *,
    revision: str = "b" * 40,
    compute_type: str = "bfloat16",
) -> dict[str, str]:
    return {
        "format": "ctranslate2",
        "repo_id": "superlinked/derived-model",
        "revision": revision,
        "manifest_path": "sie-serving-artifact.json",
        "manifest_sha256": digest,
        "compute_type": compute_type,
    }


def _profile(
    *,
    loadtime: dict[str, object],
    extends: str | None = None,
    compute_precision: ComputePrecision | None = "bfloat16",
) -> ProfileConfig:
    return ProfileConfig(
        extends=extends,
        adapter_path=None if extends else "test.adapter:Adapter",
        max_batch_tokens=None if extends else 128,
        compute_precision=None if extends else compute_precision,
        adapter_options=AdapterOptions(loadtime=loadtime),
    )


def test_requires_immutable_hf_source_identity() -> None:
    with pytest.raises(ValidationError, match="immutable 40-char hf_revision"):
        ModelConfig(
            sie_id="google/source-model",
            hf_id="google/source-model",
            hf_revision="main",
            tasks=Tasks(),
            profiles={"default": _profile(loadtime={"serving_artifact": _declaration("c" * 64)})},
        )


@pytest.mark.parametrize(
    "loader_field",
    [
        "artifact_path",
        "ct2_compute_type",
    ],
)
def test_rejects_loader_owned_catalog_fields(loader_field: str) -> None:
    with pytest.raises(ValidationError, match="loader-owned"):
        _profile(loadtime={loader_field: "operator-injected"})


def test_effective_profiles_may_select_distinct_immutable_artifacts() -> None:
    default_declaration = _declaration("c" * 64)
    quantized_declaration = _declaration("d" * 64, revision="e" * 40, compute_type="int8_bfloat16")
    config = ModelConfig(
        sie_id="google/source-model",
        hf_id="google/source-model",
        hf_revision=SOURCE_REVISION,
        tasks=Tasks(),
        profiles={
            "default": _profile(loadtime={"serving_artifact": default_declaration}),
            "inherited": _profile(loadtime={}, extends="default"),
            "quantized": _profile(loadtime={"serving_artifact": quantized_declaration}),
        },
    )

    assert config.serving_artifact_declaration("default").manifest_sha256 == "c" * 64
    assert config.serving_artifact_declaration("inherited").manifest_sha256 == "c" * 64
    assert config.serving_artifact_declaration("quantized").manifest_sha256 == "d" * 64
    assert config.serving_artifact_declaration("quantized").compute_type == "int8_bfloat16"

    expanded = expand_profile_variants([config])
    assert expanded["google/source-model:inherited"].serving_artifact_declaration().manifest_sha256 == "c" * 64
    assert expanded["google/source-model:quantized"].serving_artifact_declaration().manifest_sha256 == "d" * 64
    assert expanded["google/source-model:quantized"].serving_artifact_declaration().compute_type == "int8_bfloat16"


def test_closed_nested_declaration_rejects_unknown_fields() -> None:
    declaration = _declaration("c" * 64)
    declaration["filesystem_root"] = "/operator/path"

    with pytest.raises(ValidationError, match="serving_artifact is invalid"):
        _profile(loadtime={"serving_artifact": declaration})


def test_explicit_null_serving_artifact_is_not_treated_as_absent() -> None:
    with pytest.raises(ValidationError, match="serving_artifact is invalid"):
        _profile(loadtime={"serving_artifact": None})


@pytest.mark.parametrize(
    ("compute_type", "compute_precision"),
    [
        ("bfloat16", "float16"),
        ("float16", "float32"),
        ("float32", "bfloat16"),
        ("int8_bfloat16", "float16"),
        ("int8_float16", "bfloat16"),
        ("int8_float32", "float16"),
    ],
)
def test_rejects_profile_precision_that_misrepresents_artifact_compute(
    compute_type: str,
    compute_precision: ComputePrecision,
) -> None:
    profile = _profile(
        loadtime={"serving_artifact": _declaration("c" * 64, compute_type=compute_type)},
        compute_precision=compute_precision,
    )

    with pytest.raises(ValidationError, match="does not match serving artifact compute_type"):
        ModelConfig(
            sie_id="google/source-model",
            hf_id="google/source-model",
            hf_revision=SOURCE_REVISION,
            tasks=Tasks(),
            profiles={"default": profile},
        )


def test_bare_int8_artifact_does_not_guess_runtime_float_precision() -> None:
    profile = _profile(
        loadtime={"serving_artifact": _declaration("c" * 64, compute_type="int8")},
        compute_precision="float16",
    )

    config = ModelConfig(
        sie_id="google/source-model",
        hf_id="google/source-model",
        hf_revision=SOURCE_REVISION,
        tasks=Tasks(),
        profiles={"default": profile},
    )

    assert config.resolve_profile("default").compute_precision == "float16"


def test_omitted_profile_precision_does_not_invent_artifact_conflict() -> None:
    profile = _profile(
        loadtime={"serving_artifact": _declaration("c" * 64, compute_type="bfloat16")},
        compute_precision=None,
    )

    config = ModelConfig(
        sie_id="google/source-model",
        hf_id="google/source-model",
        hf_revision=SOURCE_REVISION,
        tasks=Tasks(),
        profiles={"default": profile},
    )

    assert config.resolve_profile("default").compute_precision is None
