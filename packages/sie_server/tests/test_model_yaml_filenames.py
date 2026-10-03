"""Regression tests for model YAML filenames in packages/sie_server/models/.

The model lookup contract maps a model name to
`model_name.replace("/", "__").replace(":", "__") + ".yaml"`.
On a case-sensitive filesystem (Linux CI), any case mismatch between the filename
and `sie_id` makes the lookup return `{}`, so instruction-aware callers cannot
load the model's modality metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _expected_filename(sie_id: str) -> str:
    return sie_id.replace("/", "__").replace(":", "__") + ".yaml"


@pytest.mark.parametrize("yaml_path", sorted(MODELS_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_yaml_filename_matches_sie_id(yaml_path: Path) -> None:
    with yaml_path.open() as f:
        config = yaml.safe_load(f) or {}
    sie_id = config.get("sie_id")
    assert sie_id, f"{yaml_path.name}: missing sie_id"
    assert yaml_path.name == _expected_filename(sie_id), (
        f"YAML filename {yaml_path.name!r} does not match sie_id {sie_id!r}; "
        f"expected {_expected_filename(sie_id)!r}. "
        f"Case-sensitive filesystems (Linux CI) will fail to find this config."
    )


@pytest.mark.parametrize("yaml_path", sorted(MODELS_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_instruction_template_has_placeholder(yaml_path: Path) -> None:
    """An ``Instruct:``-prefixed ``query_template`` must contain the ``{instruction}``
    placeholder.

    Otherwise instruction-aware evaluation callers treat the model as
    non-instruction-following and silently drop the per-task prompt, so the model
    is measured with a hardcoded generic instruction instead of each task's.
    """
    with yaml_path.open() as f:
        config = yaml.safe_load(f) or {}
    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        return
    offenders: list[str] = []
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        adapter_options = profile.get("adapter_options")
        runtime = adapter_options.get("runtime") if isinstance(adapter_options, dict) else None
        query_template = runtime.get("query_template") if isinstance(runtime, dict) else None
        if isinstance(query_template, str) and "Instruct:" in query_template and "{instruction}" not in query_template:
            offenders.append(profile_name)
    assert not offenders, (
        f"{yaml_path.name}: profile(s) {offenders} hardcode an 'Instruct:' instruction "
        f"without a '{{instruction}}' placeholder; instruction-aware evaluation "
        f"callers drop the per-task instruction. Use 'Instruct: {{instruction}}' + "
        f"'default_instruction'."
    )


_GREEDY_ONLY_ADAPTERS = frozenset({"sie_server.adapters.ctranslate2.generation:CTranslate2GenerationAdapter"})


@pytest.mark.parametrize("yaml_path", sorted(MODELS_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_greedy_only_backend_declares_greedy_default_sampling(yaml_path: Path) -> None:
    """A profile served by a greedy-only backend must pin ``default_sampling.temperature`` to 0.

    ``/v1/generate`` fills ``temperature=1.0`` for a request that omits it unless the
    profile declares its own sampling default, and the CTranslate2 adapter rejects any
    non-greedy request, so without the declaration every customer request that leaves
    the sampler at its default is refused with a 400.
    """
    with yaml_path.open() as f:
        config = yaml.safe_load(f) or {}
    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        return
    offenders: list[str] = []
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict) or profile.get("adapter_path") not in _GREEDY_ONLY_ADAPTERS:
            continue
        adapter_options = profile.get("adapter_options")
        runtime = adapter_options.get("runtime") if isinstance(adapter_options, dict) else None
        sampling = runtime.get("default_sampling") if isinstance(runtime, dict) else None
        temperature = sampling.get("temperature") if isinstance(sampling, dict) else None
        if not isinstance(temperature, int | float) or isinstance(temperature, bool) or float(temperature) != 0.0:
            offenders.append(profile_name)
    assert not offenders, (
        f"{yaml_path.name}: profile(s) {offenders} use a greedy-only generation backend "
        f"without 'adapter_options.runtime.default_sampling.temperature: 0.0'; a request "
        f"that omits temperature resolves to 1.0 and the backend refuses it."
    )
