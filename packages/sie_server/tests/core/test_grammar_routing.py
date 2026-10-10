"""Tests for the worker-side grammar-profile admission helper."""

from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from sie_server.adapters._generation_base import GenerationUnsupportedFieldError
from sie_server.core.grammar_routing import resolve_grammar_serving_model
from sie_server.core.registry import ModelRegistry

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_CATALOG = ("Qwen__Qwen3.5-4B.yaml", "Qwen__Qwen3.6-27B.yaml", "sie-fake.yaml")
_GEMMA_FILE = _MODELS_DIR / "google__gemma-4-31B-it.yaml"
_GEMMA_ID = "google/gemma-4-31B-it"
_GEMMA_THINKING = "h100-96k-hires-thinking-no-spec"


def _registry(tmp_path: Path, model_filter: list[str] | None = None) -> ModelRegistry:
    for name in _CATALOG:
        shutil.copy(_MODELS_DIR / name, tmp_path / name)
    return ModelRegistry(models_dir=tmp_path, model_filter=model_filter, enable_hot_reload=False)


def _gemma_registry(
    tmp_path: Path, *, config: dict[str, Any] | None = None, model_filter: list[str] | None = None
) -> ModelRegistry:
    destination = tmp_path / _GEMMA_FILE.name
    if config is None:
        shutil.copy(_GEMMA_FILE, destination)
    else:
        destination.write_text(yaml.safe_dump(config), encoding="utf-8")
    return ModelRegistry(models_dir=tmp_path, model_filter=model_filter, enable_hot_reload=False)


@pytest.mark.parametrize(
    ("requested", "serving"),
    [
        ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B:no-spec"),
        ("Qwen/Qwen3.5-4B:a100-40gb", "Qwen/Qwen3.5-4B:no-spec"),
        ("Qwen/Qwen3.5-4B:long-context", "Qwen/Qwen3.5-4B:no-spec"),
        ("Qwen/Qwen3.5-4B:no-spec", "Qwen/Qwen3.5-4B:no-spec"),
        ("Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-27B"),
        ("Qwen/Qwen3.6-27B:long-context", "Qwen/Qwen3.6-27B:long-context-no-spec"),
        ("Qwen/Qwen3.6-27B:long-context-thinking", "Qwen/Qwen3.6-27B:long-context-thinking-no-spec"),
        ("Qwen/Qwen3.6-27B:long-context-no-spec", "Qwen/Qwen3.6-27B:long-context-no-spec"),
        ("sie-fake", "sie-fake"),
        ("org/unknown", "org/unknown"),
    ],
)
def test_grammar_requests_resolve_to_the_grammar_safe_profile(tmp_path: Path, requested: str, serving: str) -> None:
    assert resolve_grammar_serving_model(_registry(tmp_path), requested) == serving


def test_missing_grammar_profile_is_rejected_instead_of_served_speculatively(tmp_path: Path) -> None:
    registry = _registry(tmp_path, model_filter=["Qwen/Qwen3.5-4B"])

    with pytest.raises(GenerationUnsupportedFieldError) as exc_info:
        resolve_grammar_serving_model(registry, "Qwen/Qwen3.5-4B")

    assert exc_info.value.param == "grammar"
    assert "'no-spec'" in str(exc_info.value)


def test_variant_served_without_its_base_keeps_a_grammar_safe_launch(tmp_path: Path) -> None:
    registry = _registry(tmp_path, model_filter=["Qwen/Qwen3.6-27B:long-context-no-spec"])

    assert (
        resolve_grammar_serving_model(registry, "Qwen/Qwen3.6-27B:long-context-no-spec")
        == "Qwen/Qwen3.6-27B:long-context-no-spec"
    )


def test_speculative_variant_served_without_its_base_needs_its_grammar_profile(tmp_path: Path) -> None:
    registry = _registry(tmp_path, model_filter=["Qwen/Qwen3.6-27B:long-context"])

    with pytest.raises(GenerationUnsupportedFieldError, match="'long-context-no-spec'"):
        resolve_grammar_serving_model(registry, "Qwen/Qwen3.6-27B:long-context")

    both = _registry(tmp_path, model_filter=["Qwen/Qwen3.6-27B:long-context", "Qwen/Qwen3.6-27B:long-context-no-spec"])
    assert (
        resolve_grammar_serving_model(both, "Qwen/Qwen3.6-27B:long-context") == "Qwen/Qwen3.6-27B:long-context-no-spec"
    )


@pytest.mark.parametrize(
    "model_filter",
    [None, [_GEMMA_ID, f"{_GEMMA_ID}:{_GEMMA_THINKING}"], [f"{_GEMMA_ID}:{_GEMMA_THINKING}"]],
)
def test_hires_thinking_grammar_requests_keep_the_requested_generation_contract(
    tmp_path: Path, model_filter: list[str] | None
) -> None:
    registry = _gemma_registry(tmp_path, model_filter=model_filter)
    requested = f"{_GEMMA_ID}:{_GEMMA_THINKING}"

    serving = resolve_grammar_serving_model(registry, requested)

    assert serving == requested
    generate = registry.get_config(serving).tasks.generate
    assert generate is not None
    assert generate.max_output_tokens == 32768
    assert generate.context_length == 98304
    assert generate.chat_template_kwargs == {"enable_thinking": True}


def test_all_existing_gemma_grammar_routes_are_preserved(tmp_path: Path) -> None:
    registry = _gemma_registry(tmp_path)
    expected = {
        "": "no-spec",
        "no-spec": "no-spec",
        "h100-fp8": "no-spec",
        "thinking": "h100-96k-thinking-no-spec",
        "long-context": "long-context-no-spec",
        "long-context-no-spec": "long-context-no-spec",
        "long-context-thinking": "long-context-thinking-no-spec",
        "long-context-thinking-no-spec": "long-context-thinking-no-spec",
        "h200-256k": "long-context-no-spec",
        "h200-256k-thinking": "long-context-thinking-no-spec",
        "h100-96k": "h100-96k-no-spec",
        "h100-96k-no-spec": "h100-96k-no-spec",
        "h100-96k-thinking": "h100-96k-thinking-no-spec",
        "h100-96k-thinking-no-spec": "h100-96k-thinking-no-spec",
        "h100-96k-hires": "h100-96k-hires-no-spec",
        "h100-96k-hires-no-spec": "h100-96k-hires-no-spec",
        "h100-96k-hires-out8k": "h100-96k-hires-out8k-no-spec",
        "h100-96k-hires-out8k-no-spec": "h100-96k-hires-out8k-no-spec",
    }
    for profile, serving_profile in expected.items():
        requested = _GEMMA_ID if not profile else f"{_GEMMA_ID}:{profile}"
        assert resolve_grammar_serving_model(registry, requested) == f"{_GEMMA_ID}:{serving_profile}", profile


@pytest.mark.parametrize("serve_fallback", [True, False])
@pytest.mark.parametrize(
    ("profile_updates", "loadtime_updates", "extra_args"),
    [
        ({"adapter_path": "sie_server.adapters.fake.adapter:FakeAdapter"}, {}, []),
        ({}, {"grammar_backend": "outlines"}, []),
        ({}, {"speculative": {"enabled": True}}, []),
        ({}, {"speculative": {"enabled": "false"}}, []),
        ({}, {"speculative": {}}, []),
        ({}, {}, ["--speculative-algo", "NEXTN"]),
        ({}, {}, ["--speculative-algo=NEXTN"]),
        ({}, {}, ["--enable-multi-layer-eagle"]),
        ({}, {}, ["--config", "overrides.json"]),
        ({}, {}, ["--grammar-backend", "outlines"]),
        ({}, {}, ["--log-level", "debug"]),
    ],
)
def test_parent_grammar_hint_does_not_admit_an_incompatible_child(
    tmp_path: Path,
    profile_updates: dict[str, Any],
    loadtime_updates: dict[str, Any],
    extra_args: list[str],
    serve_fallback: bool,
) -> None:
    config = yaml.safe_load(_GEMMA_FILE.read_text(encoding="utf-8"))
    profile = config["profiles"][_GEMMA_THINKING]
    profile.update(profile_updates)
    loadtime = profile["adapter_options"]["loadtime"]
    loadtime.update(loadtime_updates)
    loadtime["extra_launch_args"].extend(extra_args)
    requested = f"{_GEMMA_ID}:{_GEMMA_THINKING}"
    model_filter = [_GEMMA_ID, requested]
    if serve_fallback:
        model_filter.append(f"{_GEMMA_ID}:no-spec")
    registry = _gemma_registry(tmp_path, config=config, model_filter=model_filter)

    if serve_fallback:
        assert resolve_grammar_serving_model(registry, requested) == f"{_GEMMA_ID}:no-spec"
    else:
        with pytest.raises(GenerationUnsupportedFieldError, match="'no-spec'"):
            resolve_grammar_serving_model(registry, requested)


@pytest.mark.parametrize("serve_fallback", [True, False])
def test_explicit_child_grammar_fallback_precedes_the_parent_hint(tmp_path: Path, serve_fallback: bool) -> None:
    config = yaml.safe_load(_GEMMA_FILE.read_text(encoding="utf-8"))
    profile = config["profiles"][_GEMMA_THINKING]
    fallback = "hires-thinking-scoped-safe"
    config["profiles"][fallback] = deepcopy(profile)
    profile["grammar_profile"] = fallback
    requested = f"{_GEMMA_ID}:{_GEMMA_THINKING}"
    model_filter = [_GEMMA_ID, requested]
    if serve_fallback:
        model_filter.append(f"{_GEMMA_ID}:{fallback}")
    registry = _gemma_registry(tmp_path, config=config, model_filter=model_filter)

    if serve_fallback:
        assert resolve_grammar_serving_model(registry, requested) == f"{_GEMMA_ID}:{fallback}"
    else:
        with pytest.raises(GenerationUnsupportedFieldError, match=f"'{fallback}'"):
            resolve_grammar_serving_model(registry, requested)
