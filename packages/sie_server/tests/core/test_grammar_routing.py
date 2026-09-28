"""Tests for the worker-side grammar-profile admission helper."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from sie_server.adapters._generation_base import GenerationUnsupportedFieldError
from sie_server.core.grammar_routing import resolve_grammar_serving_model
from sie_server.core.registry import ModelRegistry

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_CATALOG = ("Qwen__Qwen3.5-4B.yaml", "Qwen__Qwen3.6-27B.yaml", "sie-fake.yaml")


def _registry(tmp_path: Path, model_filter: list[str] | None = None) -> ModelRegistry:
    for name in _CATALOG:
        shutil.copy(_MODELS_DIR / name, tmp_path / name)
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
