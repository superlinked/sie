from __future__ import annotations

from types import SimpleNamespace

from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.processors.streaming import _VISION_TOKENS_PER_IMAGE_ESTIMATE, _vision_tokens_for_images

_TODAY = ["--mm-process-config", '{"image":{"min_pixels":65536,"max_pixels":1003520}}']


def _adapter(args: list[str], env: dict[str, str] | None = None) -> SGLangGenerationAdapter:
    adapter = SGLangGenerationAdapter.__new__(SGLangGenerationAdapter)
    adapter._extra_launch_args = args
    adapter._extra_env = env or {}
    return adapter


def test_single_image_budget_follows_the_raised_bound() -> None:
    adapter = _adapter(_TODAY, {"SIE_SGLANG_SINGLE_IMAGE_MAX_PIXELS": "3211264"})
    assert adapter.image_token_budget == 3136
    assert adapter.multi_image_token_budget == 980


def test_without_the_raised_bound_both_budgets_are_the_launch_bound() -> None:
    adapter = _adapter(_TODAY)
    assert adapter.image_token_budget == adapter.multi_image_token_budget == 980


def test_budget_is_absent_without_an_image_bound() -> None:
    assert _adapter([]).image_token_budget is None
    assert _adapter(["--mm-process-config"]).multi_image_token_budget is None
    assert _adapter(["--mm-process-config", "not json"]).multi_image_token_budget is None
    assert _adapter(["--mm-process-config", '{"video":{"total_pixels":8388608}}']).multi_image_token_budget is None
    assert _adapter([], {"SIE_SGLANG_SINGLE_IMAGE_MAX_PIXELS": "junk"}).image_token_budget is None


def test_worker_reserves_the_raised_bound_only_for_one_image() -> None:
    adapter = _adapter(_TODAY, {"SIE_SGLANG_SINGLE_IMAGE_MAX_PIXELS": "3211264"})
    assert _vision_tokens_for_images(adapter, 0) == 0
    assert _vision_tokens_for_images(adapter, 1) == 3136
    # Two or more images reserve exactly what they reserved before the raise.
    assert _vision_tokens_for_images(adapter, 2) == 2 * _VISION_TOKENS_PER_IMAGE_ESTIMATE
    assert _vision_tokens_for_images(adapter, 4) == 4 * _VISION_TOKENS_PER_IMAGE_ESTIMATE


def test_worker_never_reserves_below_the_family_estimate() -> None:
    assert _vision_tokens_for_images(SimpleNamespace(), 1) == _VISION_TOKENS_PER_IMAGE_ESTIMATE
    assert _vision_tokens_for_images(SimpleNamespace(image_token_budget=980), 1) == _VISION_TOKENS_PER_IMAGE_ESTIMATE
    assert _vision_tokens_for_images(SimpleNamespace(image_token_budget=True), 1) == _VISION_TOKENS_PER_IMAGE_ESTIMATE
