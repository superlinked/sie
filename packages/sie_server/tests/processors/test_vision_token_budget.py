from __future__ import annotations

from types import SimpleNamespace

from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.processors.streaming import _VISION_TOKENS_PER_IMAGE_ESTIMATE, _vision_tokens_per_image


def _adapter(args: list[str]) -> SGLangGenerationAdapter:
    adapter = SGLangGenerationAdapter.__new__(SGLangGenerationAdapter)
    adapter._extra_launch_args = args
    return adapter


def test_budget_follows_the_launch_image_bound() -> None:
    assert (
        _adapter(["--mm-process-config", '{"image":{"min_pixels":65536,"max_pixels":3211264}}']).image_token_budget
        == 3136
    )
    assert (
        _adapter(["--mm-process-config", '{"image":{"min_pixels":65536,"max_pixels":1003520}}']).image_token_budget
        == 980
    )


def test_budget_is_absent_without_an_image_bound() -> None:
    assert _adapter([]).image_token_budget is None
    assert _adapter(["--mm-process-config"]).image_token_budget is None
    assert _adapter(["--mm-process-config", "not json"]).image_token_budget is None
    assert _adapter(["--mm-process-config", '{"video":{"total_pixels":8388608}}']).image_token_budget is None


def test_worker_reserves_the_larger_of_the_launch_bound_and_the_family_estimate() -> None:
    assert _vision_tokens_per_image(SimpleNamespace(image_token_budget=3136)) == 3136
    # A lower-resolution launch keeps today's conservative reservation.
    assert _vision_tokens_per_image(SimpleNamespace(image_token_budget=980)) == _VISION_TOKENS_PER_IMAGE_ESTIMATE
    assert _vision_tokens_per_image(SimpleNamespace()) == _VISION_TOKENS_PER_IMAGE_ESTIMATE
    assert _vision_tokens_per_image(SimpleNamespace(image_token_budget=True)) == _VISION_TOKENS_PER_IMAGE_ESTIMATE
