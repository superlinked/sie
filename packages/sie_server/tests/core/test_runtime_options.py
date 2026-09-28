"""Unit tests for core.runtime_options.merge_runtime_options.

This is the single merge used by BOTH the single-server HTTP path
(api.options.resolve_runtime_options) and the cluster queue worker
(queue_executor.process_encode_batch). Regression coverage for #1489: the
worker path historically forwarded raw SDK options and dropped profile
``adapter_options.runtime`` defaults (query_template / default_instruction /
pooling / normalize) for every queued request.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from sie_server.adapters._generation_base import GenerationChunk
from sie_server.adapters.fake.adapter import FakeAdapter
from sie_server.config.model import ModelConfig
from sie_server.core import runtime_options
from sie_server.core.encode_pipeline import resolve_encode_output_types
from sie_server.core.runtime_options import (
    GenerationTimeoutError,
    GenerationTimeouts,
    apply_generation_runtime_options,
    bound_generation,
    merge_runtime_options,
    merge_runtime_options_with_profile,
    resolve_generation_timeouts,
)
from sie_server.types.inputs import InvalidInputError


def _embedder_config() -> ModelConfig:
    """An instruction-tuned embedder whose prompt lives in profile runtime."""
    return ModelConfig.model_validate(
        {
            "sie_id": "test/instruct-embedder",
            "hf_id": "test/instruct-embedder",
            "inputs": {"text": True},
            "tasks": {"encode": {"dense": {"dim": 8}}},
            "max_sequence_length": 512,
            "profiles": {
                "default": {
                    "max_batch_tokens": 8192,
                    "adapter_path": "sie_server.adapters.sglang.embedding:SGLangEmbeddingAdapter",
                    "adapter_options": {
                        "runtime": {
                            "pooling": "last_token",
                            "normalize": True,
                            "query_template": "Instruct: {instruction}\nQuery: {text}",
                            "default_instruction": "Given a query, retrieve relevant passages",
                        },
                    },
                },
                "alt": {
                    "max_batch_tokens": 8192,
                    "adapter_path": "sie_server.adapters.sglang.embedding:SGLangEmbeddingAdapter",
                    "adapter_options": {"runtime": {"query_template": "alt: {text}"}},
                },
            },
        }
    )


def test_merges_profile_runtime_under_request_options() -> None:
    """Profile runtime defaults appear even when the request only sends is_query."""
    config = _embedder_config()
    merged = merge_runtime_options(config, {"is_query": True})

    assert merged["query_template"] == "Instruct: {instruction}\nQuery: {text}"
    assert merged["default_instruction"] == "Given a query, retrieve relevant passages"
    assert merged["pooling"] == "last_token"
    assert merged["normalize"] is True
    # Request-supplied key is preserved alongside the merged defaults.
    assert merged["is_query"] is True


def test_request_options_win_over_runtime_defaults() -> None:
    """Per-request overrides take precedence over profile runtime defaults."""
    config = _embedder_config()
    merged = merge_runtime_options(config, {"query_template": "custom: {text}", "normalize": False})

    assert merged["query_template"] == "custom: {text}"
    assert merged["normalize"] is False


def test_none_request_options_yield_profile_defaults() -> None:
    """An empty/None request still receives the profile's runtime defaults."""
    config = _embedder_config()
    merged = merge_runtime_options(config, None)

    assert merged["query_template"] == "Instruct: {instruction}\nQuery: {text}"
    assert merged["default_instruction"] == "Given a query, retrieve relevant passages"


def test_profile_key_selects_profile_and_is_consumed() -> None:
    """The 'profile' key chooses the profile and is not forwarded to the adapter."""
    config = _embedder_config()
    merged = merge_runtime_options(config, {"profile": "alt", "is_query": True})

    assert merged["query_template"] == "alt: {text}"
    assert "profile" not in merged
    assert merged["is_query"] is True


def test_merge_returns_the_selected_profile_with_options() -> None:
    config = _embedder_config()

    merged, selected_profile = merge_runtime_options_with_profile(config, {"profile": "alt"})

    assert merged["query_template"] == "alt: {text}"
    assert selected_profile.runtime["query_template"] == "alt: {text}"


def test_encode_output_type_lists_are_independent() -> None:
    config = _embedder_config()
    effective_options, selected_profile = merge_runtime_options_with_profile(config, None)

    adapter_output_types, response_output_types = resolve_encode_output_types(
        config,
        ["dense"],
        selected_profile,
        effective_options,
    )
    adapter_output_types.append("sparse")

    assert response_output_types == ["dense"]


def test_profile_output_types_restrict_model_wide_capabilities() -> None:
    config = ModelConfig.model_validate(
        {
            "sie_id": "test/multi-output",
            "hf_id": "test/multi-output",
            "inputs": {"text": True},
            "tasks": {"encode": {"dense": {"dim": 8}, "sparse": {"dim": 32}}},
            "max_sequence_length": 512,
            "profiles": {
                "default": {
                    "max_batch_tokens": 8192,
                    "adapter_path": "sie_server.adapters.fake.adapter:FakeAdapter",
                    "adapter_options": {"runtime": {"output_types": ["sparse"]}},
                },
            },
        }
    )
    effective_options, selected_profile = merge_runtime_options_with_profile(
        config,
        {"output_types": ["dense"]},
    )

    with pytest.raises(InvalidInputError, match="does not support output types"):
        resolve_encode_output_types(
            config,
            ["dense"],
            selected_profile,
            effective_options,
        )


def test_unknown_profile_raises_invalid_input() -> None:
    """An unknown caller-selected profile is classified as invalid input."""
    config = _embedder_config()
    with pytest.raises(InvalidInputError, match="nope"):
        merge_runtime_options(config, {"profile": "nope"})


@pytest.mark.parametrize("profile", ["", " ", [], {}, 0, False, ["alt"], {"name": "alt"}])
def test_malformed_profile_selector_raises_invalid_input(profile: object) -> None:
    config = _embedder_config()

    with pytest.raises(InvalidInputError, match=r"options\.profile"):
        merge_runtime_options(config, {"profile": profile})


@pytest.mark.parametrize("policy", [[], ["truncate_text"], {}, {"a": 1}, 0, True, "drop", ""])
def test_invalid_overflow_policy_raises_invalid_input_on_both_ingress_paths(policy: object) -> None:
    """The queue worker merges options here too, so an invalid policy is a 400, not an inference error."""
    config = _embedder_config()

    with pytest.raises(InvalidInputError, match="Invalid overflow_policy"):
        merge_runtime_options(config, {"overflow_policy": policy})


@pytest.mark.parametrize("policy", ["default", "truncate_text", "error", None])
def test_valid_overflow_policies_pass_through(policy: str | None) -> None:
    merged = merge_runtime_options(_embedder_config(), {"overflow_policy": policy})
    assert merged["overflow_policy"] == policy


def _generation_config() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "sie_id": "test/generator",
            "hf_id": "test/generator",
            "inputs": {"text": True},
            "tasks": {"generate": {"context_length": 4096, "max_output_tokens": 512}},
            "max_sequence_length": 4096,
            "profiles": {
                "default": {
                    "max_batch_tokens": 4096,
                    "kv_budget_tokens": 2048,
                    "adapter_path": "sie_server.adapters.fake.adapter:FakeAdapter",
                    "adapter_options": {
                        "runtime": {
                            "default_sampling": {"temperature": 0.7, "top_p": 0.8},
                            "stop_tokens": ["</s>"],
                            "overall_timeout_s": 60,
                        }
                    },
                }
            },
        }
    )


def test_generation_runtime_defaults_apply_below_typed_fields() -> None:
    resolved = apply_generation_runtime_options(
        _generation_config(),
        {"profile": "default"},
        {"prompt": "hi", "temperature": 1.0, "stop": ["DONE"]},
    )

    assert resolved["temperature"] == 1.0
    assert resolved["top_p"] == 0.8
    assert resolved["stop"] == ["DONE", "</s>"]
    assert "profile" not in resolved


def test_generation_request_runtime_overrides_profile_defaults() -> None:
    resolved = apply_generation_runtime_options(
        _generation_config(),
        {"default_sampling": {"temperature": 0.2}},
        {"prompt": "hi"},
    )

    assert resolved["temperature"] == 0.2
    assert resolved["top_p"] == 0.8


def test_generation_frequency_penalty_and_seed_defaults_apply() -> None:
    config = _generation_config()
    config.profiles["default"].adapter_options.runtime["default_sampling"] |= {
        "frequency_penalty": 0.5,
        "seed": -(1 << 63),
    }

    resolved = apply_generation_runtime_options(config, None, {"prompt": "hi"})

    assert resolved["frequency_penalty"] == 0.5
    assert resolved["seed"] == -(1 << 63)


def test_generation_profile_default_min_new_tokens_caps_to_explicit_max() -> None:
    config = _generation_config()
    config.profiles["default"].adapter_options.runtime["default_sampling"]["min_new_tokens"] = 10

    resolved = apply_generation_runtime_options(
        config,
        None,
        {"prompt": "hi", "max_new_tokens": 1},
    )

    assert resolved["max_new_tokens"] == 1
    assert resolved["min_tokens"] == 1


def test_generation_request_sampling_min_new_tokens_above_max_fails() -> None:
    with pytest.raises(
        ValueError,
        match=r"options\.default_sampling\.min_new_tokens.*must not exceed max_new_tokens",
    ):
        apply_generation_runtime_options(
            _generation_config(),
            {"default_sampling": {"min_new_tokens": 10}},
            {"prompt": "hi", "max_new_tokens": 1},
        )


def test_generation_explicit_min_tokens_above_max_fails() -> None:
    config = _generation_config()
    del config.profiles["default"].adapter_options.runtime["default_sampling"]

    with pytest.raises(
        ValueError,
        match=r"min_tokens \(10\) must not exceed max_new_tokens \(1\)",
    ):
        apply_generation_runtime_options(
            config,
            None,
            {"prompt": "hi", "max_new_tokens": 1, "min_tokens": 10},
        )


def test_generation_non_default_profile_requires_model_variant_identity() -> None:
    with pytest.raises(ValueError, match="model:profile"):
        apply_generation_runtime_options(
            _generation_config(),
            {"profile": "fast"},
            {"prompt": "hi"},
        )


def test_generation_unknown_option_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported generation option"):
        apply_generation_runtime_options(
            _generation_config(),
            {"not_executable": True},
            {"prompt": "hi"},
        )


@pytest.mark.parametrize(
    "sampling",
    [
        {"temperature": "0.7"},
        {"top_p": None},
        {"presence_penalty": float("inf")},
        {"temperature": 1 << 1024},
        {"top_k": 1 << 1024},
        {"min_new_tokens": 1 << 1024},
        {"frequency_penalty": -2.1},
        {"top_k": True},
        {"min_new_tokens": -1},
        {"seed": True},
        {"seed": 1 << 63},
        {"seed": 1 << 1024},
    ],
)
def test_generation_invalid_sampling_option_fails_closed(sampling: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="invalid value"):
        apply_generation_runtime_options(
            _generation_config(),
            {"default_sampling": sampling},
            {"prompt": "hi"},
        )


@pytest.mark.parametrize("value", [float("inf"), 1 << 1024])
def test_generation_non_finite_timeout_fails_closed(value: float) -> None:
    with pytest.raises(ValueError, match="positive number"):
        apply_generation_runtime_options(_generation_config(), {"overall_timeout_s": value}, {"prompt": "hi"})


def test_generation_timeouts_resolve_from_profile_and_request() -> None:
    config = _generation_config()

    assert resolve_generation_timeouts(config, None) == GenerationTimeouts(first_chunk_s=None, overall_s=60.0)
    assert resolve_generation_timeouts(
        config,
        {"first_chunk_timeout_s": 5, "overall_timeout_s": 12.5},
    ) == GenerationTimeouts(first_chunk_s=5.0, overall_s=12.5)

    undeclared = _generation_config()
    del undeclared.profiles["default"].adapter_options.runtime["overall_timeout_s"]
    assert resolve_generation_timeouts(undeclared, None) == GenerationTimeouts()


class _Engine:
    def __init__(self, delays: list[float]) -> None:
        self.delays = delays
        self.closed = False

    async def chunks(self) -> AsyncIterator[int]:
        try:
            for index, delay in enumerate(self.delays):
                await asyncio.sleep(delay)
                yield index
        finally:
            self.closed = True


async def _drain(chunks: AsyncIterator[int]) -> list[int]:
    return [chunk async for chunk in chunks]


async def test_bound_generation_passes_chunks_through_within_timeouts() -> None:
    engine = _Engine([0.0, 0.0, 0.0])

    chunks = await _drain(bound_generation(engine.chunks(), GenerationTimeouts(first_chunk_s=1.0, overall_s=1.0)))

    assert chunks == [0, 1, 2]
    assert engine.closed


async def test_bound_generation_without_timeouts_is_unbounded() -> None:
    engine = _Engine([0.05, 0.05])

    assert await _drain(bound_generation(engine.chunks(), GenerationTimeouts())) == [0, 1]


async def test_bound_generation_first_chunk_timeout_aborts_the_engine() -> None:
    engine = _Engine([5.0])

    with pytest.raises(GenerationTimeoutError) as raised:
        await _drain(bound_generation(engine.chunks(), GenerationTimeouts(first_chunk_s=0.05, overall_s=5.0)))

    assert raised.value.code == "first_chunk_timeout"
    assert engine.closed


async def test_bound_generation_overall_timeout_applies_after_the_first_chunk() -> None:
    engine = _Engine([0.0, 0.0, 5.0])

    with pytest.raises(GenerationTimeoutError) as raised:
        await _drain(bound_generation(engine.chunks(), GenerationTimeouts(first_chunk_s=0.05, overall_s=0.2)))

    assert raised.value.code == "overall_timeout"
    assert engine.closed


class _EngineWithHangingAbort:
    def __init__(self) -> None:
        self.close_started = False

    def __aiter__(self) -> _EngineWithHangingAbort:
        return self

    async def __anext__(self) -> int:
        await asyncio.sleep(10)
        return 0

    async def aclose(self) -> None:
        self.close_started = True
        await asyncio.sleep(10)


async def test_bound_generation_does_not_wait_on_a_hung_engine_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_options, "_GENERATION_CLOSE_TIMEOUT_S", 0.05)
    engine = _EngineWithHangingAbort()
    loop = asyncio.get_running_loop()
    started = loop.time()

    with pytest.raises(GenerationTimeoutError) as raised:
        await _drain(bound_generation(engine, GenerationTimeouts(first_chunk_s=0.05)))

    assert raised.value.code == "first_chunk_timeout"
    assert engine.close_started
    assert loop.time() - started < 2.0


class _AdapterWithHangingAbort(FakeAdapter):
    """A generation engine whose abort on cancellation takes a while."""

    def __init__(self, abort_s: float) -> None:
        super().__init__()
        self.abort_s = abort_s
        self.abort_started = False
        self.abort_finished = False

    async def generate(self, prompt: str, **kwargs: Any) -> AsyncIterator[GenerationChunk]:
        _ = (prompt, kwargs)
        try:
            await asyncio.sleep(30)
            yield GenerationChunk(text_delta="late")
        except asyncio.CancelledError:
            self.abort_started = True
            await asyncio.sleep(self.abort_s)
            self.abort_finished = True
            raise


async def test_bound_generation_answers_before_a_slow_engine_abort_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_options, "_GENERATION_CLOSE_TIMEOUT_S", 0.1)
    adapter = _AdapterWithHangingAbort(abort_s=0.5)
    adapter.load("cpu")
    chunks = adapter.generate_with_preflight({"prompt": "hi", "max_new_tokens": 4}, None)
    loop = asyncio.get_running_loop()
    started = loop.time()

    with pytest.raises(GenerationTimeoutError) as raised:
        await _drain(bound_generation(chunks, GenerationTimeouts(first_chunk_s=0.05)))

    elapsed = loop.time() - started
    assert raised.value.code == "first_chunk_timeout"
    assert elapsed < 0.4, elapsed
    assert adapter.abort_started
    assert not adapter.abort_finished

    await asyncio.sleep(0.6)
    assert adapter.abort_finished, "the engine abort must be allowed to finish in the background"


async def test_bound_generation_keeps_engine_timeout_errors() -> None:
    async def failing() -> AsyncIterator[int]:
        raise TimeoutError("engine read timed out")
        yield 0

    with pytest.raises(TimeoutError) as raised:
        await _drain(bound_generation(failing(), GenerationTimeouts(overall_s=5.0)))

    assert not isinstance(raised.value, GenerationTimeoutError)
