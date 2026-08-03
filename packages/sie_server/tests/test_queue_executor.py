from __future__ import annotations

import asyncio
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import numpy as np
import pytest
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.inference_output import ExtractOutput, ScoreOutput
from sie_server.core.loader import expand_profile_variants, load_model_config
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker.types import WorkerResult
from sie_server.ipc_types import (
    EncodeBatchItem,
    ExtractBatchItem,
    ProcessEncodeBatchRequest,
    ProcessExtractBatchRequest,
    ProcessScoreBatchRequest,
    ScoreBatchItem,
)
from sie_server.queue_executor import QueueExecutor, _validate_prepared_audio

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _load_profile_variant(model_file: str, route: str) -> ModelConfig:
    return expand_profile_variants([load_model_config(_MODELS_DIR / model_file)])[route]


def test_prepared_audio_rejects_zero_source_sample_rate() -> None:
    audio = MagicMock(
        sample_rate=16_000,
        source_sample_rate=0,
        sample_count=16_000,
        source_sample_count=48_000,
        duration_ms=1_000,
        pcm_s16le=b"\x00\x00" * 16_000,
    )

    with pytest.raises(ValueError, match="source_sample_rate must be between 8000 and 48000"):
        _validate_prepared_audio(audio)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_channels", 3, "source_channels must be 1 or 2"),
        ("container", "aac", "container is unsupported"),
        ("duration_ms", 720_001, "duration_ms exceeds"),
        ("sample_count", 11_522_049, "canonical sample_count exceeds"),
    ],
)
def test_prepared_audio_rejects_untrusted_provenance(field: str, value: object, message: str) -> None:
    audio = MagicMock(
        sample_rate=16_000,
        source_sample_rate=16_000,
        source_channels=1,
        sample_count=16_000,
        source_sample_count=16_000,
        duration_ms=1_000,
        pcm_s16le=b"\x00\x00" * 16_000,
        container="wav",
    )
    setattr(audio, field, value)
    if field == "duration_ms":
        audio.source_sample_count = 11_520_016
    elif field == "sample_count":
        audio.pcm_s16le = b"\x00\x00" * value

    with pytest.raises(ValueError, match=message):
        _validate_prepared_audio(audio)


@pytest.mark.parametrize(("source_sample_count", "duration_ms"), [(1, 1), (1_601, 101)])
def test_prepared_audio_accepts_ceil_duration(source_sample_count: int, duration_ms: int) -> None:
    audio = MagicMock(
        sample_rate=16_000,
        source_sample_rate=16_000,
        source_channels=1,
        sample_count=source_sample_count,
        source_sample_count=source_sample_count,
        duration_ms=duration_ms,
        pcm_s16le=b"\x00\x00" * source_sample_count,
        container="wav",
    )

    _validate_prepared_audio(audio)


@pytest.mark.parametrize(
    ("source_sample_count", "duration_ms", "message"),
    [(1, 0, "duration_ms must be positive"), (1_601, 100, "does not match")],
)
def test_prepared_audio_rejects_nonpositive_or_floor_duration(
    source_sample_count: int, duration_ms: int, message: str
) -> None:
    audio = MagicMock(
        sample_rate=16_000,
        source_sample_rate=16_000,
        source_channels=1,
        sample_count=source_sample_count,
        source_sample_count=source_sample_count,
        duration_ms=duration_ms,
        pcm_s16le=b"\x00\x00" * source_sample_count,
        container="wav",
    )

    with pytest.raises(ValueError, match=message):
        _validate_prepared_audio(audio)


def _make_registry(*, loaded: bool = True, loading: bool = False) -> MagicMock:
    reg = MagicMock()
    reg.model_names = ["test/model"]
    reg.device = "cpu"
    reg.has_model.return_value = True
    reg.is_loaded.return_value = loaded
    reg.is_loading.return_value = loading
    reg.get_config.return_value = ModelConfig(
        sie_id="test/model",
        package_backed=True,
        tasks=Tasks(
            encode=EncodeTask(
                dense=EmbeddingDim(dim=768),
                sparse=EmbeddingDim(dim=768),
                multivector=EmbeddingDim(dim=128),
            )
        ),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                max_batch_tokens=8192,
            )
        },
    )
    # No recorded load failure by default — a bare MagicMock would otherwise
    # return a truthy stub for ``get_failure(...)`` and its ``.is_permanent``,
    # which ``ensure_model_ready`` now reads to gate the terminal ``failed``
    # state (#1786 fast-path). Tests that want the terminal path set this.
    reg.get_failure.return_value = None
    return reg


def _make_config(name: str) -> ModelConfig:
    return ModelConfig(
        sie_id=name,
        package_backed=True,
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                max_batch_tokens=8192,
            )
        },
    )


async def _wait_until_loaded(registry: ModelRegistry, *model_ids: str, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(registry.is_loaded(model_id) for model_id in model_ids):
            return
        await asyncio.sleep(0.01)
    if all(registry.is_loaded(model_id) for model_id in model_ids):
        return
    missing = [model_id for model_id in model_ids if not registry.is_loaded(model_id)]
    raise AssertionError(f"timed out waiting for models to load: {missing}")


def _encode_item(
    *,
    wiid: str = "req-1.0",
    item: dict | None = None,
    output_types: list[str] | None = None,
    instruction: str | None = None,
    is_query: bool = False,
    options: dict | None = None,
    item_index: int = 0,
) -> EncodeBatchItem:
    return EncodeBatchItem(
        work_item_id=wiid,
        request_id=wiid.split(".", maxsplit=1)[0],
        item_index=item_index,
        total_items=1,
        timestamp=time.time(),
        item=item or {"text": "hello"},
        output_types=output_types,
        instruction=instruction,
        is_query=is_query,
        options=options,
    )


def _score_item(
    *,
    wiid: str = "req-1.0",
    options: dict | None = None,
    item_index: int = 0,
) -> ScoreBatchItem:
    return ScoreBatchItem(
        work_item_id=wiid,
        request_id=wiid.split(".", maxsplit=1)[0],
        item_index=item_index,
        total_items=1,
        timestamp=time.time(),
        query_item={"text": "q"},
        score_items=[{"text": "a", "id": "doc-a"}],
        options=options,
    )


def _extract_item(
    *,
    wiid: str = "req-1.0",
    options: dict | None = None,
    item_index: int = 0,
) -> ExtractBatchItem:
    return ExtractBatchItem(
        work_item_id=wiid,
        request_id=wiid.split(".", maxsplit=1)[0],
        item_index=item_index,
        total_items=1,
        timestamp=time.time(),
        item={"text": "Alice works at Acme."},
        labels=["person"],
        options=options,
    )


def _runtime_profile_config() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "sie_id": "test/runtime-profiles",
            "hf_id": "test/runtime-profiles",
            "inputs": {"text": True},
            "tasks": {"encode": {"dense": {"dim": 8}}},
            "max_sequence_length": 512,
            "profiles": {
                "default": {
                    "max_batch_tokens": 8192,
                    "adapter_path": "sie_server.adapters.fake.adapter:FakeAdapter",
                    "adapter_options": {"runtime": {"threshold": 0.25}},
                },
                "tuned": {
                    "max_batch_tokens": 8192,
                    "adapter_path": "sie_server.adapters.fake.adapter:FakeAdapter",
                    "adapter_options": {
                        "runtime": {
                            "threshold": 0.5,
                            "lora": "profile-lora",
                        }
                    },
                },
            },
        }
    )


# -----------------------------------------------------------------------------
# ensure_model_ready
# -----------------------------------------------------------------------------


class TestEnsureModelReady:
    @pytest.mark.asyncio
    async def test_loaded_returns_ready(self) -> None:
        reg = _make_registry(loaded=True)
        ex = QueueExecutor(reg)
        assert await ex.ensure_model_ready("test/model") == "ready"
        reg.start_load_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_loaded_but_no_config_returns_retry_later(self) -> None:
        reg = _make_registry(loaded=True)
        reg.has_model.return_value = False
        ex = QueueExecutor(reg)
        assert await ex.ensure_model_ready("test/model") == "retry_later"
        reg.start_load_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_already_loading_returns_in_progress(self) -> None:
        reg = _make_registry(loaded=False, loading=True)
        ex = QueueExecutor(reg)
        assert await ex.ensure_model_ready("test/model") == "loading_in_progress"
        reg.start_load_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_triggers_new_load(self) -> None:
        reg = _make_registry(loaded=False, loading=False)
        reg.start_load_async = AsyncMock(return_value=True)
        ex = QueueExecutor(reg)
        assert await ex.ensure_model_ready("test/model") == "loading_started"
        reg.start_load_async.assert_awaited_once_with("test/model", "cpu")

    @pytest.mark.asyncio
    async def test_start_load_already_running_returns_in_progress(self) -> None:
        reg = _make_registry(loaded=False, loading=False)
        reg.start_load_async = AsyncMock(return_value=False)  # already loading/loaded in another task
        ex = QueueExecutor(reg)
        assert await ex.ensure_model_ready("test/model") == "loading_in_progress"

    @pytest.mark.asyncio
    async def test_unknown_model_returns_retry_later(self) -> None:
        reg = _make_registry(loaded=False, loading=False)
        reg.start_load_async = AsyncMock(side_effect=KeyError("test/model"))
        ex = QueueExecutor(reg)
        assert await ex.ensure_model_ready("test/model") == "retry_later"

    @pytest.mark.asyncio
    async def test_permanent_failure_returns_terminal_failed(self) -> None:
        """#1786 fast-path: a PERMANENT load failure surfaces as the terminal
        ``failed`` state (not ``loading_in_progress``) so the sidecar
        dead-letters instead of re-driving forever.
        """
        from sie_server.core.load_errors import LoadErrorClass, LoadFailure

        reg = _make_registry(loaded=False, loading=False)
        reg.get_failure.return_value = LoadFailure(
            error_class=LoadErrorClass.GATED,
            message="repository is gated",
            attempts=1,
            last_attempt_ts=time.monotonic(),
            cooldown_s=None,  # permanent
        )
        ex = QueueExecutor(reg)
        assert await ex.ensure_model_ready("test/model") == "failed"
        # Terminal — must NOT try to (re)start a doomed load.
        reg.start_load_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_transient_failure_is_not_terminal(self) -> None:
        """A TRANSIENT in-cooldown failure (OOM/NETWORK/TIMEOUT) stays
        retryable — it must not be reported as terminal ``failed``.
        """
        from sie_server.core.load_errors import LoadErrorClass, LoadFailure

        reg = _make_registry(loaded=False, loading=False)
        reg.get_failure.return_value = LoadFailure(
            error_class=LoadErrorClass.OOM,
            message="cuda oom",
            attempts=1,
            last_attempt_ts=time.monotonic(),
            cooldown_s=30.0,  # transient
        )
        reg.start_load_async = AsyncMock(return_value=False)
        ex = QueueExecutor(reg)
        assert await ex.ensure_model_ready("test/model") == "loading_in_progress"

    @pytest.mark.asyncio
    async def test_permanent_failure_recorded_mid_start_returns_failed(self) -> None:
        """A load that flips to a PERMANENT failure while ``start_load_async``
        runs (so it returns ``False``) is dead-lettered, not re-driven.
        """
        from sie_server.core.load_errors import LoadErrorClass, LoadFailure

        reg = _make_registry(loaded=False, loading=False)
        # Clean at the up-front check, permanent by the time start returns False.
        reg.get_failure.side_effect = [
            None,
            LoadFailure(
                error_class=LoadErrorClass.DEPENDENCY,
                message="missing dep",
                attempts=1,
                last_attempt_ts=time.monotonic(),
                cooldown_s=None,
            ),
        ]
        reg.start_load_async = AsyncMock(return_value=False)
        ex = QueueExecutor(reg)
        assert await ex.ensure_model_ready("test/model") == "failed"

    @pytest.mark.asyncio
    @patch("sie_server.core.model_loader.load_adapter")
    async def test_concurrent_sidecar_ready_loads_use_distinct_configured_devices(
        self,
        mock_load_adapter: MagicMock,
    ) -> None:
        """Concurrent sidecar-triggered loads serialize placement through the registry lock."""
        registry = ModelRegistry(device="cuda", devices=["cuda:0", "cuda:1"])
        registry.add_config(_make_config("model-a"))
        registry.add_config(_make_config("model-b"))

        adapters = [MagicMock(), MagicMock()]
        for adapter in adapters:
            adapter.aclose_client = None
            adapter.capabilities.outputs = ["dense"]
            adapter.memory_footprint.return_value = 1000
            adapter.requires_main_thread = False
        mock_load_adapter.side_effect = adapters

        with ExitStack() as stack:
            for manager in registry.memory_managers.values():
                stack.enter_context(patch.object(manager, "check_pressure", return_value=False))

            executor = QueueExecutor(registry)
            states = await asyncio.gather(
                executor.ensure_model_ready("model-a"),
                executor.ensure_model_ready("model-b"),
            )
            await _wait_until_loaded(registry, "model-a", "model-b")

        assert states == ["loading_started", "loading_started"]
        assert registry._loaded["model-a"].device == "cuda:0"
        assert registry._loaded["model-b"].device == "cuda:1"
        adapters[0].load.assert_called_once_with("cuda:0")
        adapters[1].load.assert_called_once_with("cuda:1")


class TestGetBatchBudget:
    """Per-model batch budget advertised to the Rust queue consumer via
    EnsureModelReadyResponse.
    """

    def test_returns_max_batch_requests_when_worker_has_batch_config(self) -> None:
        reg = _make_registry()
        worker = MagicMock()
        worker._batch_config = MagicMock(max_batch_requests=37)
        reg.get_worker.return_value = worker
        ex = QueueExecutor(reg)
        assert ex.get_batch_budget("test/model") == 37

    def test_returns_none_when_worker_missing(self) -> None:
        reg = _make_registry()
        reg.get_worker.return_value = None
        ex = QueueExecutor(reg)
        assert ex.get_batch_budget("test/model") is None

    def test_returns_none_when_worker_has_no_batch_config(self) -> None:
        reg = _make_registry()
        worker = object()  # no `_batch_config` attribute
        reg.get_worker.return_value = worker
        ex = QueueExecutor(reg)
        assert ex.get_batch_budget("test/model") is None

    def test_returns_none_when_registry_raises(self) -> None:
        reg = _make_registry()
        reg.get_worker.side_effect = KeyError("test/model")
        ex = QueueExecutor(reg)
        assert ex.get_batch_budget("test/model") is None

    def test_returns_none_when_budget_is_non_int(self) -> None:
        # Protect the wire: a misconfigured BatchConfig with a non-int
        # max_batch_requests should not leak a MagicMock / string / None
        # into the IPC response where msgspec would choke on it.
        reg = _make_registry()
        worker = MagicMock()
        worker._batch_config = MagicMock(max_batch_requests="64")  # str, not int
        reg.get_worker.return_value = worker
        ex = QueueExecutor(reg)
        assert ex.get_batch_budget("test/model") is None

    def test_returns_none_when_budget_is_non_positive(self) -> None:
        reg = _make_registry()
        worker = MagicMock()
        worker._batch_config = MagicMock(max_batch_requests=0)
        reg.get_worker.return_value = worker
        ex = QueueExecutor(reg)
        assert ex.get_batch_budget("test/model") is None


# -----------------------------------------------------------------------------
# process_encode_batch
# -----------------------------------------------------------------------------


class TestProcessEncodeBatch:
    @pytest.mark.asyncio
    async def test_single_item_publish_and_ack(self) -> None:
        reg = _make_registry()
        ex = QueueExecutor(reg)

        fake_outputs = [{"dense": [0.1, 0.2]}]
        fake_timing = RequestTiming()

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=(fake_outputs, fake_timing),
        ) as mock_encode:
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(model_id="test/model", items=[_encode_item()])
            )

        assert len(outcome.outcomes) == 1
        item = outcome.outcomes[0]
        assert item.disposition == "publish_and_ack"
        assert item.result_msgpack is not None
        inner = msgpack.unpackb(item.result_msgpack, raw=False)
        assert inner == {"dense": [0.1, 0.2]}
        mock_encode.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_malformed_item_isolated_as_invalid_input(self) -> None:
        """A typed-decode failure on one item in a sub-group is isolated as an
        INVALID_INPUT outcome; the valid item in the same group still runs.
        Regression for #1537 (queue path bypassed Item validation).
        """
        reg = _make_registry()
        ex = QueueExecutor(reg)

        async def fake_run_encode(**kwargs):
            n = len(kwargs["items"])
            return [{"dense": [0.0]} for _ in range(n)], RequestTiming()

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new=AsyncMock(side_effect=fake_run_encode),
        ) as mock_encode:
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id="test/model",
                    items=[
                        _encode_item(wiid="req-1.0", item={"text": "ok"}, item_index=0),
                        # text must be str | None — an int is rejected at the seam.
                        _encode_item(wiid="req-1.1", item={"text": 123}, item_index=1),
                    ],
                )
            )

        by_id = {o.work_item_id: o for o in outcome.outcomes}
        assert by_id["req-1.0"].disposition == "publish_and_ack"
        assert by_id["req-1.1"].disposition == "publish_error_and_ack"
        assert by_id["req-1.1"].error_code == "INVALID_INPUT"
        # The malformed item never reached inference; only the valid one did.
        mock_encode.assert_awaited_once()
        assert len(mock_encode.await_args.kwargs["items"]) == 1

    @pytest.mark.asyncio
    async def test_merges_profile_runtime_options_into_adapter_call(self) -> None:
        """Regression for #1489: the cluster worker path must merge the model
        profile's ``adapter_options.runtime`` defaults (query_template,
        default_instruction, …) into the options the adapter sees — the Rust
        gateway publishes only raw SDK options, so the worker has to do the
        merge that ``api.encode`` does for the single-server path.
        """
        config = ModelConfig.model_validate(
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
                                "query_template": "Instruct: {instruction}\nQuery: {text}",
                                "default_instruction": "Given a query, retrieve relevant passages",
                                "normalize": True,
                            },
                        },
                    },
                },
            }
        )
        reg = _make_registry()
        reg.get_config.return_value = config
        ex = QueueExecutor(reg)

        captured: dict = {}

        async def fake_run_encode(**kwargs):
            captured.update(kwargs["options"])
            n = len(kwargs["items"])
            return [{"dense": [0.0]} for _ in range(n)], RequestTiming()

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new=AsyncMock(side_effect=fake_run_encode),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id="test/model",
                    items=[_encode_item(options={"is_query": True})],
                )
            )

        assert outcome.outcomes[0].disposition == "publish_and_ack"
        # The adapter must receive the profile's runtime template + instruction,
        # not just the raw {"is_query": True} the SDK sent.
        assert captured["query_template"] == "Instruct: {instruction}\nQuery: {text}"
        assert captured["default_instruction"] == "Given a query, retrieve relevant passages"
        assert captured["normalize"] is True
        assert captured["is_query"] is True

    @pytest.mark.asyncio
    async def test_merges_non_default_profile_runtime_from_options(self) -> None:
        """A queued request selects its profile via options["profile"] (the gateway
        forwards raw SDK options; the wire profile_id is a hardcoded routing
        placeholder), so the worker must resolve that profile's runtime defaults.
        """
        config = ModelConfig.model_validate(
            {
                "sie_id": "test/multi-profile-embedder",
                "hf_id": "test/multi-profile-embedder",
                "inputs": {"text": True},
                "tasks": {"encode": {"dense": {"dim": 8}}},
                "max_sequence_length": 512,
                "profiles": {
                    "default": {
                        "max_batch_tokens": 8192,
                        "adapter_path": "sie_server.adapters.sglang.embedding:SGLangEmbeddingAdapter",
                        "adapter_options": {"runtime": {"query_template": "default: {text}"}},
                    },
                    "fast": {
                        "max_batch_tokens": 8192,
                        "adapter_path": "sie_server.adapters.sglang.embedding:SGLangEmbeddingAdapter",
                        "adapter_options": {"runtime": {"query_template": "fast: {text}"}},
                    },
                },
            }
        )
        reg = _make_registry()
        reg.get_config.return_value = config
        ex = QueueExecutor(reg)

        captured: dict = {}

        async def fake_run_encode(**kwargs):
            captured.update(kwargs["options"])
            return [{"dense": [0.0]} for _ in kwargs["items"]], RequestTiming()

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new=AsyncMock(side_effect=fake_run_encode),
        ):
            await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id="test/model",
                    items=[_encode_item(options={"is_query": True, "profile": "fast"})],
                )
            )

        # The non-default profile's runtime wins, and "profile" is consumed.
        assert captured["query_template"] == "fast: {text}"
        assert "profile" not in captured

    @pytest.mark.asyncio
    async def test_profile_default_output_types_applied(self) -> None:
        """Regression for dade77d64: a profile whose runtime declares an
        ``output_types`` default must resolve on the managed path exactly as
        OSS ``api.encode`` does (profile > request > default).

        Mirrors the ``bge-m3:sparse`` synthetic variant — its promoted
        "default" profile carries ``output_types: [sparse]``. A queued request
        with NO request-level output_types must reach the engine as
        ``["sparse"]`` and be served as sparse, not silently fall back to
        dense-only (the managed path dropped the profile default).
        """
        config = ModelConfig.model_validate(
            {
                "sie_id": "test/bge-m3:sparse",
                "hf_id": "test/bge-m3",
                "inputs": {"text": True},
                "tasks": {"encode": {"dense": {"dim": 8}, "sparse": {"dim": 250002}}},
                "max_sequence_length": 512,
                "profiles": {
                    "default": {
                        "max_batch_tokens": 8192,
                        "adapter_path": "sie_server.adapters.bge_m3_flash:BGEM3FlashAdapter",
                        "adapter_options": {
                            "runtime": {"pooling": "cls", "normalize": True, "output_types": ["sparse"]},
                        },
                    },
                },
            }
        )
        reg = _make_registry()
        reg.get_config.return_value = config
        ex = QueueExecutor(reg)

        observed: list[list[str]] = []

        async def fake_run_encode(**kwargs):
            observed.append(kwargs["output_types"])
            sparse = {
                "sparse": {
                    "indices": np.array([1, 2], dtype=np.int32),
                    "values": np.array([0.5, 0.25], dtype=np.float32),
                }
            }
            return [sparse for _ in kwargs["items"]], RequestTiming()

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new=AsyncMock(side_effect=fake_run_encode),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id="test/bge-m3:sparse",
                    items=[_encode_item(options=None)],  # no request-level output_types
                )
            )

        # Engine received the profile's head, not the ["dense"] fallback...
        assert observed == [["sparse"]]
        # ...and the item is served as sparse.
        assert outcome.outcomes[0].disposition == "publish_and_ack"
        assert outcome.outcomes[0].raw_output is not None
        assert outcome.outcomes[0].raw_output.sparse is not None

    @pytest.mark.parametrize(
        ("model_file", "route"),
        [
            (
                "nvidia__llama-nemoretriever-colembed-3b-v1.yaml",
                "nvidia/llama-nemoretriever-colembed-3b-v1:muvera",
            ),
            ("nvidia__nemotron-colembed-vl-4b-v2.yaml", "nvidia/nemotron-colembed-vl-4b-v2:muvera"),
            ("vidore__colpali-v1.3-hf.yaml", "vidore/colpali-v1.3-hf:muvera"),
        ],
    )
    @pytest.mark.asyncio
    async def test_multimodal_muvera_route_translates_managed_adapter_output(
        self,
        model_file: str,
        route: str,
    ) -> None:
        """Canonical multimodal routes authorize dense while adapters emit multivectors."""
        config = _load_profile_variant(model_file, route)
        assert config.inputs.image is True
        assert config.synthetic_profile_variant_source is not None
        reg = _make_registry()
        reg.get_config.return_value = config
        ex = QueueExecutor(reg)

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=([{"dense": [0.1, 0.2]}], RequestTiming()),
        ) as mock_encode:
            outcome = await ex.process_encode_batch(ProcessEncodeBatchRequest(model_id=route, items=[_encode_item()]))

        call = mock_encode.await_args.kwargs
        assert call["model"] == route
        assert call["output_types"] == ["multivector"]
        assert call["response_output_types"] == ["dense"]
        assert outcome.outcomes[0].disposition == "publish_and_ack"
        assert outcome.outcomes[0].result_msgpack is not None
        assert msgpack.unpackb(outcome.outcomes[0].result_msgpack, raw=False) == {"dense": [0.1, 0.2]}

    @pytest.mark.asyncio
    async def test_request_options_cannot_self_authorize_managed_output(self) -> None:
        """Caller options cannot expand the managed route's output allowlist."""
        config = ModelConfig(
            sie_id="test/colbert",
            hf_id="test/colbert",
            tasks=Tasks(encode=EncodeTask(multivector=EmbeddingDim(dim=8))),
            profiles={
                "default": ProfileConfig(
                    adapter_path="sie_server.adapters.colbert:ColBERTAdapter",
                    max_batch_tokens=8192,
                )
            },
        )
        reg = _make_registry()
        reg.get_config.return_value = config
        ex = QueueExecutor(reg)

        outcome = await ex.process_encode_batch(
            ProcessEncodeBatchRequest(
                model_id="test/colbert",
                items=[_encode_item(options={"muvera": {}, "output_types": ["dense"]})],
            )
        )

        assert outcome.outcomes[0].disposition == "publish_error_and_ack"
        assert outcome.outcomes[0].error_code == "INVALID_INPUT"
        assert "dense" in (outcome.outcomes[0].error or "")

    @pytest.mark.parametrize("output_types", ["dense", [], [{}], ["score"], [1]])
    @pytest.mark.asyncio
    async def test_managed_output_types_reject_malformed_values(self, output_types: object) -> None:
        outcome = await QueueExecutor(_make_registry()).process_encode_batch(
            ProcessEncodeBatchRequest(
                model_id="test/model",
                items=[_encode_item(options={"output_types": output_types})],
            )
        )

        assert outcome.outcomes[0].disposition == "publish_error_and_ack"
        assert outcome.outcomes[0].error_code == "INVALID_INPUT"

    @pytest.mark.parametrize("output_types", [[], ["score"]])
    @pytest.mark.asyncio
    async def test_managed_top_level_output_types_fail_closed(self, output_types: list[str]) -> None:
        outcome = await QueueExecutor(_make_registry()).process_encode_batch(
            ProcessEncodeBatchRequest(
                model_id="test/model",
                items=[_encode_item(output_types=output_types)],
            )
        )

        assert outcome.outcomes[0].disposition == "publish_error_and_ack"
        assert outcome.outcomes[0].error_code == "INVALID_INPUT"

    @pytest.mark.parametrize("profile", ["", " ", [], {}, 0, False, ["default"], {"name": "default"}])
    @pytest.mark.asyncio
    async def test_managed_profile_selector_rejects_malformed_values(self, profile: object) -> None:
        outcome = await QueueExecutor(_make_registry()).process_encode_batch(
            ProcessEncodeBatchRequest(
                model_id="test/model",
                items=[_encode_item(options={"profile": profile})],
            )
        )

        assert outcome.outcomes[0].disposition == "publish_error_and_ack"
        assert outcome.outcomes[0].error_code == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_managed_unknown_profile_is_invalid_input(self) -> None:
        outcome = await QueueExecutor(_make_registry()).process_encode_batch(
            ProcessEncodeBatchRequest(
                model_id="test/model",
                items=[_encode_item(options={"profile": "missing"})],
            )
        )

        assert outcome.outcomes[0].disposition == "publish_error_and_ack"
        assert outcome.outcomes[0].error_code == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_muvera_profile_translates_managed_adapter_output(self) -> None:
        """Managed queue requests must use the same MuVERA contract as HTTP."""
        config = ModelConfig.model_validate(
            {
                "sie_id": "test/colbert:muvera",
                "hf_id": "test/colbert",
                "inputs": {"text": True},
                "tasks": {"encode": {"multivector": {"dim": 8}}},
                "profiles": {
                    "default": {
                        "max_batch_tokens": 8192,
                        "adapter_path": "sie_server.adapters.colbert:ColBERTAdapter",
                        "adapter_options": {
                            "runtime": {"muvera": {}, "output_types": ["dense"]},
                        },
                    },
                },
            }
        )
        reg = _make_registry()
        reg.get_config.return_value = config
        ex = QueueExecutor(reg)

        fake_timing = RequestTiming()
        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=([{"dense": [0.1, 0.2]}], fake_timing),
        ) as mock_encode:
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(model_id="test/colbert:muvera", items=[_encode_item()])
            )

        call = mock_encode.await_args.kwargs
        assert call["output_types"] == ["multivector"]
        assert call["response_output_types"] == ["dense"]
        assert outcome.outcomes[0].disposition == "publish_and_ack"
        assert outcome.outcomes[0].result_msgpack is not None
        assert msgpack.unpackb(outcome.outcomes[0].result_msgpack, raw=False) == {"dense": [0.1, 0.2]}

    @pytest.mark.asyncio
    async def test_request_output_types_and_base_default_unchanged(self) -> None:
        """The profile-default fix must not over-reach: with NO profile-level
        ``output_types`` default, the managed path keeps OSS ``request >
        default`` — an explicit request output_types is honoured, an absent one
        falls back to ``["dense"]``.
        """
        config = ModelConfig.model_validate(
            {
                "sie_id": "test/plain-embedder",
                "hf_id": "test/plain-embedder",
                "inputs": {"text": True},
                "tasks": {"encode": {"dense": {"dim": 8}, "multivector": {"dim": 8}}},
                "max_sequence_length": 512,
                "profiles": {
                    "default": {
                        "max_batch_tokens": 8192,
                        "adapter_path": "sie_server.adapters.bge_m3_flash:BGEM3FlashAdapter",
                        "adapter_options": {"runtime": {"pooling": "cls", "normalize": True}},
                    },
                },
            }
        )
        reg = _make_registry()
        reg.get_config.return_value = config
        ex = QueueExecutor(reg)

        observed: list[tuple[str, ...]] = []

        async def fake_run_encode(**kwargs):
            observed.append(tuple(kwargs["output_types"]))
            return [{"dense": [0.0]} for _ in kwargs["items"]], RequestTiming()

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new=AsyncMock(side_effect=fake_run_encode),
        ):
            await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id="test/plain-embedder",
                    items=[
                        # Explicit request output_types -> honoured (not clobbered).
                        _encode_item(wiid="a.0", output_types=["multivector"], item_index=0),
                        # No request output_types -> base ["dense"] default.
                        _encode_item(wiid="b.0", output_types=None, item_index=1),
                    ],
                )
            )

        assert sorted(observed) == [("dense",), ("multivector",)]

    @pytest.mark.asyncio
    async def test_model_evicted_naks_all(self) -> None:
        reg = _make_registry()
        reg.get_config.side_effect = KeyError("test/model")
        ex = QueueExecutor(reg)

        outcome = await ex.process_encode_batch(
            ProcessEncodeBatchRequest(
                model_id="test/model",
                items=[_encode_item(wiid="a.0"), _encode_item(wiid="b.0", item_index=1)],
            )
        )
        assert len(outcome.outcomes) == 2
        assert all(o.disposition == "nak_retry" for o in outcome.outcomes)
        assert all((o.nak_delay_ms or 0) > 0 for o in outcome.outcomes)

    @pytest.mark.asyncio
    async def test_pipeline_exception_error_outcome(self) -> None:
        reg = _make_registry()
        ex = QueueExecutor(reg)

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(model_id="test/model", items=[_encode_item()])
            )

        assert len(outcome.outcomes) == 1
        assert outcome.outcomes[0].disposition == "publish_error_and_ack"
        # Pin the literal lowercase wire value on purpose (not the
        # _INFERENCE_ERROR_CODE constant): this guards the sidecar↔gateway
        # contract so a change to the constant surfaces as a failure here.
        assert outcome.outcomes[0].error_code == "inference_error"
        assert "boom" in (outcome.outcomes[0].error or "")

    @pytest.mark.asyncio
    async def test_pipeline_oom_naks_without_publishing_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_OOM_NAK_DELAY_S", "11.0")
        reg = _make_registry()
        ex = QueueExecutor(reg)

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB"),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(model_id="test/model", items=[_encode_item()])
            )

        assert len(outcome.outcomes) == 1
        assert outcome.outcomes[0].disposition == "nak_retry"
        assert outcome.outcomes[0].nak_delay_ms == 11_000
        assert outcome.outcomes[0].error_code is None
        assert outcome.outcomes[0].error is None

    @pytest.mark.asyncio
    async def test_heterogeneous_batch_groups_by_params(self) -> None:
        """Items with different (output_types, instruction, is_query, options) run as separate sub-batches."""
        reg = _make_registry()
        ex = QueueExecutor(reg)

        call_count = 0
        observed_output_types: list[list[str]] = []

        async def fake_run_encode(**kwargs):
            nonlocal call_count
            call_count += 1
            observed_output_types.append(kwargs["output_types"])
            n = len(kwargs["items"])
            return [{"dense": [float(call_count)]} for _ in range(n)], RequestTiming()

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new=AsyncMock(side_effect=fake_run_encode),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id="test/model",
                    items=[
                        _encode_item(wiid="a.0", output_types=["dense"]),
                        _encode_item(wiid="b.0", output_types=["dense"], item_index=1),
                        _encode_item(wiid="c.0", output_types=["sparse"], item_index=2),
                    ],
                )
            )

        assert call_count == 2
        assert sorted([tuple(ot) for ot in observed_output_types]) == [("dense",), ("sparse",)]
        assert len(outcome.outcomes) == 3
        assert all(o.disposition == "publish_and_ack" for o in outcome.outcomes)
        # Outcomes preserve input order regardless of grouping.
        assert [o.work_item_id for o in outcome.outcomes] == ["a.0", "b.0", "c.0"]

    @pytest.mark.asyncio
    async def test_output_ordering_matches_input(self) -> None:
        reg = _make_registry()
        ex = QueueExecutor(reg)

        fake_outputs = [{"dense": [float(i)]} for i in range(3)]
        fake_timing = RequestTiming()

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=(fake_outputs, fake_timing),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id="test/model",
                    items=[
                        _encode_item(wiid="a.0", item_index=0),
                        _encode_item(wiid="b.0", item_index=1),
                        _encode_item(wiid="c.0", item_index=2),
                    ],
                )
            )

        assert [o.work_item_id for o in outcome.outcomes] == ["a.0", "b.0", "c.0"]


# -----------------------------------------------------------------------------
# process_score_batch
# -----------------------------------------------------------------------------


class TestProcessScoreBatch:
    @pytest.mark.asyncio
    async def test_single_item_publish_and_ack(self) -> None:
        reg = _make_registry()
        worker = AsyncMock()
        score_output = ScoreOutput(scores=np.array([0.9, 0.1], dtype=np.float32))
        wr = WorkerResult(output=score_output, timing=RequestTiming())
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(wr)
        worker.submit_score_preformed_batch = AsyncMock(return_value=[fut])
        reg.start_worker = AsyncMock(return_value=worker)

        ex = QueueExecutor(reg)
        outcome = await ex.process_score_batch(
            ProcessScoreBatchRequest(
                model_id="test/model",
                items=[
                    ScoreBatchItem(
                        work_item_id="req-1.0",
                        request_id="req-1",
                        item_index=0,
                        total_items=1,
                        timestamp=time.time(),
                        query_item={"text": "q"},
                        score_items=[{"text": "a", "id": "doc-a"}, {"text": "b", "id": "doc-b"}],
                    )
                ],
            )
        )

        assert len(outcome.outcomes) == 1
        o = outcome.outcomes[0]
        assert o.disposition == "publish_and_ack"
        # Score output is Rust-frameable. The legacy ``result_msgpack``
        # is None; the parallel arrays land on ``raw_output.score`` and
        # ``sie_server_sidecar::output::build_score_payload`` does
        # the descending stable sort + rank assignment byte-identically.
        assert o.result_msgpack is None
        assert o.raw_output is not None
        assert o.raw_output.score is not None
        assert o.raw_output.score.item_ids == ["doc-a", "doc-b"]
        assert o.raw_output.score.scores == pytest.approx([0.9, 0.1])
        assert o.units is not None
        assert o.units.pairs == 2

    @pytest.mark.asyncio
    async def test_profile_runtime_is_effective_and_invalid_profile_is_per_item(self) -> None:
        reg = _make_registry()
        reg.get_config.return_value = _runtime_profile_config()
        worker = AsyncMock()
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(
            WorkerResult(output=ScoreOutput(scores=np.array([0.9], dtype=np.float32)), timing=RequestTiming())
        )
        worker.submit_score_preformed_batch = AsyncMock(return_value=[fut])
        reg.start_worker = AsyncMock(return_value=worker)

        outcome = await QueueExecutor(reg).process_score_batch(
            ProcessScoreBatchRequest(
                model_id="test/model",
                items=[
                    _score_item(
                        wiid="good.0",
                        options={
                            "profile": "tuned",
                            "threshold": 0.75,
                            "request_only": True,
                        },
                    ),
                    _score_item(wiid="bad.0", options={"profile": "missing"}),
                ],
            )
        )

        by_id = {item.work_item_id: item for item in outcome.outcomes}
        assert by_id["good.0"].disposition == "publish_and_ack"
        assert by_id["bad.0"].disposition == "publish_error_and_ack"
        assert by_id["bad.0"].error_code == "INVALID_INPUT"
        worker.submit_score_preformed_batch.assert_awaited_once()
        requests = worker.submit_score_preformed_batch.await_args.args[0]
        assert len(requests) == 1
        assert requests[0].options == {
            "threshold": 0.75,
            "lora": "profile-lora",
            "request_only": True,
        }

    @pytest.mark.asyncio
    async def test_malformed_item_isolated_as_invalid_input(self) -> None:
        """A typed-decode failure on the query item is isolated as INVALID_INPUT
        while a sibling valid item still runs. Regression for #1537 (the score
        path now decodes through the shared decode_item seam).
        """
        reg = _make_registry()
        worker = AsyncMock()
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(
            WorkerResult(output=ScoreOutput(scores=np.array([0.9], dtype=np.float32)), timing=RequestTiming())
        )
        worker.submit_score_preformed_batch = AsyncMock(return_value=[fut])
        reg.start_worker = AsyncMock(return_value=worker)

        ex = QueueExecutor(reg)
        outcome = await ex.process_score_batch(
            ProcessScoreBatchRequest(
                model_id="test/model",
                items=[
                    ScoreBatchItem(
                        work_item_id="req-1.0",
                        request_id="req-1",
                        item_index=0,
                        total_items=1,
                        timestamp=time.time(),
                        query_item={"text": "q"},
                        score_items=[{"text": "a", "id": "doc-a"}],
                    ),
                    # query_item.text must be str | None — an int is rejected at the seam.
                    ScoreBatchItem(
                        work_item_id="req-1.1",
                        request_id="req-1",
                        item_index=1,
                        total_items=1,
                        timestamp=time.time(),
                        query_item={"text": 123},
                        score_items=[{"text": "a", "id": "doc-a"}],
                    ),
                ],
            )
        )

        by_id = {o.work_item_id: o for o in outcome.outcomes}
        assert by_id["req-1.0"].disposition == "publish_and_ack"
        assert by_id["req-1.1"].disposition == "publish_error_and_ack"
        assert by_id["req-1.1"].error_code == "INVALID_INPUT"
        assert worker.submit_score_preformed_batch.await_count == 1
        requests = worker.submit_score_preformed_batch.await_args.args[0]
        assert len(requests) == 1

    @pytest.mark.asyncio
    async def test_multimodal_score_items_contribute_media_batch_cost(self) -> None:
        reg = _make_registry()
        worker = AsyncMock()
        score_output = ScoreOutput(scores=np.array([0.7], dtype=np.float32))
        wr = WorkerResult(output=score_output, timing=RequestTiming())
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(wr)
        worker.submit_score_preformed_batch = AsyncMock(return_value=[fut])
        reg.start_worker = AsyncMock(return_value=worker)

        ex = QueueExecutor(reg)
        await ex.process_score_batch(
            ProcessScoreBatchRequest(
                model_id="test/model",
                items=[
                    ScoreBatchItem(
                        work_item_id="req-1.0",
                        request_id="req-1",
                        item_index=0,
                        total_items=1,
                        timestamp=time.time(),
                        query_item={"text": "query"},
                        score_items=[{"id": "doc-image", "images": [{"data": b"fake-png", "format": "png"}]}],
                    )
                ],
            )
        )

        requests = worker.submit_score_preformed_batch.await_args.args[0]
        prepared_items = requests[0].prepared_items
        assert prepared_items[0].cost == 5 + 1024

    @pytest.mark.asyncio
    async def test_model_evicted_naks_individual_item(self) -> None:
        reg = _make_registry()
        reg.start_worker = AsyncMock(side_effect=KeyError("test/model"))
        ex = QueueExecutor(reg)

        outcome = await ex.process_score_batch(
            ProcessScoreBatchRequest(
                model_id="test/model",
                items=[
                    ScoreBatchItem(
                        work_item_id="req-1.0",
                        request_id="req-1",
                        item_index=0,
                        total_items=1,
                        timestamp=time.time(),
                        query_item={"text": "q"},
                        score_items=[{"text": "a"}],
                    )
                ],
            )
        )
        assert outcome.outcomes[0].disposition == "nak_retry"

    @pytest.mark.asyncio
    async def test_score_oom_naks_without_publishing_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_OOM_NAK_DELAY_S", "12.0")
        reg = _make_registry()
        worker = AsyncMock()
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_exception(RuntimeError("MPS backend out of memory"))
        worker.submit_score_preformed_batch = AsyncMock(return_value=[fut])
        reg.start_worker = AsyncMock(return_value=worker)

        ex = QueueExecutor(reg)
        outcome = await ex.process_score_batch(ProcessScoreBatchRequest(model_id="test/model", items=[_score_item()]))

        assert outcome.outcomes[0].disposition == "nak_retry"
        assert outcome.outcomes[0].nak_delay_ms == 12_000
        assert outcome.outcomes[0].error_code is None
        assert outcome.outcomes[0].error is None


# -----------------------------------------------------------------------------
# process_extract_batch
# -----------------------------------------------------------------------------


class TestProcessExtractBatch:
    @pytest.mark.asyncio
    async def test_single_item_publish_and_ack(self) -> None:
        reg = _make_registry()
        worker = AsyncMock()
        extract_output = ExtractOutput(
            entities=[[{"text": "Alice", "label": "person", "score": 0.99, "start": 0, "end": 5}]]
        )
        wr = WorkerResult(output=extract_output, timing=RequestTiming())
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(wr)
        worker.submit_extract_preformed_batch = AsyncMock(return_value=[fut])
        reg.start_worker = AsyncMock(return_value=worker)

        ex = QueueExecutor(reg)
        outcome = await ex.process_extract_batch(
            ProcessExtractBatchRequest(
                model_id="test/model",
                items=[
                    ExtractBatchItem(
                        work_item_id="req-1.0",
                        request_id="req-1",
                        item_index=0,
                        total_items=1,
                        timestamp=time.time(),
                        item={"text": "Alice works at Acme."},
                        labels=["person"],
                    )
                ],
            )
        )

        assert outcome.outcomes[0].disposition == "publish_and_ack"
        inner = msgpack.unpackb(outcome.outcomes[0].result_msgpack, raw=False)
        assert "entities" in inner

    @pytest.mark.asyncio
    async def test_effective_profile_options_drive_lora_grouping_and_invalid_is_per_item(self) -> None:
        reg = _make_registry()
        reg.get_config.return_value = _runtime_profile_config()
        worker = AsyncMock()
        calls: list[tuple[str | None, list]] = []

        async def submit(requests, *, lora):
            calls.append((lora, list(requests)))
            futures: list[asyncio.Future[WorkerResult]] = []
            for _ in requests:
                fut: asyncio.Future[WorkerResult] = asyncio.Future()
                fut.set_result(
                    WorkerResult(
                        output=ExtractOutput(entities=[[]]),
                        timing=RequestTiming(),
                    )
                )
                futures.append(fut)
            return futures

        worker.submit_extract_preformed_batch = AsyncMock(side_effect=submit)
        reg.start_worker = AsyncMock(return_value=worker)

        outcome = await QueueExecutor(reg).process_extract_batch(
            ProcessExtractBatchRequest(
                model_id="test/model",
                items=[
                    _extract_item(
                        wiid="profile.0",
                        options={"profile": "tuned", "threshold": 0.75},
                    ),
                    _extract_item(
                        wiid="override.0",
                        options={
                            "profile": "tuned",
                            "lora": "request-lora",
                        },
                    ),
                    _extract_item(
                        wiid="invalid.0",
                        options={"profile": "missing"},
                    ),
                ],
            )
        )

        by_id = {item.work_item_id: item for item in outcome.outcomes}
        assert by_id["profile.0"].disposition == "publish_and_ack"
        assert by_id["override.0"].disposition == "publish_and_ack"
        assert by_id["invalid.0"].disposition == "publish_error_and_ack"
        assert by_id["invalid.0"].error_code == "INVALID_INPUT"

        assert len(calls) == 2
        by_lora = dict(calls)
        assert set(by_lora) == {"profile-lora", "request-lora"}
        assert by_lora["profile-lora"][0].options == {
            "threshold": 0.75,
            "lora": "profile-lora",
        }
        assert by_lora["request-lora"][0].options == {
            "threshold": 0.5,
            "lora": "request-lora",
        }

    @pytest.mark.asyncio
    async def test_malformed_item_isolated_as_invalid_input(self) -> None:
        """A typed-decode failure is isolated as INVALID_INPUT while a sibling
        valid item still runs. Regression for #1537 (the extract path now
        decodes through the shared decode_item seam).
        """
        reg = _make_registry()
        worker = AsyncMock()
        extract_output = ExtractOutput(
            entities=[[{"text": "Alice", "label": "person", "score": 0.99, "start": 0, "end": 5}]]
        )
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(WorkerResult(output=extract_output, timing=RequestTiming()))
        worker.submit_extract_preformed_batch = AsyncMock(return_value=[fut])
        reg.start_worker = AsyncMock(return_value=worker)

        ex = QueueExecutor(reg)
        outcome = await ex.process_extract_batch(
            ProcessExtractBatchRequest(
                model_id="test/model",
                items=[
                    ExtractBatchItem(
                        work_item_id="req-1.0",
                        request_id="req-1",
                        item_index=0,
                        total_items=1,
                        timestamp=time.time(),
                        item={"text": "Alice works at Acme."},
                        labels=["person"],
                    ),
                    # item.text must be str | None — an int is rejected at the seam.
                    ExtractBatchItem(
                        work_item_id="req-1.1",
                        request_id="req-1",
                        item_index=1,
                        total_items=1,
                        timestamp=time.time(),
                        item={"text": 123},
                        labels=["person"],
                    ),
                ],
            )
        )

        by_id = {o.work_item_id: o for o in outcome.outcomes}
        assert by_id["req-1.0"].disposition == "publish_and_ack"
        assert by_id["req-1.1"].disposition == "publish_error_and_ack"
        assert by_id["req-1.1"].error_code == "INVALID_INPUT"
        assert worker.submit_extract_preformed_batch.await_count == 1
        requests = worker.submit_extract_preformed_batch.await_args.args[0]
        assert len(requests) == 1

    @pytest.mark.asyncio
    async def test_model_evicted_naks(self) -> None:
        reg = _make_registry()
        reg.start_worker = AsyncMock(side_effect=RuntimeError("evicted"))
        ex = QueueExecutor(reg)
        outcome = await ex.process_extract_batch(
            ProcessExtractBatchRequest(
                model_id="test/model",
                items=[
                    ExtractBatchItem(
                        work_item_id="req-1.0",
                        request_id="req-1",
                        item_index=0,
                        total_items=1,
                        timestamp=time.time(),
                        item={"text": "x"},
                    )
                ],
            )
        )
        assert outcome.outcomes[0].disposition == "nak_retry"

    @pytest.mark.asyncio
    async def test_extract_oom_naks_without_publishing_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_OOM_NAK_DELAY_S", "13.0")
        reg = _make_registry()
        worker = AsyncMock()
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_exception(RuntimeError("CUDA out of memory"))
        worker.submit_extract_preformed_batch = AsyncMock(return_value=[fut])
        reg.start_worker = AsyncMock(return_value=worker)

        ex = QueueExecutor(reg)
        outcome = await ex.process_extract_batch(
            ProcessExtractBatchRequest(model_id="test/model", items=[_extract_item()])
        )

        assert outcome.outcomes[0].disposition == "nak_retry"
        assert outcome.outcomes[0].nak_delay_ms == 13_000
        assert outcome.outcomes[0].error_code is None
        assert outcome.outcomes[0].error is None


# -----------------------------------------------------------------------------
# _wrap_encode_output: queue path MUST match HTTP wire shape for every
# output_type the SDK understands. Otherwise HTTP and queue consumers
# see different encodings for the same model.
# -----------------------------------------------------------------------------


class TestWrapEncodeOutput:
    @staticmethod
    def _config(
        *,
        dense_dim: int | None = None,
        sparse_dim: int | None = None,
        multivector_dim: int | None = None,
    ) -> MagicMock:
        encode_task = MagicMock()
        encode_task.dense = MagicMock(dim=dense_dim) if dense_dim is not None else None
        encode_task.sparse = MagicMock(dim=sparse_dim) if sparse_dim is not None else None
        encode_task.multivector = MagicMock(dim=multivector_dim) if multivector_dim is not None else None
        cfg = MagicMock()
        cfg.tasks = MagicMock(encode=encode_task)
        return cfg

    def test_dense_float32_wraps_with_dims_and_dtype(self) -> None:
        from sie_server.queue_executor import _wrap_encode_output

        arr = np.zeros(768, dtype=np.float32)
        wrapped = _wrap_encode_output({"dense": arr}, self._config(dense_dim=768))
        assert wrapped["dense"]["dims"] == 768
        assert wrapped["dense"]["dtype"] == "float32"
        assert np.array_equal(wrapped["dense"]["values"], arr)

    def test_sparse_wraps_like_http_format_sparse(self) -> None:
        from sie_server.queue_executor import _wrap_encode_output

        indices = np.array([1, 7, 42], dtype=np.int64)
        values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        wrapped = _wrap_encode_output(
            {"sparse": {"indices": indices, "values": values}},
            self._config(sparse_dim=30522),
        )
        sp = wrapped["sparse"]
        assert sp["dims"] == 30522
        assert sp["dtype"] == "float32"
        assert np.array_equal(sp["indices"], indices)
        assert np.array_equal(sp["values"], values)

    def test_multivector_wraps_with_token_dims_and_num_tokens(self) -> None:
        from sie_server.queue_executor import _wrap_encode_output

        arr = np.zeros((12, 128), dtype=np.float32)
        wrapped = _wrap_encode_output({"multivector": arr}, self._config(multivector_dim=128))
        mv = wrapped["multivector"]
        assert mv["token_dims"] == 128
        assert mv["num_tokens"] == 12
        assert mv["dtype"] == "float32"
        assert np.array_equal(mv["values"], arr)

    def test_binary_multivector_sets_dtype_binary(self) -> None:
        from sie_server.queue_executor import _wrap_encode_output

        arr = np.zeros((7, 16), dtype=np.uint8)  # 128-dim packed into 16 bytes/token
        wrapped = _wrap_encode_output({"multivector": arr}, self._config(multivector_dim=128))
        mv = wrapped["multivector"]
        assert mv["token_dims"] == 128
        assert mv["num_tokens"] == 7
        assert mv["dtype"] == "binary"

    def test_full_width_uint8_dense_stays_uint8(self) -> None:
        """A full-width uint8 dense vector (NOT bit-packed) must keep the
        ``uint8`` label — only the shape-based binary check may emit ``binary``,
        else the SDK would bit-unpack values that were never packed.
        """
        from sie_server.queue_executor import _wrap_encode_output

        arr = np.zeros(128, dtype=np.uint8)  # full width == dense_dim, not packed
        wrapped = _wrap_encode_output({"dense": arr}, self._config(dense_dim=128))
        assert wrapped["dense"]["dims"] == 128
        assert wrapped["dense"]["dtype"] == "uint8"

    def test_full_width_uint8_multivector_stays_uint8(self) -> None:
        """A full-width uint8 multivector (shape[1] == mv_dim) is not bit-packed,
        so it must keep the ``uint8`` label rather than fall through to binary.
        """
        from sie_server.queue_executor import _wrap_encode_output

        arr = np.zeros((5, 128), dtype=np.uint8)  # full width == mv_dim, not packed
        wrapped = _wrap_encode_output({"multivector": arr}, self._config(multivector_dim=128))
        mv = wrapped["multivector"]
        assert mv["token_dims"] == 128
        assert mv["num_tokens"] == 5
        assert mv["dtype"] == "uint8"
