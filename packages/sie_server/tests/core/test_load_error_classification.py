"""Load failures are classified by cause, with real exception types from the locked libraries."""

from __future__ import annotations

import errno
import socket
from pathlib import Path
from unittest.mock import patch

import boto3.exceptions
import botocore.exceptions
import httpx
import pytest
import requests
import s3transfer.exceptions
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    DisabledRepoError,
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from sie_sdk.cache import CacheConfig, _download_from_huggingface
from sie_sdk.exceptions import GatedModelError
from sie_server.adapters.sglang import _server as sglang_server
from sie_server.core.load_errors import (
    MAX_COOLDOWN_S,
    MAX_TRANSIENT_ATTEMPTS,
    DevicePlacementError,
    EngineExitedError,
    EngineStartupError,
    LoadErrorClass,
    ModelConfigurationError,
    ModelLoadTimeoutError,
    classify_load_error,
    cooldown_for,
)
from sie_server.core.loader import reject_unknown_loadtime_options


def _response(status: int) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = "https://huggingface.co/api/models/org/model"
    return response


def _hub_error(status: int) -> HfHubHTTPError:
    return HfHubHTTPError(f"{status} error", response=_response(status))


def _aws_error(code: str, status: int) -> botocore.exceptions.ClientError:
    return botocore.exceptions.ClientError(
        {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "GetObject",
    )


def _httpx_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.com/weights")
    return httpx.HTTPStatusError(f"{status} error", request=request, response=httpx.Response(status, request=request))


def _caused_by(outer: BaseException, cause: BaseException) -> BaseException:
    outer.__cause__ = cause
    return outer


def _raised_while_handling(outer: BaseException, context: BaseException) -> BaseException:
    try:
        try:
            raise context
        except type(context):
            raise outer  # noqa: B904
    except type(outer) as raised:
        return raised


TRANSIENT_CASES: list[tuple[str, BaseException, LoadErrorClass]] = [
    ("hub 408", _hub_error(408), LoadErrorClass.NETWORK),
    ("hub 429", _hub_error(429), LoadErrorClass.NETWORK),
    ("hub 500", _hub_error(500), LoadErrorClass.NETWORK),
    ("hub 502", _hub_error(502), LoadErrorClass.NETWORK),
    ("hub 503", _hub_error(503), LoadErrorClass.NETWORK),
    ("hub 504", _hub_error(504), LoadErrorClass.NETWORK),
    ("s3 slowdown", _aws_error("SlowDown", 503), LoadErrorClass.NETWORK),
    ("aws throttling on 400", _aws_error("Throttling", 400), LoadErrorClass.NETWORK),
    ("aws request timeout on 400", _aws_error("RequestTimeout", 400), LoadErrorClass.NETWORK),
    ("s3 internal error", _aws_error("InternalError", 500), LoadErrorClass.NETWORK),
    (
        "botocore endpoint connection",
        botocore.exceptions.EndpointConnectionError(endpoint_url="https://s3.amazonaws.com"),
        LoadErrorClass.NETWORK,
    ),
    (
        "botocore connect timeout",
        botocore.exceptions.ConnectTimeoutError(endpoint_url="https://s3.amazonaws.com"),
        LoadErrorClass.NETWORK,
    ),
    (
        "botocore read timeout",
        botocore.exceptions.ReadTimeoutError(endpoint_url="https://s3.amazonaws.com"),
        LoadErrorClass.NETWORK,
    ),
    (
        "botocore connection closed",
        botocore.exceptions.ConnectionClosedError(endpoint_url="https://s3.amazonaws.com"),
        LoadErrorClass.NETWORK,
    ),
    (
        "boto3 retries exceeded",
        boto3.exceptions.RetriesExceededError(
            botocore.exceptions.EndpointConnectionError(endpoint_url="https://s3.amazonaws.com")
        ),
        LoadErrorClass.NETWORK,
    ),
    (
        "s3transfer retries exceeded",
        s3transfer.exceptions.RetriesExceededError(_aws_error("SlowDown", 503)),
        LoadErrorClass.NETWORK,
    ),
    ("requests connection", requests.exceptions.ConnectionError("connection refused"), LoadErrorClass.NETWORK),
    ("requests read timeout", requests.exceptions.ReadTimeout("read timed out"), LoadErrorClass.NETWORK),
    ("requests chunked", requests.exceptions.ChunkedEncodingError("connection broken"), LoadErrorClass.NETWORK),
    ("httpx connect", httpx.ConnectError("connection refused"), LoadErrorClass.NETWORK),
    ("httpx read timeout", httpx.ReadTimeout("read timed out"), LoadErrorClass.NETWORK),
    ("httpx 503", _httpx_status_error(503), LoadErrorClass.NETWORK),
    ("builtin connection", ConnectionResetError("reset by peer"), LoadErrorClass.NETWORK),
    ("builtin timeout", TimeoutError("socket read timeout"), LoadErrorClass.NETWORK),
    ("dns", socket.gaierror(socket.EAI_NONAME, "Name or service not known"), LoadErrorClass.NETWORK),
    ("enospc", OSError(errno.ENOSPC, "No space left on device"), LoadErrorClass.STORAGE),
    ("edquot", OSError(errno.EDQUOT, "Disk quota exceeded"), LoadErrorClass.STORAGE),
    (
        "native io error text",
        RuntimeError("Data processing error: No space left on device (os error 28)"),
        LoadErrorClass.STORAGE,
    ),
    ("engine crash", sglang_server.startup_failure_error(None, crash_exit_code=1), LoadErrorClass.ENGINE),
    ("engine exit", EngineExitedError("process exited with code -9"), LoadErrorClass.ENGINE),
    ("port in use", OSError(errno.EADDRINUSE, "Address already in use"), LoadErrorClass.ENGINE),
    ("placement", DevicePlacementError("no free block"), LoadErrorClass.PLACEMENT),
    (
        "load timeout",
        ModelLoadTimeoutError(model="m", stage="load", elapsed_s=10.0, timeout_s=5.0),
        LoadErrorClass.TIMEOUT,
    ),
    ("cuda oom", RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"), LoadErrorClass.OOM),
    ("unrecognised", RuntimeError("some unrelated failure"), LoadErrorClass.UNKNOWN),
    ("hub 400", _hub_error(400), LoadErrorClass.UNKNOWN),
    ("httpx 404", _httpx_status_error(404), LoadErrorClass.UNKNOWN),
    ("s3 access denied", _aws_error("AccessDenied", 403), LoadErrorClass.UNKNOWN),
]

PERMANENT_CASES: list[tuple[str, BaseException, LoadErrorClass]] = [
    ("sdk gated", GatedModelError("org/private", RuntimeError("401 unauthorized")), LoadErrorClass.GATED),
    ("hub gated", GatedRepoError("gated", response=_response(403)), LoadErrorClass.GATED),
    ("hub 401", _hub_error(401), LoadErrorClass.GATED),
    ("hub repo missing", RepositoryNotFoundError("missing", response=_response(404)), LoadErrorClass.NOT_FOUND),
    ("hub revision missing", RevisionNotFoundError("missing", response=_response(404)), LoadErrorClass.NOT_FOUND),
    ("hub file missing", EntryNotFoundError("missing", response=_response(404)), LoadErrorClass.NOT_FOUND),
    ("hub repo disabled", DisabledRepoError("disabled", response=_response(403)), LoadErrorClass.NOT_FOUND),
    ("import", ImportError("Gemma3TextModel not found in transformers"), LoadErrorClass.DEPENDENCY),
    ("module", ModuleNotFoundError("no module named transformers"), LoadErrorClass.DEPENDENCY),
    ("configuration", ModelConfigurationError("has no mlx_repo set"), LoadErrorClass.CONFIG),
    ("validation", ValueError("pooling_method is not a load-time option"), LoadErrorClass.CONFIG),
    ("hub offline", OfflineModeIsEnabled("offline mode is enabled"), LoadErrorClass.CONFIG),
    (
        "offline and not cached",
        LocalEntryNotFoundError("Cannot find an appropriate cached snapshot folder"),
        LoadErrorClass.CONFIG,
    ),
]

CHAINED_CASES: list[tuple[str, BaseException, LoadErrorClass]] = [
    (
        "hub download without a cached snapshot",
        _caused_by(
            LocalEntryNotFoundError("cannot find the requested files on the Hub or in the local cache"),
            requests.exceptions.ConnectionError("connection refused"),
        ),
        LoadErrorClass.NETWORK,
    ),
    (
        "hub outage without a cached snapshot",
        _caused_by(LocalEntryNotFoundError("cannot find the requested files"), _hub_error(503)),
        LoadErrorClass.NETWORK,
    ),
    (
        "sdk not-found wrapper",
        _caused_by(
            RuntimeError("Model 'org/model' not found."), RepositoryNotFoundError("missing", response=_response(404))
        ),
        LoadErrorClass.NOT_FOUND,
    ),
    (
        "port reservation",
        _caused_by(EngineStartupError("port 30000 is already in use"), OSError(errno.EADDRINUSE, "in use")),
        LoadErrorClass.ENGINE,
    ),
    (
        "typed wrapper wins over its cause",
        _caused_by(ModelLoadTimeoutError(model="m", stage="load", elapsed_s=1.0, timeout_s=1.0), ConnectionError("x")),
        LoadErrorClass.TIMEOUT,
    ),
    (
        "sdk gated wrapper wins over its 503 cause",
        _caused_by(GatedModelError("org/private", RuntimeError("503")), _hub_error(503)),
        LoadErrorClass.GATED,
    ),
    (
        "transient cause beats an untyped validation wrapper",
        _caused_by(ValueError("could not read config.json"), ConnectionError("connection reset")),
        LoadErrorClass.NETWORK,
    ),
    (
        "implicit context",
        _raised_while_handling(RuntimeError("download failed"), OSError(errno.ENOSPC, "No space left on device")),
        LoadErrorClass.STORAGE,
    ),
]


def _ids(cases: list[tuple[str, BaseException, LoadErrorClass]]) -> list[str]:
    return [case[0] for case in cases]


class TestClassifyByCause:
    @pytest.mark.parametrize(("name", "exc", "expected"), TRANSIENT_CASES, ids=_ids(TRANSIENT_CASES))
    def test_transient_causes_retry_after_a_cooldown(
        self, name: str, exc: BaseException, expected: LoadErrorClass
    ) -> None:
        result = classify_load_error(exc)

        assert result.error_class is expected
        assert not result.is_permanent
        assert result.cooldown_s == cooldown_for(expected, 1)

    @pytest.mark.parametrize(("name", "exc", "expected"), PERMANENT_CASES, ids=_ids(PERMANENT_CASES))
    def test_configuration_and_identity_errors_stay_permanent(
        self, name: str, exc: BaseException, expected: LoadErrorClass
    ) -> None:
        result = classify_load_error(exc)

        assert result.error_class is expected
        assert result.is_permanent

    @pytest.mark.parametrize(("name", "exc", "expected"), CHAINED_CASES, ids=_ids(CHAINED_CASES))
    def test_wrapped_errors_are_classified_by_what_they_wrap(
        self, name: str, exc: BaseException, expected: LoadErrorClass
    ) -> None:
        assert classify_load_error(exc).error_class is expected

    def test_unknown_errors_get_a_long_but_bounded_cooldown(self) -> None:
        result = classify_load_error(RuntimeError("some unrelated failure"))

        assert result.error_class is LoadErrorClass.UNKNOWN
        assert result.cooldown_s == 600.0

    def test_a_self_referencing_chain_terminates(self) -> None:
        exc = RuntimeError("outer")
        inner = RuntimeError("inner")
        exc.__cause__ = inner
        inner.__cause__ = exc

        assert classify_load_error(exc).error_class is LoadErrorClass.UNKNOWN

    def test_an_unknown_load_time_option_stays_permanent(self) -> None:
        from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter

        with pytest.raises(ValueError, match="pooling_method") as caught:
            reject_unknown_loadtime_options(SGLangEmbeddingAdapter, {"pooling_method": "cls"}, model_name="org/embed")

        result = classify_load_error(caught.value)
        assert result.error_class is LoadErrorClass.CONFIG
        assert result.is_permanent

    def test_a_missing_mlx_repo_stays_permanent(self) -> None:
        from sie_server.adapters.mlx.generation import MLXGenerationAdapter

        with pytest.raises(RuntimeError, match="mlx_repo") as caught:
            MLXGenerationAdapter(model_name_or_path="Qwen/Qwen3.6-27B").load("mps")

        result = classify_load_error(caught.value)
        assert result.error_class is LoadErrorClass.CONFIG
        assert result.is_permanent

    def test_a_port_held_by_another_process_is_an_engine_startup_error(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
            holder.bind(("localhost", 0))
            port = holder.getsockname()[1]
            with pytest.raises(EngineStartupError, match="already in use") as caught:
                sglang_server.reserve_port(port)

        result = classify_load_error(caught.value)
        assert result.error_class is LoadErrorClass.ENGINE
        assert not result.is_permanent


class TestHubDownloadFailures:
    """``snapshot_download`` rewrites fetch failures before the SDK cache wrapper sees them."""

    @pytest.mark.parametrize(
        ("api_error", "raised", "expected"),
        [
            (
                requests.exceptions.ConnectionError("connection refused"),
                LocalEntryNotFoundError,
                LoadErrorClass.NETWORK,
            ),
            (requests.exceptions.ReadTimeout("read timed out"), LocalEntryNotFoundError, LoadErrorClass.NETWORK),
            (_hub_error(429), LocalEntryNotFoundError, LoadErrorClass.NETWORK),
            (_hub_error(503), LocalEntryNotFoundError, LoadErrorClass.NETWORK),
            (OfflineModeIsEnabled("offline mode is enabled"), LocalEntryNotFoundError, LoadErrorClass.CONFIG),
            (_hub_error(401), GatedModelError, LoadErrorClass.GATED),
            (RepositoryNotFoundError("missing", response=_response(404)), RuntimeError, LoadErrorClass.NOT_FOUND),
        ],
        ids=["connection", "timeout", "429", "503", "offline", "401", "not-found"],
    )
    def test_the_class_follows_the_hub_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        api_error: BaseException,
        raised: type[BaseException],
        expected: LoadErrorClass,
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        config = CacheConfig(local_cache=tmp_path, cluster_cache=None, hf_fallback=True)

        with (
            patch("sie_sdk.cache._get_hf_ignore_patterns", return_value=[]),
            patch.object(HfApi, "repo_info", side_effect=api_error),
            pytest.raises(raised) as caught,
        ):
            _download_from_huggingface("org/model", config)

        assert classify_load_error(caught.value).error_class is expected


class TestBackoff:
    def test_transient_cooldowns_double_up_to_the_cap(self) -> None:
        cooldowns = [cooldown_for(LoadErrorClass.NETWORK, attempts) for attempts in range(1, 8)]

        assert cooldowns == [30.0, 60.0, 120.0, 240.0, 480.0, 960.0, MAX_COOLDOWN_S]

    def test_a_transient_failure_becomes_permanent_after_the_attempt_budget(self) -> None:
        assert cooldown_for(LoadErrorClass.NETWORK, MAX_TRANSIENT_ATTEMPTS) == MAX_COOLDOWN_S
        assert cooldown_for(LoadErrorClass.NETWORK, MAX_TRANSIENT_ATTEMPTS + 1) is None
        assert classify_load_error(_hub_error(503), attempts=MAX_TRANSIENT_ATTEMPTS + 1).is_permanent

    @pytest.mark.parametrize(
        "error_class",
        [LoadErrorClass.GATED, LoadErrorClass.NOT_FOUND, LoadErrorClass.DEPENDENCY, LoadErrorClass.CONFIG],
    )
    def test_permanent_classes_have_no_cooldown(self, error_class: LoadErrorClass) -> None:
        assert cooldown_for(error_class, 1) is None
