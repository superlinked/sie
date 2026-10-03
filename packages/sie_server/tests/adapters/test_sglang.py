"""Tests for SGLang embedding adapter (HTTP server mode)."""

import json
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sie_server.adapters.sglang import _server
from sie_server.adapters.sglang._server import STARTUP_TIMEOUT_S as _STARTUP_TIMEOUT_S
from sie_server.adapters.sglang._server import parse_device_index
from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter
from sie_server.types.inputs import Item

# Create a random generator for tests
_RNG = np.random.default_rng(42)


class TestSGLangEmbeddingAdapter:
    """Tests for SGLangEmbeddingAdapter with mocked HTTP server."""

    @pytest.fixture
    def adapter(self) -> "SGLangEmbeddingAdapter":
        """Create an adapter instance."""
        return SGLangEmbeddingAdapter(
            model_name_or_path="Qwen/Qwen3-Embedding-8B",
            normalize=True,
            max_seq_length=8192,
            mem_fraction_static=0.85,
            query_template="Instruct: {instruction}\nQuery:{text}",
            default_instruction="Given a query, retrieve relevant passages",
        )

    def test_capabilities(self) -> None:
        """Adapter reports correct capabilities."""
        adapter = SGLangEmbeddingAdapter("test-model")
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert caps.outputs == ["dense"]

    def test_load_contract_flags(self) -> None:
        assert SGLangEmbeddingAdapter.requires_main_thread is False
        assert SGLangEmbeddingAdapter.manages_own_load_timeout is True

    def test_load_required_memory_bytes_uses_mem_fraction_static(self) -> None:
        gb = 1024**3
        adapter = SGLangEmbeddingAdapter("test-model", mem_fraction_static=0.8)

        assert adapter.load_required_memory_bytes(device_type="cuda", device_total_bytes=10 * gb) == 9 * gb
        assert adapter.load_required_memory_bytes(device_type="cpu", device_total_bytes=10 * gb) is None

    def test_startup_timeout_prefers_explicit_profile_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in _server.STARTUP_TIMEOUT_ENV_VARS:
            monkeypatch.setenv(name, "120")

        assert _server.resolve_startup_timeout(1800) == 1800

    def test_startup_timeout_accepts_operator_aliases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in _server.STARTUP_TIMEOUT_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("SIE_MODEL_READY_TIMEOUT_S", "1234")

        assert _server.resolve_startup_timeout() == 1234

    def test_startup_timeout_skips_non_finite_environment_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in _server.STARTUP_TIMEOUT_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("SIE_MODEL_READY_TIMEOUT_S", "inf")
        monkeypatch.setenv("SIE_SERVER_STARTUP_TIMEOUT_S", "1200")

        assert _server.resolve_startup_timeout() == 1200

    @pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), True, "900"])
    def test_startup_timeout_refuses_an_invalid_profile_value(self, monkeypatch: pytest.MonkeyPatch, bad) -> None:
        """A declared budget is used exactly or refused, never replaced by a fallback."""
        for name in _server.STARTUP_TIMEOUT_ENV_VARS:
            monkeypatch.setenv(name, "1200")

        with pytest.raises(ValueError, match="startup_timeout_s"):
            _server.resolve_startup_timeout(bad)

    def test_embedding_adapter_refuses_an_invalid_profile_startup_timeout(self) -> None:
        with pytest.raises(ValueError, match="startup_timeout_s"):
            SGLangEmbeddingAdapter("test-model", startup_timeout_s=0)

    def test_startup_timeout_rejects_liveness_budget_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in _server.STARTUP_TIMEOUT_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("SIE_WORKER_LIVENESS_BUDGET_S", "900")

        with pytest.raises(ValueError, match="SIE_WORKER_LIVENESS_BUDGET_S"):
            _server.resolve_startup_timeout(900)

    def test_dims_before_load_returns_none(self) -> None:
        """Dims returns None before first encode."""
        adapter = SGLangEmbeddingAdapter("test-model")
        assert adapter.dims.dense is None

    @patch("sie_server.adapters.sglang._server.subprocess.Popen")
    @patch("sie_server.adapters.sglang._server.requests.get")
    @patch("sie_server.adapters.sglang._server.find_free_port")
    def test_load(
        self,
        mock_find_port: MagicMock,
        mock_requests_get: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """Load starts SGLang server subprocess."""
        # Setup mocks
        mock_find_port.return_value = 30000
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process still running
        mock_popen.return_value = mock_process
        mock_requests_get.return_value = MagicMock(status_code=200)

        adapter = SGLangEmbeddingAdapter(
            model_name_or_path="Qwen/Qwen3-Embedding-8B",
            mem_fraction_static=0.85,
            compute_precision="bfloat16",
        )
        adapter.load("cuda:0")

        # Verify Popen was called
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]

        assert "python" in cmd[0]
        assert "-m" in cmd
        assert "sglang.launch_server" in cmd
        assert "--model-path" in cmd
        assert "Qwen/Qwen3-Embedding-8B" in cmd
        assert "--is-embedding" in cmd
        assert "--port" in cmd
        assert "30000" in cmd
        assert "--dtype" in cmd
        assert "bfloat16" in cmd

        # Verify health check was called
        mock_requests_get.assert_called()

        # Verify server URL is set
        assert adapter._server_url == "http://localhost:30000"

    @patch("sie_server.adapters.sglang._server.subprocess.Popen")
    @patch("sie_server.adapters.sglang._server.requests.get")
    @patch("sie_server.adapters.sglang._server.find_free_port")
    def test_load_trust_remote_code_flag(
        self,
        mock_find_port: MagicMock,
        mock_requests_get: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """--trust-remote-code is emitted only when trust_remote_code is set.

        gte-Qwen2-7B-instruct opts out (trust_remote_code=False) so SGLang loads
        the stock tokenizer instead of the model's fragile remote-code tokenizer.
        """
        mock_find_port.return_value = 30000
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        mock_requests_get.return_value = MagicMock(status_code=200)

        def _cmd_for(*, trust_remote_code: bool) -> list[str]:
            mock_popen.reset_mock()
            adapter = SGLangEmbeddingAdapter(
                model_name_or_path="Alibaba-NLP/gte-Qwen2-7B-instruct",
                trust_remote_code=trust_remote_code,
            )
            adapter.load("cuda:0")
            return mock_popen.call_args[0][0]

        assert "--trust-remote-code" in _cmd_for(trust_remote_code=True)
        assert "--trust-remote-code" not in _cmd_for(trust_remote_code=False)

    @patch("sie_server.adapters.sglang._server.subprocess.Popen")
    @patch("sie_server.adapters.sglang._server.requests.get")
    @patch("sie_server.adapters.sglang._server.find_free_port")
    def test_load_different_device(
        self,
        mock_find_port: MagicMock,
        mock_requests_get: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """Load parses device index and sets CUDA_VISIBLE_DEVICES."""
        mock_find_port.return_value = 30001
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        mock_requests_get.return_value = MagicMock(status_code=200)

        adapter = SGLangEmbeddingAdapter("test-model")
        adapter.load("cuda:1")

        # Check that CUDA_VISIBLE_DEVICES was set to 1
        call_kwargs = mock_popen.call_args[1]
        env = call_kwargs.get("env", {})
        assert env.get("CUDA_VISIBLE_DEVICES") == "1"

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode(self, mock_post: MagicMock) -> None:
        """Encode returns dense embeddings via HTTP."""
        # Setup mock response - OpenAI-compatible format
        embeddings = _RNG.standard_normal((2, 4096)).astype(np.float32)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": embeddings[0].tolist(), "index": 0},
                {"embedding": embeddings[1].tolist(), "index": 1},
            ]
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Pin single-POST concurrency: the default fan-out (SIE_SGLANG_EMBED_
        # CONCURRENCY=8) shards this 2-text batch into two 1-text POSTs, but the
        # mock above serves 2 embeddings per POST — the shard/response mismatch
        # (a893384ba) is what broke this test. concurrency=1 keeps the legacy
        # single blocking POST so the asserts on the one call still hold.
        adapter = SGLangEmbeddingAdapter("test-model", normalize=False, embed_concurrency=1)
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="hello"), Item(text="world")]
        output = adapter.encode(items, output_types=["dense"])

        assert output.batch_size == 2
        assert output.dense is not None
        assert output.dense[0].shape == (4096,)

        # Verify HTTP call (now uses OpenAI-compatible endpoint)
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:30000/v1/embeddings"
        assert call_args[1]["json"]["input"] == ["hello", "world"]

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_normalizes(self, mock_post: MagicMock) -> None:
        """Encode normalizes embeddings when configured."""
        # Setup mock with non-normalized embeddings - OpenAI-compatible format
        embeddings = [3.0, 4.0, 0.0] + [0.0] * 4093
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": embeddings, "index": 0}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter("test-model", normalize=True)
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="test")]
        output = adapter.encode(items, output_types=["dense"])

        # Check normalization (3-4-5 triangle -> norm of 5)
        norm = np.linalg.norm(output.dense[0])
        assert abs(norm - 1.0) < 1e-5

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_with_query_template(self, mock_post: MagicMock) -> None:
        """Encode applies query template for queries."""
        embeddings = _RNG.standard_normal((1, 4096)).astype(np.float32)
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": embeddings[0].tolist(), "index": 0}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter(
            "test-model",
            query_template="Instruct: {instruction}\nQuery:{text}",
            default_instruction="search",
            normalize=False,
        )
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="hello")]
        adapter.encode(items, output_types=["dense"], is_query=True)

        # Check that the formatted text was passed (now uses "input" key)
        call_args = mock_post.call_args
        texts = call_args[1]["json"]["input"]
        assert texts == ["Instruct: search\nQuery:hello"]

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_with_instruction_override(self, mock_post: MagicMock) -> None:
        """Encode uses provided instruction over default."""
        embeddings = _RNG.standard_normal((1, 4096)).astype(np.float32)
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": embeddings[0].tolist(), "index": 0}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter(
            "test-model",
            query_template="Instruct: {instruction}\nQuery:{text}",
            default_instruction="default",
            normalize=False,
        )
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="hello")]
        adapter.encode(items, output_types=["dense"], instruction="custom", is_query=True)

        call_args = mock_post.call_args
        texts = call_args[1]["json"]["input"]
        assert texts == ["Instruct: custom\nQuery:hello"]

    @patch("sie_server.adapters.sglang._server.os.getpgid")
    @patch("sie_server.adapters.sglang._server.os.killpg")
    def test_unload(self, mock_killpg: MagicMock, mock_getpgid: MagicMock) -> None:
        """Unload stops the SGLang server subprocess."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.wait.return_value = None
        mock_getpgid.return_value = 12345  # Return same as pid

        adapter = SGLangEmbeddingAdapter("test-model")
        adapter._process = mock_process
        adapter._server_url = "http://localhost:30000"
        adapter.unload()

        # Verify process was terminated
        mock_killpg.assert_called()
        mock_getpgid.assert_called_with(12345)
        assert adapter._server_url is None
        assert adapter._process is None

    @patch("sie_server.adapters.sglang._server.os.getpgid")
    @patch("sie_server.adapters.sglang._server.os.killpg")
    def test_unload_releases_port_and_cleans_output_log(
        self,
        mock_killpg: MagicMock,
        mock_getpgid: MagicMock,
    ) -> None:
        """Unload returns the reserved port to the pool and removes the temp log."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.wait.return_value = None
        mock_getpgid.return_value = 12345

        adapter = SGLangEmbeddingAdapter("test-model")
        adapter._process = mock_process
        adapter._server_url = "http://localhost:30000"
        adapter._port = 30000
        adapter._output_file = _server.open_output_log(prefix="sie_test_sglang_")
        log_path = Path(adapter._output_file.name)

        # Order matters: releasing the port before the child is down would hand
        # a live port to a concurrent load, so record both calls on one list.
        events: list[str] = []
        with (
            patch(
                "sie_server.adapters.sglang._server.terminate_process",
                side_effect=lambda *_args, **_kwargs: events.append("terminate"),
            ),
            patch(
                "sie_server.adapters.sglang._server.release_port",
                side_effect=lambda port: events.append(f"release:{port}"),
            ),
        ):
            adapter.unload()

        assert events == ["terminate", "release:30000"]
        assert adapter._port is None
        assert adapter._output_file is None
        assert not log_path.exists()

    @patch("sie_server.adapters.sglang._server.os.getpgid")
    @patch("sie_server.adapters.sglang._server.os.killpg")
    @patch("sie_server.adapters.sglang._server.wait_for_server", return_value=False)
    @patch("sie_server.adapters.sglang._server.subprocess.Popen")
    @patch("sie_server.adapters.sglang._server.find_free_port")
    def test_load_failure_releases_port_and_cleans_output_log(
        self,
        mock_find_port: MagicMock,
        mock_popen: MagicMock,
        mock_wait: MagicMock,
        mock_killpg: MagicMock,
        mock_getpgid: MagicMock,
    ) -> None:
        """A startup-health failure must not leak the reserved port or the temp log."""
        mock_find_port.return_value = 30000
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None  # Process running but not healthy (timeout path)
        mock_popen.return_value = mock_process
        mock_getpgid.return_value = 12345

        # A real temp file, so the assertion below is "gone from disk" rather
        # than the weaker "attribute was reset".
        real_log = _server.open_output_log(prefix="sie_test_sglang_")
        log_path = Path(real_log.name)

        adapter = SGLangEmbeddingAdapter("test-model")
        with (
            patch("sie_server.adapters.sglang._server.open_output_log", return_value=real_log),
            patch("sie_server.adapters.sglang._server.release_port") as mock_release,
            pytest.raises(RuntimeError, match="failed to start"),
        ):
            adapter.load("cuda:0")

        mock_release.assert_called_once_with(30000)
        assert adapter._port is None
        assert adapter._server_url is None
        assert adapter._output_file is None
        assert not log_path.exists()

    @patch("sie_server.adapters.sglang._server.open_output_log", side_effect=OSError("no space left on device"))
    @patch("sie_server.adapters.sglang._server.find_free_port")
    def test_output_log_failure_releases_port(
        self,
        mock_find_port: MagicMock,
        mock_open_log: MagicMock,
    ) -> None:
        """The port is reserved before the log is opened, so a /tmp failure must
        still hand it back — otherwise repeated failures exhaust the span.
        """
        mock_find_port.return_value = 30000

        adapter = SGLangEmbeddingAdapter("test-model")
        with (
            patch("sie_server.adapters.sglang._server.release_port") as mock_release,
            pytest.raises(OSError, match="no space left on device"),
        ):
            adapter.load("cuda:0")

        mock_release.assert_called_once_with(30000)
        assert adapter._port is None
        assert adapter._server_url is None

    @patch("sie_server.adapters.sglang._server.subprocess.Popen", side_effect=OSError("exec format error"))
    @patch("sie_server.adapters.sglang._server.find_free_port")
    def test_launch_failure_releases_port_and_cleans_output_log(
        self,
        mock_find_port: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """A failed exec must release the port and delete the log it opened."""
        mock_find_port.return_value = 30000

        # A real temp file, so the assertion below is "gone from disk" rather
        # than the weaker "attribute was reset".
        real_log = _server.open_output_log(prefix="sie_test_sglang_")
        log_path = Path(real_log.name)

        adapter = SGLangEmbeddingAdapter("test-model")
        with (
            patch("sie_server.adapters.sglang._server.open_output_log", return_value=real_log),
            patch("sie_server.adapters.sglang._server.release_port") as mock_release,
            pytest.raises(OSError, match="exec format error"),
        ):
            adapter.load("cuda:0")

        mock_release.assert_called_once_with(30000)
        assert adapter._port is None
        assert adapter._server_url is None
        assert adapter._output_file is None
        assert not log_path.exists()

    def test_encode_before_load_raises(self) -> None:
        """Encode before load raises error."""
        adapter = SGLangEmbeddingAdapter("test-model")
        items = [Item(text="test")]

        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["dense"])

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_unsupported_output_type_raises(self, mock_post: MagicMock) -> None:
        """Encode raises for unsupported output types."""
        adapter = SGLangEmbeddingAdapter("test-model")
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_requires_text(self, mock_post: MagicMock) -> None:
        """Encode raises for items without text."""
        adapter = SGLangEmbeddingAdapter("test-model")
        adapter._server_url = "http://localhost:30000"

        items = [Item()]  # No text
        with pytest.raises(ValueError, match="requires text"):
            adapter.encode(items, output_types=["dense"])

    def test_memory_footprint_before_load(self) -> None:
        """Memory footprint returns 0 before load."""
        adapter = SGLangEmbeddingAdapter("test-model")
        assert adapter.memory_footprint() == 0

    def test_memory_footprint_after_load(self) -> None:
        """Memory footprint returns 0 (uses system monitoring)."""
        adapter = SGLangEmbeddingAdapter("test-model")
        adapter._server_url = "http://localhost:30000"
        adapter._process = MagicMock()

        # SGLang adapter returns 0 because we rely on system GPU memory monitoring
        assert adapter.memory_footprint() == 0

    def test_parse_device_index(self) -> None:
        """Device index parsing works correctly.

        ``_parse_device_index`` moved to the shared ``_server`` module as
        :func:`parse_device_index` when the adapter package was split.
        """
        assert parse_device_index("cuda") == 0
        assert parse_device_index("cuda:0") == 0
        assert parse_device_index("cuda:1") == 1
        assert parse_device_index("cuda:7") == 7
        assert parse_device_index("cpu") == 0

    @patch("sie_server.adapters.sglang._server.os.getpgid")
    @patch("sie_server.adapters.sglang._server.os.killpg")
    @patch("sie_server.adapters.sglang._server.subprocess.Popen")
    @patch("sie_server.adapters.sglang._server.requests.get")
    @patch("sie_server.adapters.sglang._server.find_free_port")
    @patch("sie_server.adapters.sglang._server.time.monotonic")
    def test_load_timeout_raises(
        self,
        mock_monotonic: MagicMock,
        mock_find_port: MagicMock,
        mock_requests_get: MagicMock,
        mock_popen: MagicMock,
        mock_killpg: MagicMock,
        mock_getpgid: MagicMock,
    ) -> None:
        """Load raises if server fails to start within timeout."""
        mock_find_port.return_value = 30000
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process running but not healthy
        mock_process.stdout = MagicMock()
        mock_process.stdout.read.return_value = b"startup failed"
        mock_popen.return_value = mock_process
        mock_getpgid.return_value = 12345  # For cleanup

        # Simulate timeout
        mock_monotonic.side_effect = [0, _STARTUP_TIMEOUT_S + 1]
        mock_requests_get.side_effect = Exception("Connection refused")

        adapter = SGLangEmbeddingAdapter("test-model")
        with pytest.raises(RuntimeError, match="failed to start"):
            adapter.load("cuda:0")

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_detects_dimension(self, mock_post: MagicMock) -> None:
        """First encode detects embedding dimension."""
        embeddings = _RNG.standard_normal((1, 2048)).astype(np.float32)
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": embeddings[0].tolist(), "index": 0}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter("test-model", normalize=False)
        adapter._server_url = "http://localhost:30000"

        # Dimension not set initially
        assert adapter._dense_dim is None

        items = [Item(text="test")]
        adapter.encode(items, output_types=["dense"])

        # Dimension detected after first encode
        assert adapter._dense_dim == 2048
        assert adapter.dims.dense == 2048

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_rejects_configured_dense_dim_mismatch(self, mock_post: MagicMock) -> None:
        """Configured dense_dim mismatches fail before NumPy row assignment."""
        embeddings = _RNG.standard_normal((1, 2048)).astype(np.float32)
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": embeddings[0].tolist(), "index": 0}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter("test-model", normalize=False, dense_dim=1024)
        adapter._server_url = "http://localhost:30000"

        with pytest.raises(ValueError, match="configured dense_dim=1024, observed=2048"):
            adapter.encode([Item(text="test")], output_types=["dense"])

    def test_embed_texts_concurrent_preserves_order(self) -> None:
        """Sharded concurrent embedding concatenates shards back in input order."""
        adapter = SGLangEmbeddingAdapter("test-model", embed_concurrency=3)
        adapter._embed_concurrency = 3  # pin past any env override
        adapter._session = MagicMock()
        adapter._post_executor = ThreadPoolExecutor(max_workers=3)
        try:
            # One row per text valued by the text's int id, so a mis-ordered
            # concat across shards would be detectable.
            def fake_post(texts: list[str], model_name: str, session: object) -> np.ndarray:
                return np.array([[float(t)] for t in texts], dtype=np.float32)

            with patch.object(adapter, "_post_embedding_batch", side_effect=fake_post):
                out = adapter._embed_texts([str(i) for i in range(6)], "test-model")
            assert out.shape == (6, 1)
            assert [int(v) for v in out[:, 0]] == list(range(6))
        finally:
            adapter._post_executor.shutdown(wait=True)

    def test_embed_texts_concurrent_propagates_shard_error(self) -> None:
        """A failure in any shard propagates out of the concurrent gather."""
        adapter = SGLangEmbeddingAdapter("test-model", embed_concurrency=3)
        adapter._embed_concurrency = 3  # pin past any env override
        adapter._session = MagicMock()
        adapter._post_executor = ThreadPoolExecutor(max_workers=3)
        try:

            def fake_post(texts: list[str], model_name: str, session: object) -> np.ndarray:
                if "3" in texts:
                    raise RuntimeError("shard boom")
                return np.array([[float(t)] for t in texts], dtype=np.float32)

            with (
                patch.object(adapter, "_post_embedding_batch", side_effect=fake_post),
                pytest.raises(RuntimeError, match="shard boom"),
            ):
                adapter._embed_texts([str(i) for i in range(6)], "test-model")
        finally:
            adapter._post_executor.shutdown(wait=True)


class TestSGLangLoRA:
    """Tests for SGLang LoRA support."""

    def test_lora_paths_init(self) -> None:
        """Adapter accepts lora_paths at init."""
        lora_paths = {"legal": "org/legal-lora", "medical": "/path/to/medical"}
        adapter = SGLangEmbeddingAdapter("test-model", lora_paths=lora_paths)

        assert adapter._lora_paths == lora_paths
        assert adapter.available_loras == ["legal", "medical"]
        assert adapter.lora_enabled is True

    def test_lora_disabled_by_default(self) -> None:
        """LoRA is disabled when no lora_paths provided."""
        adapter = SGLangEmbeddingAdapter("test-model")

        assert adapter._lora_paths == {}
        assert adapter.available_loras == []
        assert adapter.lora_enabled is False

    def test_max_loras_per_batch_init(self) -> None:
        """Adapter accepts max_loras_per_batch at init."""
        adapter = SGLangEmbeddingAdapter("test-model", max_loras_per_batch=16)
        assert adapter._max_loras_per_batch == 16

    def test_max_loras_per_batch_default(self) -> None:
        """max_loras_per_batch defaults to 8."""
        adapter = SGLangEmbeddingAdapter("test-model")
        assert adapter._max_loras_per_batch == 8

    @patch("sie_server.adapters.sglang._server.subprocess.Popen")
    @patch("sie_server.adapters.sglang._server.requests.get")
    @patch("sie_server.adapters.sglang._server.find_free_port")
    def test_load_with_lora(
        self,
        mock_find_port: MagicMock,
        mock_requests_get: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """Load adds LoRA flags when lora_paths is provided."""
        mock_find_port.return_value = 30000
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        mock_requests_get.return_value = MagicMock(status_code=200)

        adapter = SGLangEmbeddingAdapter(
            model_name_or_path="test-model",
            lora_paths={"legal": "org/legal-lora", "medical": "/path/to/medical"},
            max_loras_per_batch=4,
        )
        adapter.load("cuda:0")

        # Verify command includes LoRA flags
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]

        assert "--enable-lora" in cmd
        assert "--lora-paths" in cmd
        # Find the lora-paths value
        lora_idx = cmd.index("--lora-paths")
        lora_paths_value = cmd[lora_idx + 1]
        assert "legal=" in lora_paths_value
        assert "medical=" in lora_paths_value

        assert "--max-loras-per-batch" in cmd
        batch_idx = cmd.index("--max-loras-per-batch")
        assert cmd[batch_idx + 1] == "4"

    @patch("sie_server.adapters.sglang._server.subprocess.Popen")
    @patch("sie_server.adapters.sglang._server.requests.get")
    @patch("sie_server.adapters.sglang._server.find_free_port")
    def test_load_without_lora_no_flags(
        self,
        mock_find_port: MagicMock,
        mock_requests_get: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """Load does not add LoRA flags when lora_paths is empty."""
        mock_find_port.return_value = 30000
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        mock_requests_get.return_value = MagicMock(status_code=200)

        adapter = SGLangEmbeddingAdapter(model_name_or_path="test-model")
        adapter.load("cuda:0")

        cmd = mock_popen.call_args[0][0]
        assert "--enable-lora" not in cmd
        assert "--lora-paths" not in cmd
        assert "--max-loras-per-batch" not in cmd

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_with_lora(self, mock_post: MagicMock) -> None:
        """Encode uses LoRA adapter name when set_active_lora() is called first."""
        embeddings = _RNG.standard_normal((1, 4096)).astype(np.float32)
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": embeddings[0].tolist(), "index": 0}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter(
            "test-model",
            lora_paths={"legal": "org/legal-lora"},
            normalize=False,
        )
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="test")]
        adapter.set_active_lora("legal")
        adapter.encode(items, output_types=["dense"])

        # Verify LoRA name is used as model name in request
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "legal"

    @patch("sie_server.adapters.sglang.embedding.requests.post")
    def test_encode_without_lora_uses_default(self, mock_post: MagicMock) -> None:
        """Encode uses 'default' model name when no lora is specified."""
        embeddings = _RNG.standard_normal((1, 4096)).astype(np.float32)
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": embeddings[0].tolist(), "index": 0}]}
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        adapter = SGLangEmbeddingAdapter(
            "test-model",
            lora_paths={"legal": "org/legal-lora"},
            normalize=False,
        )
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="test")]
        adapter.encode(items, output_types=["dense"])

        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "default"

    def test_encode_invalid_lora_raises(self) -> None:
        """Encode raises ValueError for unknown LoRA adapter set via set_active_lora."""
        adapter = SGLangEmbeddingAdapter(
            "test-model",
            lora_paths={"legal": "org/legal-lora"},
        )
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="test")]
        adapter.set_active_lora("unknown")
        with pytest.raises(ValueError, match="LoRA 'unknown' not loaded"):
            adapter.encode(items, output_types=["dense"])

    def test_encode_lora_without_any_loaded_raises(self) -> None:
        """Encode raises ValueError when LoRA set but none loaded."""
        adapter = SGLangEmbeddingAdapter("test-model")  # No lora_paths
        adapter._server_url = "http://localhost:30000"

        items = [Item(text="test")]
        adapter.set_active_lora("legal")
        with pytest.raises(ValueError, match=r"LoRA 'legal' not loaded.*Available: \[\]"):
            adapter.encode(items, output_types=["dense"])


class TestPortReservation:
    """find_free_port reserves ports; release_port returns them to the pool."""

    def test_release_returns_reserved_port_to_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_server, "_RESERVED_PORTS", set())
        port = _server.find_free_port()
        assert port in _server._RESERVED_PORTS
        _server.release_port(port)
        assert port not in _server._RESERVED_PORTS

    def test_exhausted_span_recovers_after_release(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Releasing one port un-bricks allocation after the span exhausts.

        This is the LRU eviction→reload churn scenario: without release_port
        the reserved set only grows, and once all 100 ports are reserved every
        generation-model load fails until a full process restart.
        """
        # Anchor the scan on a port the OS just proved bindable, so the test
        # doesn't depend on the state of the real SGLang range (30000-30099).
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            anchor = s.getsockname()[1]
        monkeypatch.setattr(_server, "_RESERVED_PORTS", set(range(anchor, anchor + 100)))
        with pytest.raises(RuntimeError, match="Could not find free port"):
            _server.find_free_port(anchor)
        _server.release_port(anchor)
        assert _server.find_free_port(anchor) == anchor

    def test_release_tolerates_none_and_unreserved_ports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_server, "_RESERVED_PORTS", set())
        _server.release_port(None)
        _server.release_port(54321)  # never reserved — must be a no-op
        assert not _server._RESERVED_PORTS


class TestKernelCacheEnvironment:
    def test_disabled_without_root(self) -> None:
        assert _server._kernel_cache_env({}, device_indices=[0]) == {}

    @patch("sie_server.adapters.sglang._server.subprocess.run")
    def test_gpu_key_uses_driver_inventory_without_initializing_cuda(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout="NVIDIA H100 80GB HBM3\nNVIDIA H200\n")

        first_key = _server._gpu_cache_key([0])
        second_key = _server._gpu_cache_key([1])

        assert first_key is not None
        assert first_key.startswith("gpu-")
        assert second_key is not None
        assert second_key.startswith("gpu-")
        assert first_key != second_key
        assert mock_run.call_args.args[0] == [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ]

    @patch("sie_server.adapters.sglang._server.subprocess.run")
    def test_gpu_key_is_shared_by_a_uniform_group(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout="NVIDIA L4\nNVIDIA L4\nNVIDIA L4\nNVIDIA L4\n")

        assert _server._gpu_cache_key([0, 1, 2, 3]) == _server._gpu_cache_key([0])

    @patch("sie_server.adapters.sglang._server.subprocess.run")
    def test_gpu_key_refuses_a_mixed_group(self, mock_run: MagicMock) -> None:
        """A mixed group must not reuse either member's artifacts.

        Kernels compiled for one product are not valid for the other, and a
        cache hit would serve them silently.
        """
        mock_run.return_value = MagicMock(stdout="NVIDIA L4\nNVIDIA H100 80GB HBM3\n")

        assert _server._gpu_cache_key([0, 1]) is None

    def test_scopes_upstream_caches_by_abi_and_gpu(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_server, "_installed_jit_abi_key", lambda: "abi123")
        monkeypatch.setattr(_server, "_gpu_cache_key", lambda _device_indices: "sm120-gpu123")

        env = {_server.KERNEL_CACHE_ROOT_ENV_VAR: str(tmp_path)}
        cache_env = _server._kernel_cache_env(env, device_indices=[0])

        namespace = tmp_path / "v1" / "abi123" / "sm120-gpu123" / "device-0"
        assert cache_env["SGLANG_DG_CACHE_DIR"] == str(namespace / "deep-gemm")
        assert cache_env["FLASHINFER_WORKSPACE_BASE"] == str(namespace / "flashinfer")
        assert cache_env["CUTE_DSL_CACHE_DIR"] == str(namespace / "cutlass")
        assert cache_env["TRITON_CACHE_DIR"] == str(namespace / "triton")
        assert all(Path(path).is_dir() for path in cache_env.values())

    def test_group_namespace_is_distinct_from_single_device(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Width-four artifacts must not be served to a width-one load.

        The namespace carries the whole group, so the same card at a different
        width reads a different directory.
        """
        monkeypatch.setattr(_server, "_installed_jit_abi_key", lambda: "abi123")
        monkeypatch.setattr(_server, "_gpu_cache_key", lambda _device_indices: "sm120-gpu123")
        env = {_server.KERNEL_CACHE_ROOT_ENV_VAR: str(tmp_path)}

        single = _server._kernel_cache_env(dict(env), device_indices=[0])
        group = _server._kernel_cache_env(dict(env), device_indices=[0, 1, 2, 3])

        assert single["TRITON_CACHE_DIR"].endswith("/device-0/triton")
        assert group["TRITON_CACHE_DIR"].endswith("/device-0_1_2_3/triton")
        assert single["TRITON_CACHE_DIR"] != group["TRITON_CACHE_DIR"]

    def test_explicit_upstream_path_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_server, "_installed_jit_abi_key", lambda: "abi123")
        monkeypatch.setattr(_server, "_gpu_cache_key", lambda _device_indices: "sm90-gpu123")
        explicit = tmp_path / "operator-deep-gemm"
        env = {
            _server.KERNEL_CACHE_ROOT_ENV_VAR: str(tmp_path / "managed"),
            "SGLANG_DG_CACHE_DIR": str(explicit),
        }

        cache_env = _server._kernel_cache_env(env, device_indices=[0])

        assert "SGLANG_DG_CACHE_DIR" not in cache_env
        assert not explicit.exists()
        assert "TRITON_CACHE_DIR" in cache_env

    def test_invalid_or_unwritable_root_falls_back(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_server, "_installed_jit_abi_key", lambda: "abi123")
        monkeypatch.setattr(_server, "_gpu_cache_key", lambda _device_indices: "sm90-gpu123")
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("x")

        assert _server._kernel_cache_env({_server.KERNEL_CACHE_ROOT_ENV_VAR: "relative"}, device_indices=[0]) == {}
        assert _server._kernel_cache_env({_server.KERNEL_CACHE_ROOT_ENV_VAR: str(blocker)}, device_indices=[0]) == {}

        with patch(
            "sie_server.adapters.sglang._server.tempfile.NamedTemporaryFile",
            side_effect=OSError("read-only cache"),
        ):
            assert (
                _server._kernel_cache_env(
                    {_server.KERNEL_CACHE_ROOT_ENV_VAR: str(tmp_path / "read-only")},
                    device_indices=[0],
                )
                == {}
            )


def test_startup_failure_error_distinguishes_crash_from_timeout() -> None:
    """A dead child is a crash, not a timeout: the loader reclassifies
    timeout-shaped messages as ModelLoadTimeoutError stamped with elapsed
    time, so the crash message must never match that pattern.
    """
    timeout_error = _server.startup_failure_error(None)
    assert "failed to start within timeout" in str(timeout_error)

    crash_error = _server.startup_failure_error(None, crash_exit_code=-9)
    assert "process exited during startup (exit code -9)" in str(crash_error)
    assert "failed to start within timeout" not in str(crash_error)


class TestWidthAgainstModelShape:
    """A width must divide the dimensions the engine shards across ranks.

    The engine asserts this itself, but only after loading weights, which on a
    multi-accelerator load holds every card in the group for minutes before the
    launch fails, and a supervisor that restarts the worker repeats that.
    """

    @staticmethod
    def _model_dir(tmp_path: Path, **config: object) -> str:
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return str(tmp_path)

    @pytest.mark.parametrize("width", [1, 2, 4])
    def test_a_dividing_width_is_accepted(self, tmp_path: Path, width: int) -> None:
        path = self._model_dir(tmp_path, num_attention_heads=28, num_key_value_heads=4)

        _server.validate_width_against_model_config(width, model_name_or_path=path)

    def test_a_width_that_splits_attention_heads_unevenly_is_refused(self, tmp_path: Path) -> None:
        path = self._model_dir(tmp_path, num_attention_heads=28, num_key_value_heads=28)

        with pytest.raises(ValueError, match="num_attention_heads=28"):
            _server.validate_width_against_model_config(8, model_name_or_path=path)

    def test_a_width_that_splits_key_value_heads_unevenly_is_refused(self, tmp_path: Path) -> None:
        """Neither a multiple nor a divisor: the heads can be neither split nor replicated."""
        path = self._model_dir(tmp_path, num_attention_heads=24, num_key_value_heads=6)

        with pytest.raises(ValueError, match="num_key_value_heads=6"):
            _server.validate_width_against_model_config(4, model_name_or_path=path)

    def test_fewer_key_value_heads_than_the_width_are_replicated(self, tmp_path: Path) -> None:
        """A width above the group count is legal: the engine replicates the heads.

        Refusing it would reject a shape that serves, which this check exists
        never to do.
        """
        path = self._model_dir(tmp_path, num_attention_heads=32, num_key_value_heads=4)

        _server.validate_width_against_model_config(8, model_name_or_path=path)

    def test_a_multimodal_config_is_checked_against_its_text_stack(self, tmp_path: Path) -> None:
        """Top-level head counts can describe the vision tower, which is not sharded."""
        path = self._model_dir(
            tmp_path,
            num_attention_heads=12,
            text_config={"num_attention_heads": 32, "num_key_value_heads": 8},
        )

        _server.validate_width_against_model_config(8, model_name_or_path=path)

    def test_width_one_is_never_checked(self, tmp_path: Path) -> None:
        """Single-device serving shards nothing, so no division has to hold."""
        path = self._model_dir(tmp_path, num_attention_heads=7, num_key_value_heads=7)

        _server.validate_width_against_model_config(1, model_name_or_path=path)

    def test_an_unreadable_config_checks_nothing(self, tmp_path: Path) -> None:
        """Best effort: this may only turn a later crash into an earlier error.

        It must never reject a shape that would have worked, so a model whose
        config cannot be read without a download is left to the engine.
        """
        _server.validate_width_against_model_config(4, model_name_or_path=str(tmp_path / "absent"))
        _server.validate_width_against_model_config(4, model_name_or_path="org/not-in-any-cache")

    def test_a_malformed_config_checks_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{not json", encoding="utf-8")

        _server.validate_width_against_model_config(4, model_name_or_path=str(tmp_path))

    def test_missing_or_nonsense_head_counts_are_skipped(self, tmp_path: Path) -> None:
        path = self._model_dir(tmp_path, num_attention_heads=0, num_key_value_heads="many")

        _server.validate_width_against_model_config(4, model_name_or_path=path)


class TestEmbeddingEngineLiveness:
    """The embedding adapter must notice its engine dying, like generation does."""

    @staticmethod
    def _adapter() -> object:
        from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter

        return SGLangEmbeddingAdapter(model_name_or_path="org/embed", dense_dim=768)

    def test_an_unloaded_adapter_reports_not_loaded(self) -> None:
        adapter = self._adapter()

        with pytest.raises(RuntimeError, match="not loaded"):
            adapter._check_loaded()

    def test_a_live_engine_passes(self) -> None:
        adapter = self._adapter()
        adapter._server_url = "http://localhost:30005"
        adapter._process = MagicMock(poll=MagicMock(return_value=None))

        adapter._check_loaded()

    def test_a_dead_engine_is_a_terminal_error_not_a_timeout(self) -> None:
        adapter = self._adapter()
        adapter._server_url = "http://localhost:30005"
        adapter._process = MagicMock(poll=MagicMock(return_value=-9))

        with pytest.raises(RuntimeError, match="exited with code -9"):
            adapter._check_loaded()

    def test_the_removed_pooling_option_is_no_longer_accepted(self) -> None:
        """Neither pinned engine declares a pooling flag, so the option was dead.

        It is better refused by the loader's unknown-option check than emitted
        and rejected by the engine at startup.
        """
        from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter
        from sie_server.core.loader import reject_unknown_loadtime_options

        with pytest.raises(ValueError, match="pooling_method"):
            reject_unknown_loadtime_options(SGLangEmbeddingAdapter, {"pooling_method": "mean"}, model_name="demo")


class TestEmbeddingGroupLaunch:
    """An embedding profile that declares a width, through the real ``load``."""

    @staticmethod
    def _reserving(port: int) -> object:
        def reserve(start_port: int = _server.BASE_PORT) -> int:
            _server._RESERVED_PORTS.add(port)
            return port

        return reserve

    @patch("sie_server.adapters.sglang._server.os.getpgid", return_value=12345)
    @patch("sie_server.adapters.sglang._server.os.killpg")
    @patch("sie_server.adapters.sglang._server.subprocess.Popen")
    @patch("sie_server.adapters.sglang._server.requests.get")
    def test_a_width_launches_the_whole_group_and_hands_every_port_back(
        self,
        mock_requests_get: MagicMock,
        mock_popen: MagicMock,
        mock_killpg: MagicMock,
        mock_getpgid: MagicMock,
    ) -> None:
        mock_popen.return_value = MagicMock(
            pid=12345, poll=MagicMock(return_value=None), wait=MagicMock(return_value=None)
        )
        mock_requests_get.return_value = MagicMock(status_code=200)
        ports = iter([30011, 30411])
        adapter = SGLangEmbeddingAdapter(
            model_name_or_path="Qwen/Qwen3-Embedding-8B",
            tensor_parallel_size=2,
            watchdog_timeout_s=90.0,
        )

        try:
            with patch(
                "sie_server.adapters.sglang._server.find_free_port",
                side_effect=lambda start_port=_server.BASE_PORT: self._reserving(next(ports))(start_port),
            ):
                adapter.load("cuda:2")

            cmd = mock_popen.call_args[0][0]
            env = mock_popen.call_args.kwargs["env"]
            assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
            assert cmd[cmd.index("--nccl-port") + 1] == "30411"
            assert cmd[cmd.index("--watchdog-timeout") + 1] == "90.0"
            assert "--disable-piecewise-cuda-graph" in cmd
            assert env["CUDA_VISIBLE_DEVICES"] == "2,3"

            adapter.unload()

            assert {30011, 30411}.isdisjoint(_server._RESERVED_PORTS)
        finally:
            _server.release_port(30011)
            _server.release_port(30411)

    def test_a_declared_collective_port_at_width_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nccl_port applies only above"):
            SGLangEmbeddingAdapter(model_name_or_path="Qwen/Qwen3-Embedding-8B", nccl_port=30499)
