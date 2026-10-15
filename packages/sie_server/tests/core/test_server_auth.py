"""Token authentication for sie-server: resolution, the persisted token, and the middleware."""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.core.data_dir import resolve_data_dir
from sie_server.core.server_auth import (
    AuthConfigError,
    ServerAuthMiddleware,
    TokenFileError,
    configured_tokens,
    is_loopback_host,
    load_or_create_token,
    loopback_host_allowlist,
    read_stored_token,
    request_credentials,
    request_host,
    resolve_auth_mode,
    server_tokens,
    token_path,
)
from sie_server.observability.telemetry import _get_or_create_worker_id
from starlette.testclient import WebSocketDenialResponse

TOKEN = "s3cret-token-value"  # noqa: S105 - test credential
OTHER = "another-token-value"


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    saved = dict(os.environ)
    for name in (
        "SIE_AUTH_MODE",
        "SIE_AUTH_TOKEN",
        "SIE_AUTH_TOKENS",
        "SIE_DATA_DIR",
        "SIE_SERVER_HOST",
        "XDG_DATA_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    os.environ.clear()
    os.environ.update(saved)


class TestResolveAuthMode:
    @pytest.mark.parametrize(
        ("requested", "has_tokens", "exposed", "expected"),
        [
            (None, False, False, "none"),
            (None, False, True, "none"),
            (None, True, False, "token"),
            (None, True, True, "token"),
            ("token", False, False, "token"),
            ("static", False, True, "token"),
            ("auto", False, False, "none"),
            ("auto", False, True, "token"),
            ("auto", True, False, "token"),
            ("none", False, False, "none"),
            ("none", False, True, "none"),
        ],
    )
    def test_resolution(self, requested: str | None, has_tokens: bool, exposed: bool, expected: str) -> None:
        assert resolve_auth_mode(requested, has_tokens=has_tokens, exposed=exposed) == expected

    def test_a_token_with_auth_explicitly_off_is_refused(self) -> None:
        with pytest.raises(AuthConfigError, match="SIE_AUTH_MODE=none"):
            resolve_auth_mode("none", has_tokens=True, exposed=False)

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(AuthConfigError, match="none, token, static or auto"):
            resolve_auth_mode("open", has_tokens=False, exposed=False)

    @pytest.mark.parametrize("requested", ["TOKEN", "Token", "None", "AUTO"])
    def test_modes_are_case_sensitive_as_on_the_gateway(self, requested: str) -> None:
        with pytest.raises(AuthConfigError, match="lowercase"):
            resolve_auth_mode(requested, has_tokens=False, exposed=False)

    @pytest.mark.parametrize("requested", [None, "token", "static", "auto"])
    def test_a_declared_but_blank_token_is_refused(self, requested: str | None) -> None:
        with pytest.raises(AuthConfigError, match="holds no token"):
            resolve_auth_mode(requested, has_tokens=False, exposed=True, declares_tokens=True)

    def test_a_blank_token_with_auth_explicitly_off_is_harmless(self) -> None:
        assert resolve_auth_mode("none", has_tokens=False, exposed=True, declares_tokens=True) == "none"


class TestConfiguredTokens:
    def test_tokens_variable_wins_over_token(self) -> None:
        env = {"SIE_AUTH_TOKENS": " a , b,,", "SIE_AUTH_TOKEN": "c"}
        assert configured_tokens(env) == ("a", "b")

    def test_single_token_variable_is_comma_separated_too(self) -> None:
        assert configured_tokens({"SIE_AUTH_TOKEN": "c,d"}) == ("c", "d")

    def test_blank_values_configure_nothing(self) -> None:
        assert configured_tokens({"SIE_AUTH_TOKENS": " , ", "SIE_AUTH_TOKEN": ""}) == ()


class TestIsLoopbackHost:
    @pytest.mark.parametrize("host", ["127.0.0.1", "127.8.9.10", "::1", "[::1]", "localhost", "LOCALHOST"])
    def test_loopback(self, host: str) -> None:
        assert is_loopback_host(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "10.0.0.1", "example.com", ""])  # noqa: S104
    def test_exposed(self, host: str) -> None:
        assert not is_loopback_host(host)


class TestDataDir:
    def test_explicit_directory_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        assert resolve_data_dir() == tmp_path / "data"

    def test_xdg_data_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        assert resolve_data_dir() == tmp_path / "sie"

    def test_relative_xdg_data_home_is_ignored(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", "relative")
        monkeypatch.setenv("HOME", str(tmp_path))
        assert resolve_data_dir() == tmp_path / ".local" / "share" / "sie"

    def test_default_is_per_user(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        assert resolve_data_dir() == tmp_path / ".local" / "share" / "sie"


class TestPersistedToken:
    def test_first_use_creates_a_private_token(self, tmp_path: Path) -> None:
        directory = tmp_path / "sie"
        stored = load_or_create_token(directory)

        assert stored.created
        assert stored.path == token_path(directory)
        assert len(stored.token) >= 43
        assert stat.S_IMODE(stored.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(directory.stat().st_mode) & 0o077 == 0
        assert stored.path.read_text() == f"{stored.token}\n"

    def test_later_use_reads_the_same_token(self, tmp_path: Path) -> None:
        first = load_or_create_token(tmp_path)
        second = load_or_create_token(tmp_path)

        assert not second.created
        assert second.token == first.token

    def test_a_token_others_can_read_is_refused(self, tmp_path: Path) -> None:
        stored = load_or_create_token(tmp_path)
        stored.path.chmod(0o644)

        with pytest.raises(TokenFileError, match="chmod 600"):
            read_stored_token(tmp_path)

    def test_concurrent_first_starts_agree_on_one_token(self, tmp_path: Path) -> None:
        for trial in range(10):
            directory = tmp_path / f"trial-{trial}"
            starts = [
                subprocess.Popen(  # noqa: S603
                    [sys.executable, "-c", _FIRST_START_SCRIPT, str(directory)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for _ in range(6)
            ]
            results = [start.communicate(timeout=60) for start in starts]
            assert [start.returncode for start in starts] == [0] * 6, results
            assert len({out.strip() for out, _ in results}) == 1
            assert [entry.name for entry in directory.iterdir()] == ["auth-token"]
            assert token_path(directory).stat().st_nlink == 1

    def test_an_empty_token_file_names_the_file_to_delete(self, tmp_path: Path) -> None:
        path = token_path(tmp_path)
        path.touch(mode=0o600)

        with pytest.raises(TokenFileError, match=f"Delete {re.escape(str(path))}"):
            load_or_create_token(tmp_path)

    def test_a_fifo_is_refused_without_blocking(self, tmp_path: Path) -> None:
        os.mkfifo(token_path(tmp_path), 0o600)
        outcome: list[BaseException] = []
        reader = threading.Thread(target=_capture, args=(lambda: read_stored_token(tmp_path), outcome), daemon=True)
        reader.start()
        reader.join(timeout=5)

        assert not reader.is_alive(), "reading a FIFO token path blocked"
        assert isinstance(outcome[0], TokenFileError)
        assert "must be a regular file" in str(outcome[0])

    def test_a_hard_linked_token_is_refused(self, tmp_path: Path) -> None:
        stored = load_or_create_token(tmp_path)
        os.link(stored.path, tmp_path / "copy")

        with pytest.raises(TokenFileError, match="hard links"):
            read_stored_token(tmp_path)

    @pytest.mark.parametrize("mode", [0o770, 0o707, 0o777])
    def test_a_directory_others_can_write_is_refused(self, tmp_path: Path, mode: int) -> None:
        directory = tmp_path / "shared"
        directory.mkdir()
        directory.chmod(mode)

        with pytest.raises(TokenFileError, match="chmod 700"):
            load_or_create_token(directory)
        assert not token_path(directory).exists()

    def test_a_directory_of_another_user_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        load_or_create_token(tmp_path)
        monkeypatch.setattr(os, "getuid", lambda: tmp_path.stat().st_uid + 1)

        with pytest.raises(TokenFileError, match="must be owned by the user"):
            read_stored_token(tmp_path)

    def test_the_worker_id_is_never_written_through_a_symbolic_link(self, tmp_path: Path) -> None:
        target = tmp_path / "elsewhere"
        (tmp_path / "worker-id").symlink_to(target)

        _get_or_create_worker_id(str(tmp_path))

        assert not target.exists()

    def test_a_symbolic_link_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "elsewhere"
        target.write_text(f"{TOKEN}\n")
        target.chmod(0o600)
        token_path(tmp_path).symlink_to(target)

        with pytest.raises(TokenFileError, match="symbolic link"):
            load_or_create_token(tmp_path)

    @pytest.mark.parametrize("content", ["", "\n", "two tokens\n", "a,b\n", "tokén\n"])
    def test_malformed_content_is_refused(self, tmp_path: Path, content: str) -> None:
        path = token_path(tmp_path)
        path.write_text(content, encoding="utf-8")
        path.chmod(0o600)

        with pytest.raises(TokenFileError, match="must hold one token"):
            read_stored_token(tmp_path)

    def test_server_tokens_prefer_the_configured_ones(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path))
        load_or_create_token(tmp_path)

        assert server_tokens({"SIE_AUTH_TOKEN": TOKEN}) == (TOKEN,)

    def test_server_tokens_fall_back_to_the_persisted_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path))
        stored = load_or_create_token(tmp_path)

        assert server_tokens(os.environ) == (stored.token,)

    def test_server_tokens_never_generate_one(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path))

        with pytest.raises(TokenFileError):
            server_tokens(os.environ)
        assert not token_path(tmp_path).exists()


def _protected_app(tokens: tuple[str, ...] = (TOKEN, OTHER)) -> FastAPI:
    app = FastAPI()

    @app.get("/healthz")
    def healthz() -> str:
        return "ok"

    @app.get("/v1/thing")
    def thing() -> dict[str, bool]:
        return {"ok": True}

    @app.websocket("/ws/status")
    async def status(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"ready": True})
        await websocket.close()

    app.add_middleware(ServerAuthMiddleware, tokens=tokens)
    return app


class TestMiddleware:
    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(_protected_app())

    def test_probes_need_no_token(self, client: TestClient) -> None:
        assert client.get("/healthz").status_code == 200

    def test_a_trailing_slash_is_not_a_probe(self, client: TestClient) -> None:
        assert client.get("/healthz/").status_code == 401

    def test_missing_credentials(self, client: TestClient) -> None:
        response = client.get("/v1/thing")

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json()["detail"]["code"] == "UNAUTHORIZED"

    @pytest.mark.parametrize(
        "headers",
        [
            {"Authorization": f"Bearer {TOKEN}"},
            {"Authorization": f"bearer {TOKEN}"},
            {"Authorization": f"  Bearer   {TOKEN}  "},
            {"Authorization": TOKEN},
            {"x-api-key": TOKEN},
            {"X-API-Key": OTHER},
            {"Authorization": f"Bearer {TOKEN}", "x-api-key": TOKEN},
            {"Authorization": f"Bearer {TOKEN}", "x-api-key": OTHER},
            {"Authorization": f"Bearer {TOKEN}", "x-api-key": "wrong"},
            {"Authorization": "", "x-api-key": TOKEN},
        ],
    )
    def test_accepted_credentials(self, client: TestClient, headers: dict[str, str]) -> None:
        assert client.get("/v1/thing", headers=headers).status_code == 200

    @pytest.mark.parametrize(
        ("headers", "message"),
        [
            ({"Authorization": "Bearer wrong"}, "Invalid token"),
            ({"x-api-key": "wrong"}, "Invalid token"),
            ({"Authorization": f"Bearer {TOKEN[:-1]}"}, "Invalid token"),
            ({"Authorization": f"Basic {TOKEN}"}, "Invalid token"),
            ({"Authorization": "Bearer"}, "Invalid token"),
            ({"Authorization": "", "x-api-key": ""}, "Missing credentials"),
            ({"Authorization": "Bearer wrong", "x-api-key": TOKEN}, "Invalid token"),
        ],
    )
    def test_refused_credentials(self, client: TestClient, headers: dict[str, str], message: str) -> None:
        response = client.get("/v1/thing", headers=headers)

        assert response.status_code == 401
        assert message in response.json()["detail"]["message"]

    @pytest.mark.parametrize(
        "headers",
        [
            [("Authorization", f"Bearer {TOKEN}"), ("Authorization", f"Bearer {OTHER}")],
            [("Authorization", f"Bearer {TOKEN}"), ("Authorization", "Bearer wrong"), ("x-api-key", TOKEN)],
            [("x-api-key", TOKEN), ("x-api-key", OTHER)],
        ],
    )
    def test_a_repeated_header_that_disagrees_is_refused(
        self, client: TestClient, headers: list[tuple[str, str]]
    ) -> None:
        # One header repeated with two tokens is not the two-header case, so it stays a refusal.
        response = client.get("/v1/thing", headers=headers)

        assert response.status_code == 401
        assert "Repeated credential headers carry different tokens" in response.json()["detail"]["message"]

    def test_websocket_without_a_token_is_denied(self, client: TestClient) -> None:
        with pytest.raises(WebSocketDenialResponse) as denied, client.websocket_connect("/ws/status"):
            pass
        assert denied.value.status_code == 401

    def test_websocket_with_a_token_connects(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/status", headers={"Authorization": f"Bearer {TOKEN}"}) as websocket:
            assert websocket.receive_json() == {"ready": True}

    def test_a_middleware_without_tokens_cannot_be_built(self) -> None:
        with pytest.raises(ValueError, match="at least one token"):
            TestClient(_protected_app(tokens=())).get("/healthz")


class TestRequestCredentials:
    def test_authorization_decides_when_both_headers_are_sent(self) -> None:
        headers = [(b"authorization", b"Bearer first"), (b"x-api-key", b"second")]
        assert request_credentials(headers) == {b"first"}

    def test_x_api_key_alone_decides(self) -> None:
        assert request_credentials([(b"x-api-key", b" second ")]) == {b"second"}

    def test_an_empty_authorization_leaves_x_api_key_to_decide(self) -> None:
        assert request_credentials([(b"authorization", b""), (b"x-api-key", b"second")]) == {b"second"}


class TestHostAllowlist:
    @pytest.mark.parametrize(
        ("host", "expected"),
        [
            ("localhost", "localhost"),
            ("LOCALHOST:8080", "localhost"),
            ("127.0.0.1:8080", "127.0.0.1"),
            ("[::1]:8080", "[::1]"),
            ("[::1]", "[::1]"),
            ("evil.example", "evil.example"),
            ("[::1", None),
            ("localhost:80x", None),
        ],
    )
    def test_request_host(self, host: str, expected: str | None) -> None:
        assert request_host([(b"host", host.encode())]) == expected

    def test_a_missing_host_matches_nothing(self) -> None:
        assert request_host([(b"accept", b"*/*")]) is None

    @pytest.mark.parametrize(
        ("bind", "extra"), [("127.0.0.1", None), ("localhost", None), ("::1", None), ("127.0.0.2", "127.0.0.2")]
    )
    def test_allowlist_holds_the_loopback_names_and_the_bind_address(self, bind: str, extra: str | None) -> None:
        allowed = loopback_host_allowlist(bind)
        assert {"localhost", "127.0.0.1", "[::1]"} <= allowed
        if extra:
            assert extra in allowed


def _loopback_client(bind_host: str, base_url: str = "http://localhost:8080", **config: object) -> TestClient:
    return TestClient(AppFactory.create_app(AppStateConfig(bind_host=bind_host, **config)), base_url=base_url)


class TestLoopbackServerApp:
    @pytest.mark.parametrize("base_url", ["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost"])
    def test_loopback_clients_are_answered(self, base_url: str) -> None:
        client = _loopback_client("127.0.0.1", base_url)

        assert client.get("/openapi.json").status_code == 200
        assert client.get("/healthz").status_code == 200

    def test_an_ipv6_loopback_host_is_answered(self) -> None:
        client = _loopback_client("::1")

        assert client.get("/openapi.json", headers={"Host": "[::1]:8080"}).status_code == 200

    @pytest.mark.parametrize("host", ["evil.example", "localhost.evil.example", "evil.example:8080", "0.0.0.0:8080"])
    def test_a_rebound_host_is_refused(self, host: str) -> None:
        response = _loopback_client("127.0.0.1").get("/openapi.json", headers={"Host": host})

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "INVALID_INPUT"

    def test_a_rebound_websocket_is_refused(self) -> None:
        client = _loopback_client("127.0.0.1")
        with (
            pytest.raises(WebSocketDenialResponse) as denied,
            client.websocket_connect("/ws/status", headers={"Host": "evil.example"}),
        ):
            pass
        assert denied.value.status_code == 400

    def test_the_bind_address_is_answered(self) -> None:
        client = _loopback_client("127.0.0.2", "http://127.0.0.2:8080")

        assert client.get("/openapi.json").status_code == 200

    @pytest.mark.parametrize("bind_host", ["0.0.0.0", "192.168.1.20", None])  # noqa: S104
    def test_an_exposed_or_unknown_bind_is_not_host_checked(self, bind_host: str | None) -> None:
        client = TestClient(AppFactory.create_app(AppStateConfig(bind_host=bind_host)))

        assert client.get("/openapi.json", headers={"Host": "evil.example"}).status_code == 200

    def test_token_auth_replaces_the_host_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_AUTH_TOKEN", TOKEN)
        client = _loopback_client("127.0.0.1", auth_mode="token")

        response = client.get("/openapi.json", headers={"Host": "evil.example", "x-api-key": TOKEN})
        assert response.status_code == 200


class TestServerApp:
    def test_token_mode_protects_the_real_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_AUTH_TOKEN", TOKEN)
        client = TestClient(AppFactory.create_app(AppStateConfig(auth_mode="token")))

        assert client.get("/healthz").status_code == 200
        assert client.get("/openapi.json").status_code == 401
        assert client.get("/openapi.json", headers={"x-api-key": TOKEN}).status_code == 200
        with pytest.raises(WebSocketDenialResponse), client.websocket_connect("/ws/status"):
            pass

    def test_default_mode_installs_no_auth(self) -> None:
        client = TestClient(AppFactory.create_app(AppStateConfig()))

        assert client.get("/openapi.json").status_code == 200

    def test_token_mode_without_any_token_refuses_to_build(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path))

        with pytest.raises(TokenFileError):
            AppFactory.create_app(AppStateConfig(auth_mode="token"))


class TestAppStateConfigRoundTrip:
    def test_the_resolved_mode_survives_the_factory_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_AUTH_TOKEN", TOKEN)
        AppStateConfig(auth_mode="token").save_to_env_vars()

        assert os.environ["SIE_AUTH_MODE"] == "token"
        assert AppStateConfig.from_env_vars().auth_mode == "token"

    def test_an_unresolved_auto_requires_a_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_AUTH_MODE", "auto")

        assert AppStateConfig.from_env_vars().auth_mode == "token"

    def test_the_bind_address_survives_the_factory_boundary(self) -> None:
        AppStateConfig(bind_host="127.0.0.1").save_to_env_vars()

        assert AppStateConfig.from_env_vars().bind_host == "127.0.0.1"

    def test_the_factory_refuses_a_blank_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_AUTH_TOKEN", " , ")

        with pytest.raises(AuthConfigError, match="holds no token"):
            AppStateConfig.from_env_vars()


_FIRST_START_SCRIPT = (
    "import sys; from pathlib import Path; "
    "from sie_server.core.server_auth import load_or_create_token; "
    "print(load_or_create_token(Path(sys.argv[1])).token)"
)


def _capture(call: Callable[[], object], outcome: list[BaseException]) -> None:
    try:
        call()
    except BaseException as exc:  # noqa: BLE001
        outcome.append(exc)
