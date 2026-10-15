"""Token authentication for sie-server: resolution, the persisted token, and the middleware."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
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
    read_stored_token,
    resolve_auth_mode,
    server_tokens,
    token_path,
)
from starlette.testclient import WebSocketDenialResponse

TOKEN = "s3cret-token-value"  # noqa: S105 - test credential
OTHER = "another-token-value"


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    saved = dict(os.environ)
    for name in ("SIE_AUTH_MODE", "SIE_AUTH_TOKEN", "SIE_AUTH_TOKENS", "SIE_DATA_DIR", "XDG_DATA_HOME"):
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
            ({"Authorization": f"Bearer {TOKEN}", "x-api-key": OTHER}, "different tokens"),
            ({"Authorization": f"Bearer {TOKEN}", "x-api-key": "wrong"}, "different tokens"),
        ],
    )
    def test_refused_credentials(self, client: TestClient, headers: dict[str, str], message: str) -> None:
        response = client.get("/v1/thing", headers=headers)

        assert response.status_code == 401
        assert message in response.json()["detail"]["message"]

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
