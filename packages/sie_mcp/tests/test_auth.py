from typing import Any
from urllib.parse import urlsplit

import pytest
from sie_mcp.auth import ConnectorSecretAuthMiddleware, authenticate, base_url, bearer_token
from sie_mcp.config import DEFAULT_MAX_DOCUMENT_BYTES, DEFAULT_MAX_IMAGE_BYTES, MCPConfig
from starlette.datastructures import Headers


def _cfg(**overrides: Any) -> MCPConfig:
    base: dict[str, Any] = {
        "sie_base_url": "http://localhost:8080",
        "sie_api_key": None,
        "connector_secrets": {},
        "allow_anonymous": False,
        "allowed_hosts": [],
        "max_document_bytes": DEFAULT_MAX_DOCUMENT_BYTES,
        "max_image_bytes": DEFAULT_MAX_IMAGE_BYTES,
        "docling_model": "docling",
        "generate_model": "Qwen/Qwen3.5-4B",
        "extract_model": "urchade/gliner_multi-v2.1",
        "pii_model": "urchade/gliner_multi_pii-v1",
        "max_output_tokens": 4096,
        "vlocr_model": "opendatalab/MinerU2.5-Pro-2604-1.2B",
        "encode_model": "BAAI/bge-m3",
        "rerank_model": "BAAI/bge-reranker-v2-m3",
        "caption_model": "microsoft/Florence-2-base-ft",
        "embed_model": "openai/clip-vit-base-patch32",
        "image_labels": [],
        "image_top_k": 5,
        "gpu": None,
        "docs_gpu": None,
        "extract_gpu": None,
        "generate_gpu": None,
        "image_gpu": None,
        "qa_gpu": None,
        "timeout_s": 300.0,
        "qa_top_k": 5,
        "qa_rerank_candidates": 20,
        "qa_chunk_chars": 1200,
        "qa_chunk_overlap_chars": 200,
        "qa_max_tokens": 512,
        "qa_max_document_chars": 2_000_000,
        "qa_max_questions": 50,
        "qa_max_chunks": 2000,
        "oauth_enabled": True,
        "public_base_url": None,
        "oauth_redirect_uris": ("https://claude.ai/api/mcp/auth_callback",),
    }
    base.update(overrides)
    return MCPConfig(**base)


async def _run_middleware(cfg: MCPConfig, *, path: str, headers: list[tuple[bytes, bytes]]):
    """Drive the ASGI auth gate over a stub app, returning (reached_app, start_message)."""
    reached: dict[str, Any] = {"app": False, "user_id": None}

    async def _app(scope: Any, _receive: Any, send: Any) -> None:
        reached["app"] = True
        reached["user_id"] = (scope.get("state") or {}).get("user_id")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    sent: list[dict[str, Any]] = []

    async def _send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {"type": "http", "scheme": "https", "path": path, "headers": headers}
    await ConnectorSecretAuthMiddleware(_app, cfg)(scope, _receive, _send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return reached, start


def test_bearer_token_parsing() -> None:
    assert bearer_token("Bearer abc") == "abc"
    assert bearer_token("bearer abc") == "abc"
    assert bearer_token("abc") == "abc"
    assert bearer_token(None) is None
    assert bearer_token("") is None
    assert bearer_token("Bearer ") is None


def test_valid_secret_maps_to_user() -> None:
    cfg = _cfg(connector_secrets={"s3cret": "user-1"})
    assert authenticate(cfg, "s3cret") == "user-1"


def test_unknown_token_rejected() -> None:
    cfg = _cfg(connector_secrets={"s3cret": "user-1"})
    assert authenticate(cfg, "nope") is None


def test_missing_token_rejected_when_closed() -> None:
    cfg = _cfg(connector_secrets={"s3cret": "user-1"})
    assert authenticate(cfg, None) is None


def test_anonymous_allowed_when_enabled() -> None:
    cfg = _cfg(connector_secrets={}, allow_anonymous=True)
    assert authenticate(cfg, None) == "anonymous"


def test_closed_by_default_without_secrets() -> None:
    cfg = _cfg(connector_secrets={}, allow_anonymous=False)
    assert authenticate(cfg, None) is None


def test_invalid_token_rejected_even_when_anonymous_allowed() -> None:
    cfg = _cfg(connector_secrets={"s3cret": "user-1"}, allow_anonymous=True)
    assert authenticate(cfg, "wrong") is None


def test_base_url_prefers_pinned_public_url() -> None:
    cfg = _cfg(public_base_url="https://mcp.example.com")
    headers = Headers({"host": "internal:8088", "x-forwarded-proto": "http"})
    assert base_url(cfg, scheme="http", headers=headers) == "https://mcp.example.com"


def test_base_url_derives_from_request_scheme_and_allowed_host() -> None:
    cfg = _cfg(public_base_url=None, allowed_hosts=["mcp.example.com"])
    headers = Headers({"host": "mcp.example.com"})
    assert base_url(cfg, scheme="https", headers=headers) == "https://mcp.example.com"


def test_base_url_ignores_forwarded_headers() -> None:
    cfg = _cfg(public_base_url=None, allowed_hosts=["mcp.example.com"])
    headers = Headers(
        {"host": "mcp.example.com", "x-forwarded-proto": "http", "x-forwarded-host": "evil.attacker.example"}
    )
    assert base_url(cfg, scheme="https", headers=headers) == "https://mcp.example.com"


@pytest.mark.parametrize("host", ["evil.attacker.example", "mcp.example.com.evil.example", "", "notlocalhost:8088"])
def test_base_url_refuses_untrusted_host(host: str) -> None:
    cfg = _cfg(public_base_url=None, allowed_hosts=["mcp.example.com", "edge.example.com:*"])
    assert base_url(cfg, scheme="https", headers=Headers({"host": host})) is None


@pytest.mark.parametrize(
    "host",
    [
        # Extra authority after the port.
        "edge.example.com:8443@evil.attacker.example",
        "edge.example.com:8443/evil",
        # Not ports.
        "edge.example.com:99999",
        "edge.example.com:0",
        "edge.example.com:abc",
        "edge.example.com:",
    ],
)
def test_base_url_refuses_wildcard_host_without_a_valid_port(host: str) -> None:
    cfg = _cfg(public_base_url=None, allowed_hosts=["edge.example.com:*"])
    assert base_url(cfg, scheme="https", headers=Headers({"host": host})) is None


@pytest.mark.parametrize(
    "host",
    [
        # Loopback is trusted without configuration, so a Host that merely starts
        # with a loopback name must not inherit that trust.
        "localhost:8088@evil.attacker.example",
        "127.0.0.1:8088@evil.attacker.example",
        "[::1]@evil.attacker.example",
        "[::1]:8088@evil.attacker.example",
        "[::1].evil.attacker.example",
        "localhost:8088/evil",
        "localhost:99999",
    ],
)
def test_base_url_refuses_loopback_lookalike_hosts(host: str) -> None:
    cfg = _cfg(public_base_url=None, allowed_hosts=[])
    assert base_url(cfg, scheme="https", headers=Headers({"host": host})) is None


@pytest.mark.parametrize("literal", ["::::", "1:2:3", "12345::1", "::ffff:192.0.2.999"])
@pytest.mark.parametrize("allowed_port", ["443", "*"])
def test_base_url_refuses_malformed_ipv6_even_when_allowed(literal: str, allowed_port: str) -> None:
    cfg = _cfg(public_base_url=None, allowed_hosts=[f"[{literal}]:{allowed_port}"])
    assert base_url(cfg, scheme="https", headers=Headers({"host": f"[{literal}]:443"})) is None


@pytest.mark.parametrize("literal", ["2001:db8::1", "::ffff:192.0.2.1"])
@pytest.mark.parametrize("allowed_port", ["443", "*"])
def test_base_url_trusts_valid_allowed_ipv6(literal: str, allowed_port: str) -> None:
    cfg = _cfg(public_base_url=None, allowed_hosts=[f"[{literal}]:{allowed_port}"])
    host = f"[{literal}]:443"
    assert base_url(cfg, scheme="https", headers=Headers({"host": host})) == f"https://{host}"


@pytest.mark.parametrize(
    "host",
    [
        "edge.example.com:8443",
        "edge.example.com:8443@evil.attacker.example",
        "edge.example.com:8443/evil",
        "evil.attacker.example",
    ],
)
def test_advertised_origin_never_resolves_to_an_unconfigured_host(host: str) -> None:
    """Whatever is advertised must parse back to a host the operator configured."""
    cfg = _cfg(public_base_url=None, allowed_hosts=["edge.example.com:*"])
    origin = base_url(cfg, scheme="https", headers=Headers({"host": host}))
    if origin is None:
        return
    metadata = f"{origin}/.well-known/oauth-protected-resource"
    assert urlsplit(metadata).hostname == "edge.example.com"
    assert urlsplit(metadata).path == "/.well-known/oauth-protected-resource"


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost:8088", "http://localhost:8088"),
        ("127.0.0.1:8088", "http://127.0.0.1:8088"),
        ("[::1]:8088", "http://[::1]:8088"),
        ("LOCALHOST:8088", "http://LOCALHOST:8088"),
        ("edge.example.com:8443", "http://edge.example.com:8443"),
        ("Edge.Example.com:8443", "http://Edge.Example.com:8443"),
        ("mcp.example.com", "http://mcp.example.com"),
    ],
)
def test_base_url_trusts_loopback_and_wildcard_port_hosts(host: str, expected: str) -> None:
    cfg = _cfg(public_base_url=None, allowed_hosts=["EDGE.example.com:*", "MCP.Example.com"])
    assert base_url(cfg, scheme="http", headers=Headers({"host": host})) == expected


async def test_unauthorized_returns_www_authenticate_challenge() -> None:
    cfg = _cfg(connector_secrets={"s3cret": "user-1"}, allowed_hosts=["mcp.example.com"])
    reached, start = await _run_middleware(cfg, path="/mcp", headers=[(b"host", b"mcp.example.com")])
    assert reached["app"] is False
    assert start["status"] == 401
    challenge = Headers(raw=start["headers"])["www-authenticate"]
    assert 'resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"' in challenge


async def test_unauthorized_challenge_ignores_forwarded_host() -> None:
    cfg = _cfg(connector_secrets={"s3cret": "user-1"}, allowed_hosts=["mcp.example.com"])
    headers = [(b"host", b"mcp.example.com"), (b"x-forwarded-host", b"evil.attacker.example")]
    _reached, start = await _run_middleware(cfg, path="/mcp", headers=headers)
    challenge = Headers(raw=start["headers"])["www-authenticate"]
    assert 'resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"' in challenge


async def test_unauthorized_omits_challenge_for_untrusted_host() -> None:
    cfg = _cfg(connector_secrets={"s3cret": "user-1"})
    _reached, start = await _run_middleware(cfg, path="/mcp", headers=[(b"host", b"evil.attacker.example")])
    assert start["status"] == 401
    assert "www-authenticate" not in Headers(raw=start["headers"])


async def test_unauthorized_omits_challenge_when_oauth_disabled() -> None:
    cfg = _cfg(connector_secrets={"s3cret": "user-1"}, oauth_enabled=False)
    _reached, start = await _run_middleware(cfg, path="/mcp", headers=[(b"host", b"mcp.example.com")])
    assert start["status"] == 401
    assert "www-authenticate" not in Headers(raw=start["headers"])


@pytest.mark.parametrize(
    "path",
    [
        "/healthz",
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-authorization-server",
        "/register",
        "/authorize",
        "/token",
    ],
)
async def test_oauth_bootstrap_paths_are_exempt(path: str) -> None:
    cfg = _cfg(connector_secrets={"s3cret": "user-1"})
    reached, start = await _run_middleware(cfg, path=path, headers=[(b"host", b"mcp.example.com")])
    assert reached["app"] is True
    assert start["status"] == 200


async def test_valid_secret_reaches_app_with_identity() -> None:
    cfg = _cfg(connector_secrets={"s3cret": "user-1"})
    reached, start = await _run_middleware(
        cfg, path="/mcp", headers=[(b"host", b"mcp.example.com"), (b"authorization", b"Bearer s3cret")]
    )
    assert start["status"] == 200
    assert reached["user_id"] == "user-1"
