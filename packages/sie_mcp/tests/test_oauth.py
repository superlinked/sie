import base64
import hashlib
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from sie_mcp.config import DEFAULT_MAX_DOCUMENT_BYTES, DEFAULT_MAX_IMAGE_BYTES, MCPConfig
from sie_mcp.oauth import (
    AuthCodeStore,
    OAuthError,
    authorization_server_metadata,
    build_oauth_routes,
    exchange_authorization_code,
    protected_resource_metadata,
    verify_pkce,
)
from starlette.applications import Starlette
from starlette.testclient import TestClient

_REDIRECT = "https://claude.ai/api/mcp/auth_callback"


def _cfg(**overrides: Any) -> MCPConfig:
    base: dict[str, Any] = {
        "sie_base_url": "http://localhost:8080",
        "sie_api_key": None,
        "connector_secrets": {"s3cret": "user-1"},
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


def _s256(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def test_verify_pkce_s256_roundtrip() -> None:
    verifier = "verifier-abc123"
    assert verify_pkce(verifier, _s256(verifier), "S256") is True
    assert verify_pkce("wrong", _s256(verifier), "S256") is False


def test_verify_pkce_rejects_non_s256_methods() -> None:
    # `plain` is intentionally unsupported — only S256 is accepted.
    assert verify_pkce("same", "same", "plain") is False
    assert verify_pkce("same", "same", "weird") is False
    assert verify_pkce("", "", "S256") is False


def test_verify_pkce_rejects_non_ascii_verifier() -> None:
    assert verify_pkce("vérifier", "x", "S256") is False


def test_auth_code_store_is_single_use() -> None:
    store = AuthCodeStore()
    code = store.issue(
        secret="s3cret",  # noqa: S106
        code_challenge=_s256("v"),
        code_challenge_method="S256",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        now=1000.0,
    )
    assert store.consume(code, now=1001.0) is not None
    assert store.consume(code, now=1001.0) is None  # already consumed


def test_auth_code_store_expires() -> None:
    store = AuthCodeStore(ttl_s=60.0)
    code = store.issue(
        secret="s3cret",  # noqa: S106
        code_challenge=_s256("v"),
        code_challenge_method="S256",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        now=1000.0,
    )
    assert store.consume(code, now=1100.0) is None  # past ttl


def test_exchange_returns_connector_secret_as_access_token() -> None:
    store = AuthCodeStore()
    redirect = "https://claude.ai/api/mcp/auth_callback"
    code = store.issue(
        secret="s3cret",  # noqa: S106
        code_challenge=_s256("verifier-1"),
        code_challenge_method="S256",
        redirect_uri=redirect,
        now=1000.0,
    )
    token = exchange_authorization_code(store, code=code, code_verifier="verifier-1", redirect_uri=redirect, now=1001.0)
    assert token == "s3cret"  # noqa: S105


def test_exchange_rejects_bad_verifier() -> None:
    store = AuthCodeStore()
    redirect = "https://claude.ai/api/mcp/auth_callback"
    code = store.issue(
        secret="s3cret",  # noqa: S106
        code_challenge=_s256("verifier-1"),
        code_challenge_method="S256",
        redirect_uri=redirect,
        now=1000.0,
    )
    with pytest.raises(OAuthError) as exc:
        exchange_authorization_code(store, code=code, code_verifier="nope", redirect_uri=redirect, now=1001.0)
    assert exc.value.error == "invalid_grant"


def test_exchange_rejects_redirect_mismatch() -> None:
    store = AuthCodeStore()
    code = store.issue(
        secret="s3cret",  # noqa: S106
        code_challenge=_s256("v"),
        code_challenge_method="S256",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        now=1000.0,
    )
    with pytest.raises(OAuthError):
        exchange_authorization_code(
            store, code=code, code_verifier="v", redirect_uri="https://evil.example/cb", now=1001.0
        )


def test_exchange_binds_code_to_client_id() -> None:
    store = AuthCodeStore()
    redirect = "https://claude.ai/api/mcp/auth_callback"
    code = store.issue(
        secret="s3cret",  # noqa: S106
        code_challenge=_s256("v"),
        code_challenge_method="S256",
        redirect_uri=redirect,
        client_id="sl-owner",
        now=1000.0,
    )
    # A different public client cannot redeem a code issued to "sl-owner".
    with pytest.raises(OAuthError) as exc:
        exchange_authorization_code(
            store, code=code, code_verifier="v", redirect_uri=redirect, client_id="sl-thief", now=1001.0
        )
    assert exc.value.error == "invalid_grant"


def test_exchange_accepts_matching_client_id() -> None:
    store = AuthCodeStore()
    redirect = "https://claude.ai/api/mcp/auth_callback"
    code = store.issue(
        secret="s3cret",  # noqa: S106
        code_challenge=_s256("v"),
        code_challenge_method="S256",
        redirect_uri=redirect,
        client_id="sl-owner",
        now=1000.0,
    )
    token = exchange_authorization_code(
        store, code=code, code_verifier="v", redirect_uri=redirect, client_id="sl-owner", now=1001.0
    )
    assert token == "s3cret"  # noqa: S105


def test_exchange_rejects_unknown_code() -> None:
    store = AuthCodeStore()
    with pytest.raises(OAuthError) as exc:
        exchange_authorization_code(store, code="missing", code_verifier="v", redirect_uri="", now=1.0)
    assert exc.value.error == "invalid_grant"


def test_exchange_requires_code_and_verifier() -> None:
    store = AuthCodeStore()
    with pytest.raises(OAuthError) as exc:
        exchange_authorization_code(store, code="", code_verifier="", redirect_uri="", now=1.0)
    assert exc.value.error == "invalid_request"


def test_metadata_documents_use_origin() -> None:
    origin = "https://mcp.example.com"
    resource = protected_resource_metadata(origin)
    assert resource["resource"] == "https://mcp.example.com/mcp"
    assert resource["authorization_servers"] == [origin]

    server = authorization_server_metadata(origin)
    assert server["issuer"] == origin
    assert server["authorization_endpoint"] == "https://mcp.example.com/authorize"
    assert server["token_endpoint"] == "https://mcp.example.com/token"  # noqa: S105
    assert server["registration_endpoint"] == "https://mcp.example.com/register"
    assert server["code_challenge_methods_supported"] == ["S256"]


def _client(cfg: MCPConfig, *, base_url: str = "https://mcp.example.com") -> TestClient:
    app = Starlette(routes=build_oauth_routes(cfg))
    return TestClient(app, base_url=base_url)


@pytest.mark.parametrize("path", ["/.well-known/oauth-authorization-server", "/.well-known/oauth-protected-resource"])
def test_metadata_origin_ignores_forwarded_headers(path: str) -> None:
    body = (
        _client(_cfg(allowed_hosts=["mcp.example.com"]))
        .get(path, headers={"X-Forwarded-Host": "evil.attacker.example", "X-Forwarded-Proto": "http"})
        .json()
    )
    assert "evil.attacker.example" not in str(body)
    assert "http://" not in str(body)
    if "issuer" in body:
        assert body["issuer"] == "https://mcp.example.com"
        assert body["token_endpoint"] == "https://mcp.example.com/token"  # noqa: S105
    else:
        assert body["authorization_servers"] == ["https://mcp.example.com"]


@pytest.mark.parametrize("path", ["/.well-known/oauth-authorization-server", "/.well-known/oauth-protected-resource"])
def test_metadata_refused_for_untrusted_host_when_unpinned(path: str) -> None:
    resp = _client(_cfg()).get(path)
    assert resp.status_code == 503
    assert "mcp.example.com" not in resp.text


def test_metadata_served_for_loopback_host_when_unpinned() -> None:
    body = _client(_cfg(), base_url="http://localhost:8088").get("/.well-known/oauth-authorization-server").json()
    assert body["issuer"] == "http://localhost:8088"


def test_metadata_uses_pinned_public_url_for_any_host() -> None:
    client = _client(_cfg(public_base_url="https://mcp.example.com"), base_url="https://evil.attacker.example")
    assert client.get("/.well-known/oauth-authorization-server").json()["issuer"] == "https://mcp.example.com"


def _authorize_params(verifier: str) -> dict[str, str]:
    return {
        "response_type": "code",
        "client_id": "sl-test",
        "redirect_uri": _REDIRECT,
        "state": "st-1",
        "code_challenge": _s256(verifier),
        "code_challenge_method": "S256",
        "scope": "mcp",
    }


def _obtain_code(client: TestClient, verifier: str) -> str:
    params = _authorize_params(verifier)
    redirect = client.post("/authorize", data={**params, "connector_secret": "s3cret"}, follow_redirects=False)
    assert redirect.status_code == 302
    return parse_qs(urlparse(redirect.headers["location"]).query)["code"][0]


def test_metadata_endpoints_use_request_origin() -> None:
    client = _client(_cfg(allowed_hosts=["mcp.example.com"]))
    resource = client.get("/.well-known/oauth-protected-resource").json()
    assert resource["resource"] == "https://mcp.example.com/mcp"
    server = client.get("/.well-known/oauth-authorization-server").json()
    assert server["issuer"] == "https://mcp.example.com"


def test_dynamic_client_registration_returns_client_id() -> None:
    client = _client(_cfg())
    resp = client.post("/register", json={"redirect_uris": [_REDIRECT]})
    assert resp.status_code == 201
    body = resp.json()
    assert body["client_id"].startswith("sl-")
    assert body["token_endpoint_auth_method"] == "none"  # noqa: S105


def test_authorize_token_roundtrip_returns_connector_secret() -> None:
    client = _client(_cfg())
    verifier = "verifier-roundtrip"
    params = _authorize_params(verifier)

    form_page = client.get("/authorize", params=params)
    assert form_page.status_code == 200
    assert 'name="connector_secret"' in form_page.text

    redirect = client.post("/authorize", data={**params, "connector_secret": "s3cret"}, follow_redirects=False)
    assert redirect.status_code == 302
    location = urlparse(redirect.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == _REDIRECT
    query = parse_qs(location.query)
    assert query["state"] == ["st-1"]
    code = query["code"][0]

    token = client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": _REDIRECT,
            "client_id": "sl-test",
        },
    )
    assert token.status_code == 200
    assert token.headers["cache-control"] == "no-store"
    body = token.json()
    assert body == {"access_token": "s3cret", "token_type": "Bearer", "scope": "mcp"}


def test_authorize_rejects_invalid_connector_secret() -> None:
    client = _client(_cfg())
    params = _authorize_params("verifier-bad")
    resp = client.post("/authorize", data={**params, "connector_secret": "wrong"}, follow_redirects=False)
    assert resp.status_code == 401
    assert "Invalid connector secret" in resp.text


def test_authorize_rejects_unlisted_redirect_uri() -> None:
    client = _client(_cfg())
    params = {**_authorize_params("v"), "redirect_uri": "https://evil.example/cb"}
    resp = client.get("/authorize", params=params)
    assert resp.status_code == 400


def test_authorize_rejects_plain_pkce_method() -> None:
    client = _client(_cfg())
    params = {**_authorize_params("v"), "code_challenge_method": "plain"}
    assert client.get("/authorize", params=params).status_code == 400


def test_authorize_post_requires_code_challenge() -> None:
    # A direct POST cannot skip the PKCE invariant the GET handler enforces.
    client = _client(_cfg())
    params = _authorize_params("v")
    params.pop("code_challenge")
    resp = client.post("/authorize", data={**params, "connector_secret": "s3cret"}, follow_redirects=False)
    assert resp.status_code == 400


def test_token_rejects_bad_pkce_over_http() -> None:
    client = _client(_cfg())
    code = _obtain_code(client, "verifier-good")
    resp = client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": "verifier-wrong",
            "redirect_uri": _REDIRECT,
            "client_id": "sl-test",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_grant"


def test_authorization_code_is_single_use_over_http() -> None:
    client = _client(_cfg())
    verifier = "verifier-reuse"
    body = {
        "grant_type": "authorization_code",
        "code": _obtain_code(client, verifier),
        "code_verifier": verifier,
        "redirect_uri": _REDIRECT,
        "client_id": "sl-test",
    }
    assert client.post("/token", data=body).status_code == 200
    replay = client.post("/token", data=body)
    assert replay.status_code == 400
    assert replay.json()["error"] == "invalid_grant"


def test_token_rejects_client_id_mismatch_over_http() -> None:
    # A code issued to the authorize-time client_id cannot be redeemed by another.
    client = _client(_cfg())
    verifier = "verifier-bind"
    resp = client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": _obtain_code(client, verifier),
            "code_verifier": verifier,
            "redirect_uri": _REDIRECT,
            "client_id": "sl-evil",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_grant"


def test_protected_resource_metadata_path_suffixed() -> None:
    # RFC 9728 path-suffixed variant resolves the same metadata.
    client = _client(_cfg(allowed_hosts=["mcp.example.com"]))
    resp = client.get("/.well-known/oauth-protected-resource/mcp")
    assert resp.status_code == 200
    assert resp.json()["resource"] == "https://mcp.example.com/mcp"


def test_token_rejects_unsupported_grant_type() -> None:
    client = _client(_cfg())
    resp = client.post("/token", data={"grant_type": "client_credentials"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "unsupported_grant_type"
