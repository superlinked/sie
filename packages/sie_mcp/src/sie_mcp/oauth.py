"""OAuth 2.0 bridge for the claude.ai connector surface.

claude.ai custom connectors are OAuth-only — a user cannot paste a Bearer/connector
secret into the connector UI the way the Cowork plugin bakes one into its header.
So this module fronts the MCP edge with the minimal OAuth 2.0 + PKCE handshake the
platform requires (metadata discovery, Dynamic Client Registration, an authorize
page, and a token endpoint) and maps it onto the **existing** connector-secret shim:
the user types their connector secret on the authorize page, and the token endpoint
returns that secret as the access token. claude.ai then sends it as
``Authorization: Bearer <secret>`` and ``auth.authenticate`` validates it unchanged.
The MCP tools and the auth shim are untouched; the static connector secret can
later be replaced with issued keys.

Security notes:
- Connector secrets, authorization codes, and access tokens are NEVER logged.
- PKCE is required; redirect URIs are validated against a configured allowlist.
- Authorization codes are bound to the requesting ``client_id`` (when one is
  presented), so a leaked code cannot be redeemed by a different public client.
  Audience (``resource``) binding belongs with real issued keys.
- Authorization codes are single-use and short-lived, held in-process (the edge runs
  as a single uvicorn worker; a multi-worker deployment requires a shared store).
"""

import base64
import functools
import hashlib
import html
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, quote, urlencode

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from sie_mcp.auth import authenticate, base_url
from sie_mcp.config import MCPConfig

_AUTH_CODE_TTL_S = 120.0
# Only S256 is accepted: `plain` would carry the verifier in cleartext (defeating
# PKCE), and the advertised authorization-server metadata is S256-only.
_SUPPORTED_PKCE_METHODS = frozenset({"S256"})


def verify_pkce(code_verifier: str, code_challenge: str, method: str) -> bool:
    """Validate a PKCE ``code_verifier`` against the stored S256 challenge (RFC 7636)."""
    if not code_verifier or not code_challenge or method != "S256":
        return False
    try:
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    except UnicodeEncodeError:
        return False
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return secrets.compare_digest(expected, code_challenge)


class OAuthError(Exception):
    """An OAuth error to surface as the spec's ``{error, error_description}`` body."""

    def __init__(self, error: str, description: str, *, status: int = 400) -> None:
        super().__init__(description)
        self.error = error
        self.description = description
        self.status = status


@dataclass(frozen=True)
class _AuthCode:
    secret: str
    code_challenge: str
    code_challenge_method: str
    redirect_uri: str
    client_id: str
    expires_at: float


class AuthCodeStore:
    """In-process, single-use, TTL-bounded authorization-code store."""

    def __init__(self, ttl_s: float = _AUTH_CODE_TTL_S) -> None:
        self._ttl = ttl_s
        self._codes: dict[str, _AuthCode] = {}

    def issue(
        self,
        *,
        secret: str,
        code_challenge: str,
        code_challenge_method: str,
        redirect_uri: str,
        client_id: str = "",
        now: float | None = None,
    ) -> str:
        now = time.time() if now is None else now
        self._purge(now)
        code = secrets.token_urlsafe(32)
        self._codes[code] = _AuthCode(
            secret=secret,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method or "S256",
            redirect_uri=redirect_uri,
            client_id=client_id,
            expires_at=now + self._ttl,
        )
        return code

    def consume(self, code: str, *, now: float | None = None) -> _AuthCode | None:
        now = time.time() if now is None else now
        record = self._codes.pop(code, None)
        if record is None or record.expires_at < now:
            return None
        return record

    def _purge(self, now: float) -> None:
        expired = [code for code, record in self._codes.items() if record.expires_at < now]
        for code in expired:
            del self._codes[code]


def protected_resource_metadata(origin: str) -> dict[str, Any]:
    """RFC 9728 metadata pointing the MCP resource at this authorization server."""
    return {
        "resource": f"{origin}/mcp",
        "authorization_servers": [origin],
        "bearer_methods_supported": ["header"],
    }


def authorization_server_metadata(origin: str) -> dict[str, Any]:
    """RFC 8414 metadata describing the bridge's OAuth endpoints."""
    return {
        "issuer": origin,
        "authorization_endpoint": f"{origin}/authorize",
        "token_endpoint": f"{origin}/token",
        "registration_endpoint": f"{origin}/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": ["mcp"],
    }


def exchange_authorization_code(
    store: AuthCodeStore,
    *,
    code: str,
    code_verifier: str,
    redirect_uri: str,
    client_id: str = "",
    now: float | None = None,
) -> str:
    """Redeem an authorization code for the access token (the connector secret)."""
    if not code or not code_verifier:
        raise OAuthError("invalid_request", "code and code_verifier are required")
    record = store.consume(code, now=now)
    if record is None:
        raise OAuthError("invalid_grant", "authorization code is invalid or expired")
    # Bind the code to the public client that requested it: a code issued to one
    # client_id cannot be redeemed by another (RFC 6749 §4.1.3 for public clients).
    if record.client_id and client_id != record.client_id:
        raise OAuthError("invalid_grant", "authorization code was issued to a different client")
    if redirect_uri and redirect_uri != record.redirect_uri:
        raise OAuthError("invalid_grant", "redirect_uri does not match the authorization request")
    if not verify_pkce(code_verifier, record.code_challenge, record.code_challenge_method):
        raise OAuthError("invalid_grant", "PKCE verification failed")
    return record.secret


def _redirect_allowed(config: MCPConfig, redirect_uri: str) -> bool:
    return redirect_uri in config.oauth_redirect_uris


def _validate_authorize_params(config: MCPConfig, params: dict[str, str]) -> HTMLResponse | None:
    """Validate the OAuth authorize request; return a 400 page on failure, else ``None``.

    Enforced identically on the GET (form render) and POST (code issuance) paths so a
    direct POST cannot skip the invariants the GET handler checks.
    """
    if not _redirect_allowed(config, params.get("redirect_uri", "")):
        return HTMLResponse("<h1>Invalid redirect_uri</h1>", status_code=400)
    if params.get("response_type") != "code":
        return HTMLResponse("<h1>Unsupported response_type (expected 'code')</h1>", status_code=400)
    if not params.get("code_challenge"):
        return HTMLResponse("<h1>Missing PKCE code_challenge</h1>", status_code=400)
    if params.get("code_challenge_method", "S256") not in _SUPPORTED_PKCE_METHODS:
        return HTMLResponse("<h1>Unsupported code_challenge_method (expected S256)</h1>", status_code=400)
    return None


def _form_fields(params: dict[str, str]) -> str:
    keep = ("response_type", "client_id", "redirect_uri", "state", "code_challenge", "code_challenge_method", "scope")
    return "".join(f'<input type="hidden" name="{name}" value="{html.escape(params.get(name, ""))}">' for name in keep)


def _render_authorize_form(params: dict[str, str], *, error: str | None = None, status: int = 200) -> HTMLResponse:
    error_html = f'<p class="error">{html.escape(error)}</p>' if error else ""
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Connect Superlinked</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 28rem; margin: 4rem auto; padding: 0 1rem; }}
  label {{ display: block; margin: 1rem 0 0.25rem; font-weight: 600; }}
  input[type=password] {{ width: 100%; padding: 0.5rem; box-sizing: border-box; }}
  button {{ margin-top: 1.25rem; padding: 0.5rem 1rem; }}
  .error {{ color: #b00020; }}
  .hint {{ color: #555; font-size: 0.9rem; }}
</style></head><body>
<h1>Connect to Superlinked</h1>
<p class="hint">Enter the connector secret your cluster operator issued. It is sent
only to the Superlinked gateway and never stored in your chat.</p>
{error_html}
<form method="post" action="/authorize">
  {_form_fields(params)}
  <label for="connector_secret">Connector secret</label>
  <input id="connector_secret" name="connector_secret" type="password" autocomplete="off" required>
  <button type="submit">Authorize</button>
</form>
</body></html>"""
    return HTMLResponse(body, status_code=status)


def _parse_form(body: bytes) -> dict[str, str]:
    try:
        decoded = body.decode("utf-8")
    except UnicodeDecodeError:
        return {}
    return {key: values[0] for key, values in parse_qs(decoded).items() if values}


_ORIGIN_UNAVAILABLE = {
    "error": "server_error",
    "error_description": "OAuth origin is not configured; set SIE_MCP_PUBLIC_URL or SIE_MCP_ALLOWED_HOSTS.",
}


async def _protected_resource(request: Request, config: MCPConfig) -> JSONResponse:
    origin = base_url(config, scheme=request.url.scheme, headers=request.headers)
    if origin is None:
        return JSONResponse(_ORIGIN_UNAVAILABLE, status_code=503)
    return JSONResponse(protected_resource_metadata(origin))


async def _authorization_server(request: Request, config: MCPConfig) -> JSONResponse:
    origin = base_url(config, scheme=request.url.scheme, headers=request.headers)
    if origin is None:
        return JSONResponse(_ORIGIN_UNAVAILABLE, status_code=503)
    return JSONResponse(authorization_server_metadata(origin))


async def _register(request: Request) -> JSONResponse:
    """Dynamic Client Registration (RFC 7591). Public PKCE clients, not persisted."""
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        body = {}
    redirect_uris = body.get("redirect_uris") if isinstance(body, dict) else None
    return JSONResponse(
        {
            "client_id": f"sl-{secrets.token_urlsafe(16)}",
            "client_id_issued_at": int(time.time()),
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "redirect_uris": redirect_uris or [],
        },
        status_code=201,
    )


async def _authorize_get(request: Request, config: MCPConfig) -> Response:
    params = dict(request.query_params)
    invalid = _validate_authorize_params(config, params)
    if invalid is not None:
        return invalid
    return _render_authorize_form(params)


async def _authorize(request: Request, config: MCPConfig, store: AuthCodeStore) -> Response:
    if request.method == "POST":
        return await _authorize_post(request, config, store)
    return await _authorize_get(request, config)


async def _authorize_post(request: Request, config: MCPConfig, store: AuthCodeStore) -> Response:
    form = _parse_form(await request.body())
    invalid = _validate_authorize_params(config, form)
    if invalid is not None:
        return invalid

    secret = form.get("connector_secret") or None
    if secret is None or authenticate(config, secret) is None:
        return _render_authorize_form(form, error="Invalid connector secret.", status=401)

    redirect_uri = form.get("redirect_uri", "")
    code = store.issue(
        secret=secret,
        code_challenge=form.get("code_challenge", ""),
        code_challenge_method=form.get("code_challenge_method", "S256"),
        redirect_uri=redirect_uri,
        client_id=form.get("client_id", ""),
    )
    query = {"code": code}
    if form.get("state"):
        query["state"] = form["state"]
    separator = "&" if "?" in redirect_uri else "?"
    return RedirectResponse(f"{redirect_uri}{separator}{urlencode(query, quote_via=quote)}", status_code=302)


async def _token(request: Request, store: AuthCodeStore) -> JSONResponse:
    form = _parse_form(await request.body())
    if form.get("grant_type") != "authorization_code":
        return JSONResponse(
            {"error": "unsupported_grant_type", "error_description": "only authorization_code is supported"},
            status_code=400,
            headers={"Cache-Control": "no-store"},
        )
    try:
        access_token = exchange_authorization_code(
            store,
            code=form.get("code", ""),
            code_verifier=form.get("code_verifier", ""),
            redirect_uri=form.get("redirect_uri", ""),
            client_id=form.get("client_id", ""),
        )
    except OAuthError as exc:
        return JSONResponse(
            {"error": exc.error, "error_description": exc.description},
            status_code=exc.status,
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(
        {"access_token": access_token, "token_type": "Bearer", "scope": "mcp"},
        headers={"Cache-Control": "no-store"},
    )


def build_oauth_routes(config: MCPConfig) -> list[Route]:
    """Build the OAuth bridge routes, sharing one in-process authorization-code store."""
    store = AuthCodeStore()
    protected_resource = functools.partial(_protected_resource, config=config)
    return [
        Route("/.well-known/oauth-protected-resource", protected_resource, methods=["GET"]),
        # RFC 9728 also derives the metadata URL with the resource path appended; serve
        # it so a strict client that ignores the advertised URL still resolves the MCP
        # resource at /mcp.
        Route("/.well-known/oauth-protected-resource/mcp", protected_resource, methods=["GET"]),
        Route(
            "/.well-known/oauth-authorization-server",
            functools.partial(_authorization_server, config=config),
            methods=["GET"],
        ),
        Route("/register", _register, methods=["POST"]),
        Route("/authorize", functools.partial(_authorize, config=config, store=store), methods=["GET", "POST"]),
        Route("/token", functools.partial(_token, store=store), methods=["POST"]),
    ]
