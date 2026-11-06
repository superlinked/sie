"""Gateway pool reads and floor-carrier writes through the SIE Python SDK, installed by the keda extra."""

from __future__ import annotations

import os
import ssl
import sys
import urllib.parse
from collections.abc import Mapping

import certifi
import httpx
from sie_sdk import SIEClient, SIEError

from sie_placement.keda import GatewayError

TIMEOUT_S = 10.0
HTTP_OK_MIN = 200
HTTP_OK_MAX = 299
HTTP_ERROR_MIN = 400
MAX_ERROR_LENGTH = 512
REDACTED = "[redacted]"
_BEARER = "Bearer "
_UNEXPECTED_REPLY = "the gateway returned an unexpected response"


class _Replies:
    def __init__(self) -> None:
        self.status: int | None = None

    def __call__(self, response: httpx.Response) -> None:
        status = response.status_code
        self.status = status
        if status < HTTP_OK_MIN or HTTP_OK_MAX < status < HTTP_ERROR_MIN:
            raise GatewayError(f"the gateway answered HTTP {status}, which is neither a success nor an error")


def _is_file(entry: os.DirEntry[str]) -> bool:
    try:
        return entry.is_file()
    except OSError:
        return False


def _holds_files(path: str) -> bool:
    try:
        with os.scandir(path) as entries:
            return any(_is_file(entry) for entry in entries)
    except PermissionError:
        return True
    except OSError:
        return False


def _verification(base_url: str) -> ssl.SSLContext:
    cafile = capath = None
    if urllib.parse.urlsplit(base_url).scheme == "https":
        cafile = os.environ.get("SSL_CERT_FILE") or None
        capath = None if cafile else os.environ.get("SSL_CERT_DIR") or None
    if capath is not None and not any(_holds_files(entry) for entry in capath.split(os.pathsep) if entry):
        raise ValueError("the CA certificates that SSL_CERT_DIR names cannot be loaded")
    if cafile is None and capath is None:
        cafile = certifi.where()
        source = f"the default CA bundle {cafile}"
    else:
        source = f"the CA certificates that {'SSL_CERT_FILE' if cafile else 'SSL_CERT_DIR'} names"
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if sys.version_info >= (3, 13):
        context.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN | ssl.VERIFY_X509_STRICT
    try:
        context.load_verify_locations(cafile=cafile, capath=capath)
    except (OSError, ssl.SSLError):
        raise ValueError(f"{source} cannot be loaded") from None
    return context


def _client(base_url: str, api_key: str | None) -> tuple[SIEClient, _Replies, str | None]:
    replies = _Replies()
    verify = _verification(base_url)
    try:
        http_client = httpx.Client(
            base_url=base_url,
            timeout=TIMEOUT_S,
            follow_redirects=False,
            trust_env=False,
            verify=verify,
            event_hooks={"response": [replies]},
        )
    except httpx.InvalidURL:
        raise ValueError("the gateway URL is not a valid URL") from None
    try:
        client = SIEClient(base_url, api_key=api_key, timeout_s=TIMEOUT_S, http_client=http_client)
    except UnicodeError:
        http_client.close()
        raise ValueError("a gateway token holds a character that an HTTP header cannot carry") from None
    except BaseException:
        http_client.close()
        raise
    authorization = http_client.headers.get("Authorization", "")
    token = authorization.removeprefix(_BEARER) if authorization.startswith(_BEARER) else ""
    return client, replies, token or None


def _gateway_error(error: Exception, status: int | None, token: str | None) -> GatewayError:
    if isinstance(error, httpx.HTTPError):
        return GatewayError(f"the request failed ({type(error).__name__})")
    if status is not None and status >= HTTP_ERROR_MIN:
        return GatewayError(f"the gateway answered HTTP {status}")
    if status is not None or isinstance(error, AttributeError | RecursionError):
        return GatewayError(_UNEXPECTED_REPLY)
    message = str(error)
    if token:
        message = message.replace(token, REDACTED)
    return GatewayError(message[:MAX_ERROR_LENGTH])


class SdkPools:
    def __init__(self, base_url: str, api_key: str | None) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._client, self._replies, self._token = _client(base_url, api_key)

    def get_pool(self, name: str) -> Mapping[str, object] | None:
        self._replies.status = None
        try:
            pool = self._client.get_pool(name)
        except (SIEError, httpx.HTTPError, ValueError, AttributeError, RecursionError) as error:
            raise _gateway_error(error, self._replies.status, self._token) from None
        return pool

    def create_pool(
        self, name: str, gpus: dict[str, int], *, bundle: str, minimum_worker_count: int, queue_pool: str
    ) -> None:
        replies = None
        try:
            client, replies, _ = _client(self._base_url, self._api_key)
            with client:
                client.create_pool(
                    name, gpus, bundle=bundle, minimum_worker_count=minimum_worker_count, queue_pool=queue_pool
                )
        except (SIEError, httpx.HTTPError, ValueError, AttributeError, RecursionError) as error:
            raise _gateway_error(error, None if replies is None else replies.status, self._token) from None

    def close(self) -> None:
        self._client.close()


def connect(base_url: str, api_key: str | None) -> SdkPools:
    return SdkPools(base_url, api_key)
