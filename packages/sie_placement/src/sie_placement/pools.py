"""Gateway pool reads and floor-carrier writes through the SIE Python SDK, installed by the keda extra."""

from __future__ import annotations

from collections.abc import Mapping

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


def _require_success_or_error(response: httpx.Response) -> None:
    status = response.status_code
    if status < HTTP_OK_MIN or HTTP_OK_MAX < status < HTTP_ERROR_MIN:
        raise GatewayError(f"the gateway answered HTTP {status}, which is neither a success nor an error")


def _client(base_url: str, api_key: str | None) -> tuple[SIEClient, str | None]:
    try:
        http_client = httpx.Client(
            base_url=base_url,
            timeout=TIMEOUT_S,
            follow_redirects=False,
            event_hooks={"response": [_require_success_or_error]},
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
    return client, token or None


def _gateway_error(error: Exception, token: str | None) -> GatewayError:
    if isinstance(error, httpx.HTTPError):
        return GatewayError(f"the request failed ({type(error).__name__})")
    if isinstance(error, AttributeError | RecursionError):
        return GatewayError(_UNEXPECTED_REPLY)
    message = str(error)
    if token:
        message = message.replace(token, REDACTED)
    return GatewayError(message[:MAX_ERROR_LENGTH])


class SdkPools:
    def __init__(self, base_url: str, api_key: str | None) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._client, self._token = _client(base_url, api_key)

    def get_pool(self, name: str) -> Mapping[str, object] | None:
        try:
            pool = self._client.get_pool(name)
        except (SIEError, httpx.HTTPError, ValueError, AttributeError, RecursionError) as error:
            raise _gateway_error(error, self._token) from None
        return pool

    def create_pool(
        self, name: str, gpus: dict[str, int], *, bundle: str, minimum_worker_count: int, queue_pool: str
    ) -> None:
        try:
            client, _ = _client(self._base_url, self._api_key)
            with client:
                client.create_pool(
                    name, gpus, bundle=bundle, minimum_worker_count=minimum_worker_count, queue_pool=queue_pool
                )
        except (SIEError, httpx.HTTPError, ValueError, AttributeError, RecursionError) as error:
            raise _gateway_error(error, self._token) from None

    def close(self) -> None:
        self._client.close()


def connect(base_url: str, api_key: str | None) -> SdkPools:
    return SdkPools(base_url, api_key)
