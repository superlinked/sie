"""Gateway pool reads and floor-carrier writes through the SIE Python SDK, installed by the keda extra."""

from __future__ import annotations

from collections.abc import Mapping

import httpx
from sie_sdk import SIEClient, SIEError

from sie_placement.keda import GatewayError

TIMEOUT_S = 10.0


class SdkPools:
    def __init__(self, client: SIEClient) -> None:
        self._client = client

    def get_pool(self, name: str) -> Mapping[str, object] | None:
        try:
            pool = self._client.get_pool(name)
        except (SIEError, httpx.HTTPError, ValueError) as error:
            raise GatewayError(str(error)) from None
        return pool

    def create_pool(
        self, name: str, gpus: dict[str, int], *, bundle: str, minimum_worker_count: int, queue_pool: str
    ) -> None:
        try:
            self._client.create_pool(
                name, gpus, bundle=bundle, minimum_worker_count=minimum_worker_count, queue_pool=queue_pool
            )
        except (SIEError, httpx.HTTPError, ValueError) as error:
            raise GatewayError(str(error)) from None

    def close(self) -> None:
        self._client.close()


def connect(base_url: str, api_key: str | None) -> SdkPools:
    return SdkPools(SIEClient(base_url, api_key=api_key, timeout_s=TIMEOUT_S))
