"""ChromaDB embedding function backed by the Superlinked Inference Engine."""

from __future__ import annotations

import logging
import time
from typing import Sequence

import httpx
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from app.config import settings

logger = logging.getLogger(__name__)

_PROVISION_TIMEOUT = 1800  # max seconds to wait for model provisioning
_PROVISION_POLL = 30       # seconds between 202 retries
_REQUEST_TIMEOUT = 120     # per-request timeout
_POST_PROVISION_400_RETRIES = 3  # extra retries for transient 400s after provisioning


class SIEEmbeddingFunction(EmbeddingFunction[Documents]):
    """Calls the SIE ``/v1/encode`` endpoint in batches.

    The SIE endpoint may return **202 Accepted** while the model is
    provisioning on a GPU.  This class retries transparently until the
    model is ready (up to ``_PROVISION_TIMEOUT`` seconds).

    Inputs are split into batches of ``batch_size`` to avoid 400 errors
    that occur when the payload is too large for a freshly-provisioned model.
    """

    def __init__(
        self,
        *,
        api_endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._endpoint = (api_endpoint or settings.sie_api_endpoint).rstrip("/")
        self._api_key = api_key or settings.sie_api_key
        self._model = model or settings.sie_embed_model
        self._batch_size = batch_size or settings.sie_embed_batch_size
        self._url = f"{self._endpoint}/v1/encode/{self._model}"

    def __call__(self, input: Documents) -> Embeddings:
        all_embeddings: Embeddings = []
        total = len(input)
        for start in range(0, total, self._batch_size):
            end = min(start + self._batch_size, total)
            batch = input[start:end]
            logger.info(
                "Encoding batch %d–%d of %d texts",
                start + 1, end, total,
            )
            embeddings = self._encode_batch(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def _encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """POST a single batch to SIE with provisioning retry."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"items": [{"text": t} for t in texts]}

        deadline = time.monotonic() + _PROVISION_TIMEOUT
        was_provisioning = False
        post_provision_400s = 0

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"SIE model did not become ready within {_PROVISION_TIMEOUT}s"
                )

            with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
                response = client.post(self._url, json=payload, headers=headers)

            if response.status_code in (202, 503):
                was_provisioning = True
                logger.info(
                    "SIE returned %d (provisioning); retrying in %ds (%.0fs remaining)",
                    response.status_code, _PROVISION_POLL, remaining,
                )
                time.sleep(_PROVISION_POLL)
                continue

            if response.status_code == 400 and was_provisioning:
                post_provision_400s += 1
                if post_provision_400s > _POST_PROVISION_400_RETRIES:
                    logger.error(
                        "SIE still returning 400 after %d post-provisioning retries — body: %s",
                        post_provision_400s - 1, response.text[:1000],
                    )
                    response.raise_for_status()
                logger.warning(
                    "SIE returned 400 after provisioning (attempt %d/%d, may be transient); "
                    "retrying in %ds.  Body: %s",
                    post_provision_400s, _POST_PROVISION_400_RETRIES,
                    _PROVISION_POLL, response.text[:500],
                )
                time.sleep(_PROVISION_POLL)
                continue

            if response.status_code != 200:
                logger.error(
                    "SIE returned %d — body: %s",
                    response.status_code, response.text[:1000],
                )
                response.raise_for_status()

            data = response.json()
            return [item["dense"]["values"] for item in data["items"]]
