# Copyright 2024 Superlinked, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import httpx
import structlog
from beartype.typing import Sequence
from typing_extensions import override

from superlinked.framework.common.exception import UnexpectedResponseException
from superlinked.framework.common.space.embedding.model_based.embedding_input import (
    ModelEmbeddingInput,
)
from superlinked.framework.common.space.embedding.model_based.engine.embedding_engine import (
    EmbeddingEngine,
)
from superlinked.framework.common.space.embedding.model_based.engine.minimax_engine_config import (
    MiniMaxEngineConfig,
)

logger = structlog.getLogger()


class MiniMaxEngine(EmbeddingEngine[MiniMaxEngineConfig]):
    """Embedding engine using MiniMax's embo-01 model via the native MiniMax Embeddings API.

    MiniMax's embedding API uses a non-OpenAI-compatible format:
    - Request: {"model": "embo-01", "texts": [...], "type": "db"|"query"}
    - Response: {"vectors": [[...]], "total_tokens": N, "base_resp": {...}}

    The `type` parameter distinguishes between storage ("db") and search ("query") embeddings.
    """

    def __init__(self, model_name: str, model_cache_dir: Path | None, config: MiniMaxEngineConfig) -> None:
        super().__init__(model_name, model_cache_dir, config)
        self._api_key = config.api_key
        self._base_url = config.base_url.rstrip("/")

    @override
    async def embed(self, inputs: Sequence[ModelEmbeddingInput], is_query_context: bool) -> list[list[float]]:
        texts = [str(inp) for inp in inputs]
        embed_type = "query" if is_query_context else "db"
        url = f"{self._base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "texts": texts,
            "type": embed_type,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=60.0)
            if response.status_code != 200:
                raise UnexpectedResponseException(
                    f"MiniMax embedding API error: {response.status_code} - {response.text}"
                )
            data = response.json()
        base_resp = data.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            raise UnexpectedResponseException(
                f"MiniMax embedding API error: {base_resp.get('status_msg', 'unknown error')}"
            )
        vectors = data.get("vectors", [])
        if len(vectors) != len(texts):
            raise UnexpectedResponseException(
                f"MiniMax embedding API returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors

    @override
    def is_query_prompt_supported(self) -> bool:
        return True

    @classmethod
    @override
    def _get_clean_model_name(cls, model_name: str) -> str:
        return model_name
