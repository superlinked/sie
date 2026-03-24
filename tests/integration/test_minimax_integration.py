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

"""Integration tests for MiniMax provider support.

These tests verify that MiniMax components can be imported and used via
the public API (superlinked.framework), and that the engine manager
correctly routes to the MiniMax engine.

Tests requiring a live MiniMax API key are skipped unless MINIMAX_API_KEY is set.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMiniMaxPublicAPIExports:
    """Verify MiniMax classes are accessible from the public API."""

    def test_minimax_client_config_exported(self) -> None:
        from superlinked.framework import MiniMaxClientConfig

        config = MiniMaxClientConfig(api_key="test")
        assert config.api_key == "test"

    def test_minimax_engine_config_exported(self) -> None:
        from superlinked.framework import MiniMaxEngineConfig

        config = MiniMaxEngineConfig(api_key="test")
        assert config.api_key == "test"

    def test_text_model_handler_minimax_exported(self) -> None:
        from superlinked.framework import TextModelHandler

        assert hasattr(TextModelHandler, "MINIMAX")
        assert TextModelHandler.MINIMAX.value == "minimax"


class TestMiniMaxEmbeddingIntegration:
    """Integration tests for MiniMax embedding engine with the engine manager."""

    def test_engine_manager_creates_minimax_engine(self) -> None:
        from superlinked.framework.common.space.embedding.model_based.embedding_engine_manager import (
            EmbeddingEngineManager,
        )
        from superlinked.framework.common.space.embedding.model_based.engine.minimax_engine import (
            MiniMaxEngine,
        )
        from superlinked.framework.common.space.embedding.model_based.engine.minimax_engine_config import (
            MiniMaxEngineConfig,
        )
        from superlinked.framework.common.space.embedding.model_based.model_handler import (
            TextModelHandler,
        )

        manager = EmbeddingEngineManager()
        config = MiniMaxEngineConfig(api_key="test-key")
        engine = manager._get_engine(TextModelHandler.MINIMAX, "embo-01", None, config)
        assert isinstance(engine, MiniMaxEngine)

    @pytest.mark.asyncio
    async def test_engine_manager_embed_delegates_to_minimax(self) -> None:
        from superlinked.framework.common.space.embedding.model_based.embedding_engine_manager import (
            EmbeddingEngineManager,
        )
        from superlinked.framework.common.space.embedding.model_based.engine.minimax_engine_config import (
            MiniMaxEngineConfig,
        )
        from superlinked.framework.common.space.embedding.model_based.model_handler import (
            TextModelHandler,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "vectors": [[0.1, 0.2, 0.3]],
            "total_tokens": 3,
            "base_resp": {"status_code": 0},
        }

        with patch(
            "superlinked.framework.common.space.embedding.model_based.engine.minimax_engine.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            manager = EmbeddingEngineManager()
            config = MiniMaxEngineConfig(api_key="test-key")
            result = await manager.embed(
                model_handler=TextModelHandler.MINIMAX,
                model_name="embo-01",
                inputs=["hello world"],
                is_query_context=False,
                model_cache_dir=None,
                config=config,
            )

            assert len(result) == 1
            assert len(result[0].value) == 3


class TestMiniMaxNLQIntegration:
    """Integration tests for MiniMax NLQ (LLM) handler."""

    def test_nlq_handler_with_minimax_config(self) -> None:
        from superlinked.framework.common.nlq.minimax import MiniMaxClientConfig
        from superlinked.framework.dsl.query.nlq.nlq_handler import NLQHandler

        config = MiniMaxClientConfig(api_key="test-key", model="MiniMax-M2.7")
        handler = NLQHandler(config)
        assert handler is not None


@pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set - skipping live API tests",
)
class TestMiniMaxLiveAPI:
    """Live API tests - only run when MINIMAX_API_KEY is available."""

    @pytest.mark.asyncio
    async def test_live_embedding(self) -> None:
        from superlinked.framework.common.space.embedding.model_based.engine.minimax_engine import (
            MiniMaxEngine,
        )
        from superlinked.framework.common.space.embedding.model_based.engine.minimax_engine_config import (
            MiniMaxEngineConfig,
        )

        api_key = os.environ["MINIMAX_API_KEY"]
        config = MiniMaxEngineConfig(api_key=api_key)
        engine = MiniMaxEngine(model_name="embo-01", model_cache_dir=None, config=config)

        result = await engine.embed(["Hello, world!"], is_query_context=False)
        assert len(result) == 1
        assert len(result[0]) == 1536

    @pytest.mark.asyncio
    async def test_live_embedding_query_vs_db(self) -> None:
        from superlinked.framework.common.space.embedding.model_based.engine.minimax_engine import (
            MiniMaxEngine,
        )
        from superlinked.framework.common.space.embedding.model_based.engine.minimax_engine_config import (
            MiniMaxEngineConfig,
        )

        api_key = os.environ["MINIMAX_API_KEY"]
        config = MiniMaxEngineConfig(api_key=api_key)
        engine = MiniMaxEngine(model_name="embo-01", model_cache_dir=None, config=config)

        db_result = await engine.embed(["test text"], is_query_context=False)
        query_result = await engine.embed(["test text"], is_query_context=True)

        assert len(db_result[0]) == 1536
        assert len(query_result[0]) == 1536

    @pytest.mark.asyncio
    async def test_live_nlq_query(self) -> None:
        from pydantic import BaseModel

        from superlinked.framework.common.nlq.minimax import MiniMaxClient, MiniMaxClientConfig

        class SimpleResponse(BaseModel):
            color: str

        api_key = os.environ["MINIMAX_API_KEY"]
        config = MiniMaxClientConfig(api_key=api_key, model="MiniMax-M2.5-highspeed")
        client = MiniMaxClient(config)

        result = await client.query(
            "I want a red car",
            "Extract the color from the user's query. Return it as a JSON with a 'color' field.",
            SimpleResponse,
        )
        assert "color" in result
