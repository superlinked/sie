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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from superlinked.framework.common.space.embedding.model_based.engine.minimax_engine import (
    MiniMaxEngine,
)
from superlinked.framework.common.space.embedding.model_based.engine.minimax_engine_config import (
    MINIMAX_EMBEDDING_BASE_URL,
    MiniMaxEngineConfig,
)


class TestMiniMaxEngineConfig:
    def test_default_values(self) -> None:
        config = MiniMaxEngineConfig(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.base_url == MINIMAX_EMBEDDING_BASE_URL

    def test_custom_base_url(self) -> None:
        config = MiniMaxEngineConfig(api_key="test-key", base_url="https://custom.api.com/v1")
        assert config.base_url == "https://custom.api.com/v1"

    def test_str_representation(self) -> None:
        config = MiniMaxEngineConfig(api_key="test-key")
        config_str = str(config)
        assert "base_url=" in config_str
        assert MINIMAX_EMBEDDING_BASE_URL in config_str

    def test_frozen_config(self) -> None:
        config = MiniMaxEngineConfig(api_key="test-key")
        with pytest.raises(AttributeError):
            config.api_key = "new-key"  # type: ignore[misc]


class TestMiniMaxEngine:
    def _create_engine(self, api_key: str = "test-key") -> MiniMaxEngine:
        config = MiniMaxEngineConfig(api_key=api_key)
        return MiniMaxEngine(model_name="embo-01", model_cache_dir=None, config=config)

    def test_is_query_prompt_supported(self) -> None:
        engine = self._create_engine()
        assert engine.is_query_prompt_supported() is True

    def test_get_clean_model_name(self) -> None:
        assert MiniMaxEngine._get_clean_model_name("embo-01") == "embo-01"

    def test_engine_key_contains_model_name(self) -> None:
        engine = self._create_engine()
        assert "embo-01" in engine.key

    @pytest.mark.asyncio
    async def test_embed_db_context(self) -> None:
        engine = self._create_engine()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "total_tokens": 10,
            "base_resp": {"status_code": 0},
        }

        with patch("superlinked.framework.common.space.embedding.model_based.engine.minimax_engine.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await engine.embed(["text1", "text2"], is_query_context=False)

            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            call_kwargs = mock_client.post.call_args
            payload = call_kwargs[1]["json"]
            assert payload["type"] == "db"
            assert payload["model"] == "embo-01"
            assert payload["texts"] == ["text1", "text2"]

    @pytest.mark.asyncio
    async def test_embed_query_context(self) -> None:
        engine = self._create_engine()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "vectors": [[0.7, 0.8, 0.9]],
            "total_tokens": 5,
            "base_resp": {"status_code": 0},
        }

        with patch("superlinked.framework.common.space.embedding.model_based.engine.minimax_engine.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await engine.embed(["search query"], is_query_context=True)

            assert result == [[0.7, 0.8, 0.9]]
            call_kwargs = mock_client.post.call_args
            payload = call_kwargs[1]["json"]
            assert payload["type"] == "query"

    @pytest.mark.asyncio
    async def test_embed_api_error(self) -> None:
        engine = self._create_engine()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("superlinked.framework.common.space.embedding.model_based.engine.minimax_engine.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from superlinked.framework.common.exception import UnexpectedResponseException

            with pytest.raises(UnexpectedResponseException, match="401"):
                await engine.embed(["test"], is_query_context=False)

    @pytest.mark.asyncio
    async def test_embed_base_resp_error(self) -> None:
        engine = self._create_engine()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "vectors": [],
            "base_resp": {"status_code": 1001, "status_msg": "Invalid API key"},
        }

        with patch("superlinked.framework.common.space.embedding.model_based.engine.minimax_engine.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from superlinked.framework.common.exception import UnexpectedResponseException

            with pytest.raises(UnexpectedResponseException, match="Invalid API key"):
                await engine.embed(["test"], is_query_context=False)

    @pytest.mark.asyncio
    async def test_embed_vector_count_mismatch(self) -> None:
        engine = self._create_engine()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "vectors": [[0.1, 0.2]],
            "total_tokens": 5,
            "base_resp": {"status_code": 0},
        }

        with patch("superlinked.framework.common.space.embedding.model_based.engine.minimax_engine.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from superlinked.framework.common.exception import UnexpectedResponseException

            with pytest.raises(UnexpectedResponseException, match="1 vectors for 2 inputs"):
                await engine.embed(["text1", "text2"], is_query_context=False)

    @pytest.mark.asyncio
    async def test_embed_sends_auth_header(self) -> None:
        engine = self._create_engine(api_key="my-secret-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "vectors": [[0.1]],
            "total_tokens": 1,
            "base_resp": {"status_code": 0},
        }

        with patch("superlinked.framework.common.space.embedding.model_based.engine.minimax_engine.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await engine.embed(["test"], is_query_context=False)

            call_kwargs = mock_client.post.call_args
            headers = call_kwargs[1]["headers"]
            assert headers["Authorization"] == "Bearer my-secret-key"
