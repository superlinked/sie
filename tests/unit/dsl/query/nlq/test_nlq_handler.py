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

from unittest.mock import AsyncMock, patch

import pytest

from superlinked.framework.common.nlq.minimax import MiniMaxClientConfig
from superlinked.framework.common.nlq.open_ai import OpenAIClientConfig
from superlinked.framework.dsl.query.nlq.nlq_handler import NLQHandler


class TestNLQHandlerDispatch:
    def test_accepts_openai_config(self) -> None:
        config = OpenAIClientConfig(api_key="test-key", model="gpt-4o")
        handler = NLQHandler(config)
        assert handler is not None

    def test_accepts_minimax_config(self) -> None:
        config = MiniMaxClientConfig(api_key="test-key", model="MiniMax-M2.7")
        handler = NLQHandler(config)
        assert handler is not None

    @pytest.mark.asyncio
    async def test_dispatches_to_minimax_client(self) -> None:
        config = MiniMaxClientConfig(api_key="test-key")

        with patch(
            "superlinked.framework.dsl.query.nlq.nlq_handler.MiniMaxClient"
        ) as mock_minimax_cls:
            mock_client = AsyncMock()
            mock_client.query.return_value = {"param1": "value1"}
            mock_minimax_cls.return_value = mock_client

            from pydantic import BaseModel

            class TestModel(BaseModel):
                param1: str

            handler = NLQHandler(config)
            result = await handler._execute_query("test query", "system prompt", TestModel)

            mock_minimax_cls.assert_called_once_with(config)
            assert result == {"param1": "value1"}

    @pytest.mark.asyncio
    async def test_dispatches_to_openai_client(self) -> None:
        config = OpenAIClientConfig(api_key="test-key", model="gpt-4o")

        with patch(
            "superlinked.framework.dsl.query.nlq.nlq_handler.OpenAIClient"
        ) as mock_openai_cls:
            mock_client = AsyncMock()
            mock_client.query.return_value = {"param1": "value1"}
            mock_openai_cls.return_value = mock_client

            from pydantic import BaseModel

            class TestModel(BaseModel):
                param1: str

            handler = NLQHandler(config)
            result = await handler._execute_query("test query", "system prompt", TestModel)

            mock_openai_cls.assert_called_once_with(config)
            assert result == {"param1": "value1"}
