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

from superlinked.framework.common.nlq.minimax import (
    MINIMAX_BASE_URL,
    MINIMAX_DEFAULT_MODEL,
    MINIMAX_TEMPERATURE_VALUE,
    MiniMaxClient,
    MiniMaxClientConfig,
    _strip_think_tags,
)


class TestMiniMaxClientConfig:
    def test_default_values(self) -> None:
        config = MiniMaxClientConfig(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.model == MINIMAX_DEFAULT_MODEL
        assert config.base_url == MINIMAX_BASE_URL

    def test_custom_model(self) -> None:
        config = MiniMaxClientConfig(api_key="test-key", model="MiniMax-M2.5")
        assert config.model == "MiniMax-M2.5"

    def test_custom_base_url(self) -> None:
        config = MiniMaxClientConfig(api_key="test-key", base_url="https://custom.api.com/v1")
        assert config.base_url == "https://custom.api.com/v1"

    def test_all_params(self) -> None:
        config = MiniMaxClientConfig(
            api_key="my-key",
            model="MiniMax-M2.5-highspeed",
            base_url="https://proxy.example.com/v1",
        )
        assert config.api_key == "my-key"
        assert config.model == "MiniMax-M2.5-highspeed"
        assert config.base_url == "https://proxy.example.com/v1"


class TestStripThinkTags:
    def test_strip_single_think_tag(self) -> None:
        text = "<think>reasoning here</think>actual response"
        assert _strip_think_tags(text) == "actual response"

    def test_strip_multiline_think_tag(self) -> None:
        text = "<think>\nline1\nline2\n</think>\nactual response"
        assert _strip_think_tags(text) == "actual response"

    def test_no_think_tags(self) -> None:
        text = "plain response without tags"
        assert _strip_think_tags(text) == "plain response without tags"

    def test_empty_think_tag(self) -> None:
        text = "<think></think>response"
        assert _strip_think_tags(text) == "response"


class TestMiniMaxClient:
    @patch("superlinked.framework.common.nlq.minimax.instructor")
    @patch("superlinked.framework.common.nlq.minimax.openAILib")
    def test_init_creates_instructor_client(self, mock_openai: MagicMock, mock_instructor: MagicMock) -> None:
        config = MiniMaxClientConfig(api_key="test-key")
        mock_async_client = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_async_client

        MiniMaxClient(config)

        mock_openai.AsyncOpenAI.assert_called_once_with(
            api_key="test-key",
            base_url=MINIMAX_BASE_URL,
        )
        mock_instructor.from_openai.assert_called_once_with(mock_async_client)

    @patch("superlinked.framework.common.nlq.minimax.settings")
    @patch("superlinked.framework.common.nlq.minimax.instructor")
    @patch("superlinked.framework.common.nlq.minimax.openAILib")
    @pytest.mark.asyncio
    async def test_query_uses_minimax_temperature(
        self, mock_openai: MagicMock, mock_instructor: MagicMock, mock_settings: MagicMock
    ) -> None:
        mock_settings.SUPERLINKED_NLQ_MAX_RETRIES = 3
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"param1": "value1"}

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_instructor.from_openai.return_value = mock_client_instance

        config = MiniMaxClientConfig(api_key="test-key", model="MiniMax-M2.7")
        client = MiniMaxClient(config)

        from pydantic import BaseModel

        class TestModel(BaseModel):
            param1: str

        result = await client.query("test query", "system prompt", TestModel)

        mock_client_instance.chat.completions.create.assert_called_once()
        call_kwargs = mock_client_instance.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "MiniMax-M2.7"
        assert call_kwargs["temperature"] == MINIMAX_TEMPERATURE_VALUE
        assert result == {"param1": "value1"}

    @patch("superlinked.framework.common.nlq.minimax.settings")
    @patch("superlinked.framework.common.nlq.minimax.instructor")
    @patch("superlinked.framework.common.nlq.minimax.openAILib")
    @pytest.mark.asyncio
    async def test_query_passes_correct_messages(
        self, mock_openai: MagicMock, mock_instructor: MagicMock, mock_settings: MagicMock
    ) -> None:
        mock_settings.SUPERLINKED_NLQ_MAX_RETRIES = 2
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {}

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_instructor.from_openai.return_value = mock_client_instance

        config = MiniMaxClientConfig(api_key="test-key")
        client = MiniMaxClient(config)

        from pydantic import BaseModel

        class TestModel(BaseModel):
            pass

        await client.query("user question", "sys prompt", TestModel)

        call_kwargs = mock_client_instance.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "sys prompt"}
        assert messages[1] == {"role": "user", "content": "user question"}
