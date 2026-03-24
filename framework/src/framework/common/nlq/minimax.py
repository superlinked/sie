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

import re
from dataclasses import dataclass, field

import instructor  # type: ignore[import-untyped]
import openai as openAILib
import structlog
from beartype.typing import Any
from pydantic import BaseModel

from superlinked.framework.common.nlq.open_ai import suppress_tokenizer_warnings
from superlinked.framework.common.settings import settings

logger = structlog.getLogger()

MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_DEFAULT_MODEL = "MiniMax-M2.7"
MINIMAX_TEMPERATURE_VALUE: float = 0.01


@dataclass
class MiniMaxClientConfig:
    """Configuration for MiniMax LLM client used in natural language queries.

    MiniMax provides an OpenAI-compatible API at https://api.minimax.io/v1.
    Supported models include MiniMax-M2.7, MiniMax-M2.5, and MiniMax-M2.5-highspeed.
    """

    api_key: str
    model: str = MINIMAX_DEFAULT_MODEL
    base_url: str = field(default=MINIMAX_BASE_URL)


# Union type for NLQ client configs
NLQClientConfig = "OpenAIClientConfig | MiniMaxClientConfig"


def _strip_think_tags(text: str) -> str:
    """Strip <think>...</think> tags from MiniMax model responses."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class MiniMaxClient:
    def __init__(self, config: MiniMaxClientConfig) -> None:
        super().__init__()
        self._client = instructor.from_openai(
            openAILib.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
            )
        )
        self._model = config.model

    async def query(self, prompt: str, instructor_prompt: str, response_model: type[BaseModel]) -> dict[str, Any]:
        max_retries = settings.SUPERLINKED_NLQ_MAX_RETRIES
        with suppress_tokenizer_warnings():
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    response_model=response_model,
                    max_retries=max_retries,
                    messages=[
                        {"role": "system", "content": instructor_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=MINIMAX_TEMPERATURE_VALUE,
                )
            except instructor.InstructorRetryException as e:  # pylint: disable=no-member
                logger.warning(
                    f"MiniMax LLM validation followup failed after {max_retries} retries."
                    " Try increasing SUPERLINKED_NLQ_MAX_RETRIES."
                )
                raise e
        return response.model_dump()
