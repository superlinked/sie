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

from dataclasses import dataclass

from typing_extensions import override

from superlinked.framework.common.space.embedding.model_based.engine.embedding_engine_config import (
    EmbeddingEngineConfig,
)

MINIMAX_EMBEDDING_BASE_URL = "https://api.minimax.io/v1"


@dataclass(frozen=True)
class MiniMaxEngineConfig(EmbeddingEngineConfig):
    """Configuration for MiniMax embedding engine.

    Args:
        api_key: MiniMax API key for authentication.
        base_url: MiniMax API base URL. Defaults to https://api.minimax.io/v1.
    """

    api_key: str = ""
    base_url: str = MINIMAX_EMBEDDING_BASE_URL

    @override
    def __str__(self) -> str:
        attributes = [
            f"base_url={self.base_url}",
        ]
        return f"{super().__str__()}, " + ", ".join(attributes)
