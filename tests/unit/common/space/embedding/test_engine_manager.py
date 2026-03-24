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

from superlinked.framework.common.space.embedding.model_based.embedding_engine_manager import (
    ENGINE_BY_HANDLER,
)
from superlinked.framework.common.space.embedding.model_based.engine.minimax_engine import (
    MiniMaxEngine,
)
from superlinked.framework.common.space.embedding.model_based.model_handler import (
    ModelHandler,
    TextModelHandler,
)


class TestEngineByHandler:
    def test_minimax_text_handler_registered(self) -> None:
        assert TextModelHandler.MINIMAX in ENGINE_BY_HANDLER
        assert ENGINE_BY_HANDLER[TextModelHandler.MINIMAX] is MiniMaxEngine

    def test_minimax_model_handler_registered(self) -> None:
        assert ModelHandler.MINIMAX in ENGINE_BY_HANDLER
        assert ENGINE_BY_HANDLER[ModelHandler.MINIMAX] is MiniMaxEngine

    def test_all_text_handlers_registered(self) -> None:
        for handler in TextModelHandler:
            assert handler in ENGINE_BY_HANDLER, f"{handler} not registered in ENGINE_BY_HANDLER"

    def test_all_model_handlers_registered(self) -> None:
        for handler in ModelHandler:
            assert handler in ENGINE_BY_HANDLER, f"{handler} not registered in ENGINE_BY_HANDLER"
