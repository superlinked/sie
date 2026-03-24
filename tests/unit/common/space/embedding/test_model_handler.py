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

from superlinked.framework.common.space.embedding.model_based.model_handler import (
    ModelHandler,
    TextModelHandler,
)


class TestModelHandler:
    def test_minimax_handler_exists(self) -> None:
        assert ModelHandler.MINIMAX.value == "minimax"

    def test_minimax_text_handler_exists(self) -> None:
        assert TextModelHandler.MINIMAX.value == "minimax"

    def test_all_handlers_present(self) -> None:
        handler_values = {h.value for h in ModelHandler}
        assert "minimax" in handler_values
        assert "sentence_transformers" in handler_values
        assert "open_clip" in handler_values
        assert "modal" in handler_values

    def test_all_text_handlers_present(self) -> None:
        handler_values = {h.value for h in TextModelHandler}
        assert "minimax" in handler_values
        assert "sentence_transformers" in handler_values
        assert "modal" in handler_values
