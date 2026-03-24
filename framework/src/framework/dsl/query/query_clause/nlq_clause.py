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

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import override

from superlinked.framework.common.exception import InvalidInputException
from superlinked.framework.common.nlq.minimax import MiniMaxClientConfig
from superlinked.framework.common.nlq.open_ai import OpenAIClientConfig
from superlinked.framework.dsl.query.clause_params import NLQClauseParams
from superlinked.framework.dsl.query.param import StringParamType
from superlinked.framework.dsl.query.query_clause.query_clause import QueryClause
from superlinked.framework.dsl.query.query_clause.single_value_param_query_clause import (
    SingleValueParamQueryClause,
)


@dataclass(frozen=True)
class NLQClause(SingleValueParamQueryClause):
    client_config: OpenAIClientConfig | MiniMaxClientConfig

    @override
    def get_altered_nql_params(self, nlq_clause_params: NLQClauseParams) -> NLQClauseParams:
        return NLQClauseParams(
            client_config=self.client_config,
            natural_query=self.__evaluate(),
            system_prompt=nlq_clause_params.system_prompt,
        )

    @override
    def _get_default_value_param_name(self) -> str:
        return "natural_query_param__"

    def __evaluate(self) -> str | None:
        value = self._get_value()
        if value is not None and not isinstance(value, str):
            raise InvalidInputException(f"NLQ prompt should be str, got {type(value).__name__}.")
        return value

    @classmethod
    def from_param(
        cls, natural_query: StringParamType, client_config: OpenAIClientConfig | MiniMaxClientConfig
    ) -> NLQClause:
        param = QueryClause._to_typed_param(natural_query, [str])
        return NLQClause(param, client_config)
