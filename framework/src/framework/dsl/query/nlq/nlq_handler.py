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


from beartype.typing import Any, Sequence
from pydantic import BaseModel

from superlinked.framework.common.exception import UnexpectedResponseException
from superlinked.framework.common.nlq.minimax import MiniMaxClient, MiniMaxClientConfig
from superlinked.framework.common.nlq.open_ai import OpenAIClient, OpenAIClientConfig
from superlinked.framework.common.telemetry.telemetry_registry import telemetry
from superlinked.framework.dsl.query.nlq.nlq_clause_collector import NLQClauseCollector
from superlinked.framework.dsl.query.nlq.param_filler.query_param_model_builder import (
    QueryParamModelBuilder,
)
from superlinked.framework.dsl.query.nlq.param_filler.query_param_prompt_builder import (
    QueryParamPromptBuilder,
)
from superlinked.framework.dsl.query.query_clause.query_clause import QueryClause
from superlinked.framework.dsl.query.space_weight_param_info import SpaceWeightParamInfo


class NLQHandler:
    def __init__(self, client_config: OpenAIClientConfig | MiniMaxClientConfig) -> None:
        self.__client_config = client_config

    async def fill_params(
        self,
        natural_query: str,
        clauses: Sequence[QueryClause],
        space_weight_param_info: SpaceWeightParamInfo,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        clause_collector = NLQClauseCollector(clauses, space_weight_param_info)
        if clause_collector.all_params_have_value_set:
            return {}
        model_class = QueryParamModelBuilder.build(clause_collector)
        instructor_prompt = QueryParamPromptBuilder.calculate_instructor_prompt(clause_collector, system_prompt)
        with telemetry.span(
            "nlq.execute",
            attributes={
                "n_clauses": len(clauses),
                "model_class": model_class.__name__,
            },
        ):
            return await self._execute_query(natural_query, instructor_prompt, model_class)

    async def _execute_query(self, query: str, instructor_prompt: str, model_class: type[BaseModel]) -> dict[str, Any]:
        try:
            if isinstance(self.__client_config, MiniMaxClientConfig):
                client = MiniMaxClient(self.__client_config)
            else:
                client = OpenAIClient(self.__client_config)
            result = await client.query(query, instructor_prompt, model_class)
            return result
        except Exception as e:
            raise UnexpectedResponseException(f"Error executing natural language query: {str(e)}") from e
