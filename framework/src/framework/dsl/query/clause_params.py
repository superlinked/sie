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

from dataclasses import dataclass, field, replace

from beartype.typing import Any, Mapping, Sequence

from superlinked.framework.common.interface.comparison_operand import (
    ComparisonOperation,
)
from superlinked.framework.common.nlq.minimax import MiniMaxClientConfig
from superlinked.framework.common.nlq.open_ai import OpenAIClientConfig
from superlinked.framework.common.schema.schema_object import SchemaField
from superlinked.framework.dsl.space.space import Space
from superlinked.framework.query.query_node_input import QueryNodeInput


@dataclass(frozen=True)
class KNNSearchClauseParams:
    limit: int | None = None
    filters: Sequence[ComparisonOperation[SchemaField]] = field(default_factory=list)
    schema_fields_to_return: Sequence[SchemaField] = field(default_factory=list)
    radius: float | None = None
    should_return_index_vector: bool = False

    def set_params(self, **params: Any) -> KNNSearchClauseParams:
        return replace(self, **params) if params else self


@dataclass(frozen=True)
class QueryVectorClauseParams:
    weight_by_space: Mapping[Space, float] = field(default_factory=dict[Space, float])
    context_time: int | None = None
    query_node_inputs_by_node_id: Mapping[str, list[QueryNodeInput]] = field(
        default_factory=dict[str, list[QueryNodeInput]]
    )

    def set_params(self, **params: Any) -> QueryVectorClauseParams:
        return replace(self, **params) if params else self


@dataclass(frozen=True)
class NLQClauseParams:
    client_config: OpenAIClientConfig | MiniMaxClientConfig | None = None
    natural_query: str | None = None
    system_prompt: str | None = None


@dataclass(frozen=True)
class MetadataExtractionClauseParams:
    vector_part_ids: Sequence[str] = field(default_factory=list[str])
