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

from collections.abc import Mapping

import structlog
from beartype.typing import Sequence, Type, cast

from superlinked.framework.common.const import constants
from superlinked.framework.common.exception import InvalidInputException
from superlinked.framework.common.interface.comparison_operand import (
    ComparisonOperation,
    _Or,
)
from superlinked.framework.common.interface.evaluated import Evaluated
from superlinked.framework.common.nlq.minimax import MiniMaxClientConfig
from superlinked.framework.common.nlq.open_ai import OpenAIClientConfig
from superlinked.framework.common.schema.id_schema_object import IdSchemaObject
from superlinked.framework.common.schema.schema_object import (
    SchemaField,
    String,
    StringList,
)
from superlinked.framework.common.util.type_validator import TypeValidator
from superlinked.framework.dsl.index.index import Index
from superlinked.framework.dsl.query.param import (
    IntParamType,
    NumericParamType,
    Param,
    ParamType,
    StringParamType,
)
from superlinked.framework.dsl.query.query_clause.base_looks_like_filter_clause import (
    BaseLooksLikeFilterClause,
)
from superlinked.framework.dsl.query.query_clause.hard_filter_clause import (
    HardFilterClause,
)
from superlinked.framework.dsl.query.query_clause.limit_clause import LimitClause
from superlinked.framework.dsl.query.query_clause.looks_like_filter_clause import (
    LooksLikeFilterClause,
)
from superlinked.framework.dsl.query.query_clause.looks_like_filter_clause_weights_by_space import (
    LooksLikeFilterClauseWeightBySpace,
)
from superlinked.framework.dsl.query.query_clause.nlq_clause import NLQClause
from superlinked.framework.dsl.query.query_clause.nlq_system_prompt_clause import (
    NLQSystemPromptClause,
)
from superlinked.framework.dsl.query.query_clause.overriden_now_clause import (
    OverriddenNowClause,
)
from superlinked.framework.dsl.query.query_clause.query_clause import (
    QueryClause,
    QueryClauseT,
)
from superlinked.framework.dsl.query.query_clause.radius_clause import RadiusClause
from superlinked.framework.dsl.query.query_clause.select_clause import SelectClause
from superlinked.framework.dsl.query.query_clause.similar_filter_clause import (
    SimilarFilterClause,
)
from superlinked.framework.dsl.query.query_clause.weight_by_space_clause import (
    WeightBySpaceClause,
)
from superlinked.framework.dsl.query.query_filter_validator import QueryFilterValidator
from superlinked.framework.dsl.query.query_user_config import QueryUserConfig
from superlinked.framework.dsl.query.space_weight_param_info import SpaceWeightParamInfo
from superlinked.framework.dsl.space.categorical_similarity_space import (
    CategoricalSimilaritySpace,
)
from superlinked.framework.dsl.space.has_space_field_set import HasSpaceFieldSet
from superlinked.framework.dsl.space.space import Space
from superlinked.framework.dsl.space.space_field_set import SpaceFieldSet

logger = structlog.getLogger()

SchemaFieldSelector = SchemaField | None | str | Param


class QueryDescriptor:  # pylint: disable=too-many-public-methods
    """
    A class representing a query object. Use .with_vector to run queries using a stored
    vector, or use .similar for queries where you supply the query at query-time. Or combine
    them, or even combine multiple .similar to supply different queries for each space in the
    Index.
    """

    @TypeValidator.wrap
    def __init__(
        self,
        index: Index,
        schema: IdSchemaObject,
        clauses: Sequence[QueryClause] | None = None,
        query_user_config: QueryUserConfig | None = None,
    ) -> None:
        self.__index = index
        self.__schema = schema
        self.__clauses: Sequence[QueryClause] = clauses if clauses else []
        self.__query_user_config = query_user_config if query_user_config else QueryUserConfig()
        self.__space_weight_param_info: SpaceWeightParamInfo | None = None
        QueryDescriptorValidator.validate(self)

    @property
    def index(self) -> Index:
        return self.__index

    @property
    def schema(self) -> IdSchemaObject:
        return self.__schema

    @property
    def clauses(self) -> Sequence[QueryClause]:
        return self.__clauses

    @property
    def with_metadata(self) -> bool:
        return self.__query_user_config.with_metadata

    @property
    def query_user_config(self) -> QueryUserConfig:
        return self.__query_user_config

    @property
    def _space_weight_param_info(self) -> SpaceWeightParamInfo:
        if self.__space_weight_param_info is None:
            self.__space_weight_param_info = SpaceWeightParamInfo.from_clauses(self.clauses)
        return self.__space_weight_param_info

    @TypeValidator.wrap
    def space_weights(self, weight_by_space: Mapping[Space, NumericParamType]) -> QueryDescriptor:
        clause = self.get_clause_by_type(WeightBySpaceClause)
        if clause is None:
            return self.__append_clauses([WeightBySpaceClause.from_params(weight_by_space, self.index._spaces)])
        extended_clause = clause.extend(weight_by_space, self.index._spaces)
        return self.__replace_clause(clause, extended_clause) if clause != extended_clause else self

    @TypeValidator.wrap
    def override_now(self, now: IntParamType) -> QueryDescriptor:
        clause = OverriddenNowClause.from_param(now)
        altered_query_descriptor = self.__append_clauses([clause])
        return altered_query_descriptor

    @TypeValidator.wrap
    def similar(
        self,
        space_field_set: HasSpaceFieldSet | SpaceFieldSet,
        param: ParamType,
        weight: NumericParamType = constants.DEFAULT_WEIGHT,
    ) -> QueryDescriptor:
        """
        Add a 'similar' clause to the query. Similar queries compile query inputs (like query text) into vectors
        using a space and then use the query_vector (weighted with weight param) to search
        in the referenced space of the index.

        Args:
            space_field_set (HasSpaceFieldSet | SpaceFieldSet): The space or field set to search within.
            param (ParamType): The parameter. Basically the query itself. It can be a fixed value,
            or a placeholder (Param) for later substitution.
            weight (NumericParamType, optional): The weight. Defaults to 1.0.

        Returns:
            Self: The query object itself.

        Raises:
            InvalidInputException: If the space is already bound in the query.
            InvalidInputException: If the schema is not in the similarity field's schema types.
        """
        field_set = (
            space_field_set.space_field_set if isinstance(space_field_set, HasSpaceFieldSet) else space_field_set
        )
        self.__validate_schema(field_set)
        clause = SimilarFilterClause.from_param(self.index._spaces, field_set, param, weight)
        altered_query_descriptor = self.__append_clauses([clause])
        return altered_query_descriptor

    def __validate_schema(self, field_set: SpaceFieldSet) -> None:
        if self.schema not in field_set.space._embedding_node_by_schema:
            raise InvalidInputException(f"'find' ({type(self.schema)}) is not in similarity field's schema types.")

    @TypeValidator.wrap
    def limit(self, limit: IntParamType | None) -> QueryDescriptor:
        """
        Set a limit to the number of results returned by the query.
        If the limit is None, -1 will be used, which is not handled by all databases.

        Args:
            limit (IntParamType): The maximum number of results to return.
        Returns:
            Self: The query object itself.
        """
        clause = LimitClause.from_param(limit or constants.DEFAULT_LIMIT)
        altered_query_descriptor = self.__append_clauses([clause])
        return altered_query_descriptor

    @TypeValidator.wrap
    def select(
        self,
        fields: SchemaFieldSelector | Sequence[SchemaFieldSelector] = None,
        metadata: Sequence[Space] | None = None,
    ) -> QueryDescriptor:
        """
        Select specific fields from the schema to be returned in the query results.

        Args:
            fields (SchemaFieldSelector | SchemaFieldSelector | None): The fields to select. Can be:
                - One or more SchemaField objects
                - One or more field names as strings
                - A single Param object that will be filled with fields later
                If no fields are provided, returns an empty selection.
            metadata (Sequence[Space] | None): The spaces identifying the requested vector parts.
                Defaults to None.
        Returns:
            Self: The query object itself.

        Raises:
            InvalidInputException:
                - If multiple Param objects are provided or Param is mixed with other field types.
                - If any of the fields are of unsupported types.
                - If any of the schema fields are not part of the schema.
                - If any of the spaces in metadata is not a Space.
        """
        field_list = (
            list(fields)
            if isinstance(fields, Sequence) and not isinstance(fields, str)
            else [] if fields is None else [fields]
        )
        param = self.__handle_select_param(field_list)
        clause = SelectClause.from_param(
            self.schema, param, [] if metadata is None else metadata, self.index._fields_to_exclude
        )
        altered_query_descriptor = self.__append_clauses([clause])
        return altered_query_descriptor

    def __handle_select_param(self, fields: Sequence[SchemaFieldSelector]) -> Param | list[str]:
        if len(fields) == 0:
            return []
        if len(fields) == 1 and isinstance(fields[0], Param):
            return fields[0]
        if any(isinstance(item, Param) for item in fields):
            raise InvalidInputException(
                "Query select clause can only contain either a single Param or non-Param fields."
            )
        return [field.name if isinstance(field, SchemaField) else cast(str, field) for field in fields]

    @TypeValidator.wrap
    def select_all(self, metadata: Sequence[Space] | None = None) -> QueryDescriptor:
        """
        Select all fields from the schema to be returned in the query results.

        Args:
            metadata (Sequence[Space] | None): The spaces identifying the requested vector parts.
                Defaults to None.

        Returns:
            Self: The query object itself.

        Raises:
            See `select`.
        """
        all_fields = list(filter(lambda field: field not in self.index._fields_to_exclude, self.__schema.schema_fields))
        return self.select(all_fields, metadata)

    @TypeValidator.wrap
    def with_natural_query(
        self,
        natural_query: StringParamType,
        client_config: OpenAIClientConfig | MiniMaxClientConfig,
        system_prompt: StringParamType | None = None,
    ) -> QueryDescriptor:
        """
        Sets a natural language query based on which empty Params will have values set.

        Args:
            natural_query (StringParamType): Query containing desired characteristics.
            client_config (OpenAIClientConfig | MiniMaxClientConfig): Client config to initialize the client with.
                Supports OpenAI and MiniMax LLM providers.
            system_prompt (StringParamType | None): Custom system prompt to use for the query. Defaults to None.
        Returns:
            Self: The query object itself.
        """
        clauses: list[QueryClause] = [NLQClause.from_param(natural_query, client_config)]
        if system_prompt:
            clauses.append(NLQSystemPromptClause.from_param(system_prompt))
        altered_query_descriptor = self.__append_clauses(clauses)
        self.__warn_if_nlq_is_used_without_recommended_param_descriptions(altered_query_descriptor)
        return altered_query_descriptor

    @TypeValidator.wrap
    def radius(self, radius: NumericParamType | None) -> QueryDescriptor:
        """
        Set a radius for the search in the query. The radius is a float value that
        determines the maximum distance to the input vector in the search.
        A lower radius value means that the enforced maximum distance is lower,
        therefore closer vectors are returned only.
        A radius of 0.05 means the lowest cosine similarity of items returned to the query vector is 0.95.
        The valid range is between 0 and 1. Otherwise it will raise InvalidInputException.

        Args:
            radius (NumericParamType | None): The maximum distance of the returned items from the query vector.
            If None, all results are returned.

        Returns:
            Self: The query object itself.

        Raises:
            InvalidInputException: If the radius is not between 0 and 1.
        """
        clause = RadiusClause.from_param(radius)
        return self.__append_clauses([clause])

    @TypeValidator.wrap
    def with_vector(
        self,
        schema_obj: IdSchemaObject,
        id_param: StringParamType,
        weight: NumericParamType | Mapping[Space, NumericParamType] = constants.DEFAULT_WEIGHT,
    ) -> QueryDescriptor:
        """
        Add a 'with_vector' clause to the query. This fetches an object with id_param
        from the db and uses the vector of that item for search purposes. Weighting
        happens at the space level (and if there is also a .similar query present,
        this part has weight=1 compared to the weight set at .similar for the query
        vector).

        Args:
            schema_obj (IdSchemaObject): The schema object the vector is originating from.
            id_param (StringParamType): The ID parameter. Eventually it is the ID of the vector to be used in the query.
            weight (NumericParamType | Mapping[Space, NumericParamType]): Weight attributed to the vector
                retrieved via this clause in the aggregated query. Can be fine-tuned with space-wise weighting,
                but resolving missing per-space weights with NLQ is not supported.

        Returns:
            Self: The query object itself.
        """
        if isinstance(weight, Mapping):
            adjusted_weights = self._adjust_with_vector_weights(weight)
            return self.__append_clauses(
                [LooksLikeFilterClauseWeightBySpace.from_param(schema_obj.id, id_param, adjusted_weights)]
            )
        return self.__append_clauses([LooksLikeFilterClause.from_param(schema_obj.id, id_param, weight)])

    def _adjust_with_vector_weights(
        self, weight_param_by_space: Mapping[Space, NumericParamType]
    ) -> dict[Space, NumericParamType]:
        missing_spaces = set(self.index._spaces).difference(weight_param_by_space.keys())
        missing_space_weights: dict[Space, NumericParamType] = {
            space: constants.DEFAULT_WEIGHT for space in missing_spaces
        }
        return dict(weight_param_by_space) | missing_space_weights

    @TypeValidator.wrap
    def filter(self, comparison_operation: ComparisonOperation[SchemaField] | _Or[SchemaField]) -> QueryDescriptor:
        """
        Add a 'filter' clause to the query. This filters the results from the db
        to only contain items based on the filtering input.
        E.g:
        filter(color_schema.color == "blue")
        filter(color_schema.color == Param("color_param"))
        filter(color_schema.color != "red")
        filter(color_schema.rating > 3)
        filter(color_schema.rating >= 3)
        filter(color_schema.rating < 3)
        filter(color_schema.rating <= 3)
        filter((color_schema.color == "blue") | (color_schema.color == "red"))
        filter(color_schema.categories.contains(["bright", "matte"]))
            - returns both bright and matte colors
        filter(color_schema.categories.not_contains(["bright", "matte"]))
            - returns colors that are non-bright and non-matte
        filter(color_schema.categories.contains_all(["bright", "blue"]))
            - returns colors that are bright and blue at the same time
        Args:
            comparison_operation ComparisonOperation[SchemaField]: The comparison operation.

        Returns:
            Self: The query object itself.
        """
        comparison_operations = cast(
            list[ComparisonOperation[SchemaField]],
            comparison_operation.operations if isinstance(comparison_operation, _Or) else [comparison_operation],
        )
        for operation in comparison_operations:
            QueryFilterValidator.validate_operation_is_supported(operation)
            QueryFilterValidator.validate_operation_operand_type(operation, allow_param=True)
            QueryFilterValidator.validate_operation_field_is_part_of_schema(operation, self.__schema)
        clauses = [HardFilterClause.from_param(operation) for operation in comparison_operations]
        altered_query_descriptor = self.__append_clauses(clauses)
        self.__warn_if_nlq_is_used_without_recommended_param_descriptions(altered_query_descriptor)
        return altered_query_descriptor

    @TypeValidator.wrap
    def include_metadata(self) -> QueryDescriptor:
        """
        Make per-item metadata to be returned in the query results.
        Current metadata includes space-wise partial scores.
        Returns:
            Self: The query object itself.
        """
        query_user_config = QueryUserConfig(
            with_metadata=True,
            redis_hybrid_policy=self.query_user_config.redis_hybrid_policy,
            redis_batch_size=self.query_user_config.redis_batch_size,
        )
        return QueryDescriptor(self.index, self.schema, self.clauses, query_user_config=query_user_config)

    @TypeValidator.wrap
    def replace_user_config(self, query_user_config: QueryUserConfig) -> QueryDescriptor:
        """
        Replace the current query user configuration with a new one.

        This method allows you to set custom configuration options for the query execution,
        such as whether to include metadata in results or Redis-specific settings.

        Args:
            query_user_config (QueryUserConfig): The new configuration to use for this query.

        Returns:
            QueryDescriptor: A new query descriptor with the updated configuration.
        """
        return QueryDescriptor(self.index, self.schema, self.clauses, query_user_config=query_user_config)

    def append_missing_mandatory_clauses(self) -> QueryDescriptor:
        clauses: list[QueryClause] = []
        if self.get_clause_by_type(LimitClause) is None:
            clauses.append(LimitClause.from_param(constants.DEFAULT_LIMIT))
        if self.get_clause_by_type(RadiusClause) is None:
            clauses.append(RadiusClause.from_param(None))
        if self.get_clause_by_type(SelectClause) is None:
            clauses.append(
                SelectClause.from_param(
                    self.schema, fields=[], vector_parts=[], fields_to_exclude=self.index._fields_to_exclude
                )
            )
        altered_query_descriptor = self.__append_clauses(clauses)
        return altered_query_descriptor._add_missing_space_weight_params()

    def _add_missing_space_weight_params(self) -> QueryDescriptor:
        weight_by_space_clause = weight_by_space_clause = self.get_clause_by_type(WeightBySpaceClause)
        if weight_by_space_clause is None:
            return self.__append_clauses([WeightBySpaceClause().add_missing_space_weight_params(self.index._spaces)])
        return self.__replace_clause(
            weight_by_space_clause, weight_by_space_clause.add_missing_space_weight_params(self.index._spaces)
        )

    def get_param_value_for_unset_space_weights(self) -> dict[str, float]:
        def is_global_weight_unset(param_name: str) -> bool:
            return not isinstance(param_by_name[param_name], Evaluated)

        global_param_name_by_space = self._space_weight_param_info.global_param_name_by_space
        param_names_by_space = self._space_weight_param_info.param_names_by_space
        param_by_name = {QueryClause.get_param(param).name: param for clause in self.clauses for param in clause.params}
        set_param_names = [
            param.item.param.name for clause in self.clauses for param in clause.params if isinstance(param, Evaluated)
        ]
        unset_spaces = [
            space for space in param_names_by_space.keys() if is_global_weight_unset(global_param_name_by_space[space])
        ]
        unset_used_spaces = {
            space
            for space, param_names in param_names_by_space.items()
            if space in unset_spaces and any(True for param_name in param_names if param_name in set_param_names)
        }
        unset_unused_spaces = set(unset_spaces).difference(unset_used_spaces)
        return {global_param_name_by_space[space]: constants.DEFAULT_WEIGHT for space in unset_used_spaces} | {
            global_param_name_by_space[space]: constants.DEFAULT_NOT_AFFECTING_WEIGHT for space in unset_unused_spaces
        }

    def get_clause_by_type(self, clause_type: Type[QueryClauseT]) -> QueryClauseT | None:
        clauses = self.get_clauses_by_type(clause_type)
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        raise InvalidInputException(f"Query cannot have more than one {clause_type.__name__}, got {len(clauses)}.")

    def get_clauses_by_type(self, clause_type: Type[QueryClauseT]) -> list[QueryClauseT]:
        return [clause for clause in self.clauses if isinstance(clause, clause_type)]

    def replace_clauses(self, clauses: Sequence[QueryClause]) -> QueryDescriptor:
        return QueryDescriptor(self.index, self.schema, clauses, self.__query_user_config)

    def __replace_clause(self, old_clause: QueryClause, new_clause: QueryClause) -> QueryDescriptor:
        clauses = [clause for clause in self.clauses if clause != old_clause]
        return QueryDescriptor(self.index, self.schema, clauses + [new_clause], self.__query_user_config)

    def __append_clauses(self, clauses: Sequence[QueryClause]) -> QueryDescriptor:
        return QueryDescriptor(self.index, self.schema, list(self.clauses) + list(clauses), self.__query_user_config)

    @classmethod
    def __warn_if_nlq_is_used_without_recommended_param_descriptions(cls, query_descriptor: QueryDescriptor) -> None:
        if not query_descriptor.get_clauses_by_type(NLQClause):
            return
        fields: set[SchemaField] = {field for space in query_descriptor.index._spaces for field in space._field_set}
        affected_param_names = [
            clause.value_param.name
            for clause in query_descriptor.get_clauses_by_type(HardFilterClause)
            if (
                isinstance(clause.value_param, Param)
                and clause.value_param.description is None
                and isinstance(clause.operand, (String, StringList))
                and clause.operand not in fields
            )
        ]
        if affected_param_names:
            affected_param_names_text = ", ".join(affected_param_names)
            logger.warning(
                f"When using a natural language query with a 'filter' for a field that has no corresponding"
                f" {type(CategoricalSimilaritySpace).__name__} and has a Param as a value,"
                f" it is recommended to provide a description for the Param that outlines the possible values."
                f" affected parameters: {affected_param_names_text}"
            )


class QueryDescriptorValidator:
    @staticmethod
    def validate(query_descriptor: QueryDescriptor) -> None:
        QueryDescriptorValidator.__validate_schema(query_descriptor)
        QueryDescriptorValidator.__validate_single_or_none_clause(query_descriptor)

    @staticmethod
    def __validate_schema(query_descriptor: QueryDescriptor) -> None:
        if not query_descriptor.index.has_schema(query_descriptor.schema):
            raise InvalidInputException(
                f"Index doesn't have the queried schema ({query_descriptor.schema._base_class_name})"
            )

    @staticmethod
    def __validate_single_or_none_clause(query_descriptor: QueryDescriptor) -> None:
        for clause_types in [
            cast(type, BaseLooksLikeFilterClause),
            LimitClause,
            RadiusClause,
            SelectClause,
            OverriddenNowClause,
        ]:
            query_descriptor.get_clause_by_type(clause_types)
