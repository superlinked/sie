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

from pathlib import Path

from beartype.typing import Sequence, cast
from typing_extensions import override

from superlinked.framework.common.dag.chunking_node import ChunkingNode
from superlinked.framework.common.dag.embedding_node import EmbeddingNode
from superlinked.framework.common.dag.node import Node
from superlinked.framework.common.dag.schema_field_node import SchemaFieldNode
from superlinked.framework.common.dag.text_embedding_node import TextEmbeddingNode
from superlinked.framework.common.data_types import Vector
from superlinked.framework.common.exception import InvalidInputException
from superlinked.framework.common.schema.id_schema_object import IdSchemaObject
from superlinked.framework.common.schema.schema_object import String
from superlinked.framework.common.space.config.aggregation.aggregation_config import (
    VectorAggregationConfig,
)
from superlinked.framework.common.space.config.embedding.text_similarity_embedding_config import (
    TextModelHandler,
    TextSimilarityEmbeddingConfig,
)
from superlinked.framework.common.space.config.normalization.normalization_config import (
    L2NormConfig,
)
from superlinked.framework.common.space.config.transformation_config import (
    TransformationConfig,
)
from superlinked.framework.common.space.embedding.model_based.engine.embedding_engine_config import (
    EmbeddingEngineConfig,
)
from superlinked.framework.common.space.embedding.model_based.engine.minimax_engine_config import (
    MiniMaxEngineConfig,
)
from superlinked.framework.common.space.embedding.model_based.engine.modal_engine_config import (
    ModalEngineConfig,
)
from superlinked.framework.common.space.embedding.model_based.singleton_embedding_engine_manager import (
    SingletonEmbeddingEngineManager,
)
from superlinked.framework.common.util.async_util import AsyncUtil
from superlinked.framework.common.util.type_validator import TypeValidator
from superlinked.framework.dsl.space.has_space_field_set import HasSpaceFieldSet
from superlinked.framework.dsl.space.space import Space
from superlinked.framework.dsl.space.space_field_set import SpaceFieldSet

TextInput = String | ChunkingNode

DEFAULT_CACHE_SIZE = 10000


class TextSimilaritySpace(Space[Vector, str], HasSpaceFieldSet):
    """
    A text similarity space is used to create vectors from documents in order to search in them
    later on. We only support (SentenceTransformers)[https://www.sbert.net/] models as they have
    finetuned pooling to encode longer text sequences most efficiently.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        text: TextInput | None | Sequence[TextInput | None],
        model: str,
        cache_size: int = DEFAULT_CACHE_SIZE,
        model_cache_dir: Path | None = None,
        model_handler: TextModelHandler = TextModelHandler.SENTENCE_TRANSFORMERS,
        embedding_engine_config: EmbeddingEngineConfig | None = None,
        salt: str | None = None,
    ) -> None:
        """
        Initialize the TextSimilaritySpace.

        Args:
            text (TextInput | list[TextInput]): The Text input or a list of Text inputs.
                It is a SchemaFieldObject (String), not a regular python string.
            model (str): The model used for text similarity.
            cache_size (int): The number of embeddings to be stored in an inmemory LRU cache.
                Set it to 0, to disable caching. Defaults to 10000.
            model_cache_dir (Path | None, optional): Directory to cache downloaded models.
                If None, uses the default cache directory. Defaults to None.
            model_handler (TextModelHandler, optional): The handler for the model,
                Defaults to ModelHandler.SENTENCE_TRANSFORMERS.
            embedding_engine_config (EmbeddingEngineConfig, optional): Configuration for the embedding engine.
                Defaults to EmbeddingEngineConfig().
            salt: (str | None, optional): Enables the creation of identical spaces to allow
                different weighted event definitions with them.
        """
        if embedding_engine_config is None:
            embedding_engine_config = EmbeddingEngineConfig()
        unchecked_texts: list[ChunkingNode | String] = self._fields_to_non_none_sequence(text)
        text_fields = [self._get_root(unchecked_text) for unchecked_text in unchecked_texts]
        super().__init__(text_fields, String, salt)
        self.text = SpaceFieldSet[str](self, set(text_fields))
        self.__validate_model_handler(model_handler, embedding_engine_config)
        self._transformation_config = self._init_transformation_config(
            model, model_cache_dir, cache_size, model_handler, embedding_engine_config
        )
        self._schema_node_map: dict[IdSchemaObject, EmbeddingNode[Vector, str]] = {
            self._get_root(unchecked_text).schema_obj: self._generate_embedding_node(
                unchecked_text, self._transformation_config, salt
            )
            for unchecked_text in unchecked_texts
        }
        self._model = model

    def __validate_model_handler(
        self, model_handler: TextModelHandler, embedding_engine_config: EmbeddingEngineConfig
    ) -> None:
        if model_handler == TextModelHandler.MODAL and not isinstance(embedding_engine_config, ModalEngineConfig):
            raise InvalidInputException(
                f"Invalid configuration: TextModelHandler.MODAL requires ModalEngineConfig, "
                f"but received {type(embedding_engine_config).__name__}. "
                f"Please provide a ModalEngineConfig instance when using MODAL as the model handler."
            )
        if model_handler == TextModelHandler.MINIMAX and not isinstance(embedding_engine_config, MiniMaxEngineConfig):
            raise InvalidInputException(
                f"Invalid configuration: TextModelHandler.MINIMAX requires MiniMaxEngineConfig, "
                f"but received {type(embedding_engine_config).__name__}. "
                f"Please provide a MiniMaxEngineConfig instance when using MINIMAX as the model handler."
            )

    def _get_root(self, text: String | Node) -> String:
        if isinstance(text, String):
            return text
        if isinstance(text, SchemaFieldNode):
            return cast(String, text.schema_field)
        return self._get_root(text.parents[0])

    def _generate_embedding_node(
        self, text: TextInput, transformation_config: TransformationConfig, salt: str | None
    ) -> TextEmbeddingNode:
        parent = text if isinstance(text, ChunkingNode) else SchemaFieldNode(text)
        return TextEmbeddingNode(
            parent=parent,
            transformation_config=transformation_config,
            fields_for_identification=self.text.fields,
            salt=salt,
        )

    @property
    @override
    def space_field_set(self) -> SpaceFieldSet:
        return self.text

    @property
    @override
    def transformation_config(self) -> TransformationConfig[Vector, str]:
        return self._transformation_config

    @property
    @override
    def _embedding_node_by_schema(
        self,
    ) -> dict[IdSchemaObject, EmbeddingNode[Vector, str]]:
        return self._schema_node_map

    @override
    def _create_default_node(self, schema: IdSchemaObject) -> EmbeddingNode[Vector, str]:
        return TextEmbeddingNode(None, self._transformation_config, self.text.fields, schema)

    @property
    @override
    def _annotation(self) -> str:
        return f"""The space encodes text using {self._model} embeddings.
        Negative weight would mean favoring text semantically dissimilar to the one present in the .similar clause
        corresponding to this space. Zero weight means insensitivity, positive weights mean favoring similar text.
        Larger positive weights increase the effect on similarity compared to other spaces.
        Accepts str type input for a corresponding .similar clause input.
        Affected fields: {self.space_field_set.field_names_text}."""

    @property
    @override
    def _allow_empty_fields(self) -> bool:
        return False

    def _init_transformation_config(  # pylint: disable=too-many-arguments
        self,
        model: str,
        model_cache_dir: Path | None,
        cache_size: int,
        model_handler: TextModelHandler,
        embedding_engine_config: EmbeddingEngineConfig,
    ) -> TransformationConfig[Vector, str]:
        length = AsyncUtil.run(
            SingletonEmbeddingEngineManager().calculate_length(
                model_handler, model, model_cache_dir, embedding_engine_config
            )
        )
        embedding_config = TextSimilarityEmbeddingConfig(
            str,
            model,
            model_cache_dir,
            length,
            embedding_engine_config,
            model_handler,
            cache_size,
        )
        aggregation_config = VectorAggregationConfig(Vector)
        normalization_config = L2NormConfig()
        return TransformationConfig(normalization_config, aggregation_config, embedding_config)


@TypeValidator.wrap
def chunk(
    text: String,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    split_chars_keep: list[str] | None = None,
    split_chars_remove: list[str] | None = None,
) -> ChunkingNode:
    """
    Create smaller chunks from the given text, a String SchemaFieldObject. It is helpful when you search
    for more granular information in your text corpus. It is recommended to try different chunk_sizes to
    find what fits best your use-case. Chunking respects word boundaries.

    Args:
        text (String): The String field the text of which is to be chunked.
        chunk_size (int | None, optional): The maximum size of each chunk in characters. Defaults to None, which means
        effectively using 250.
        chunk_overlap (int | None, optional): The maximum overlap between chunks in characters. Defaults to None, which
        means effectively using {}.
        split_chars_keep: Characters to split at, but also keep in the text. Should be characters that can signal
        meaningful breakpoints in the text. Effectively defaults to ["!", "?", "."].
        split_chars_remove: Characters to split at and remove from the text. Should be characters that can signal
        meaningful breakpoints in the text. Effectively defaults to ["\n"].

    Returns:
        ChunkingNode: The chunking node.
    """
    return ChunkingNode(
        SchemaFieldNode(text),
        chunk_size,
        chunk_overlap,
        split_chars_keep,
        split_chars_remove,
    )
