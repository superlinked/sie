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

from pathlib import Path

import structlog
from beartype.typing import Awaitable, Callable, Mapping, Sequence

from superlinked.framework.common.data_types import Vector
from superlinked.framework.common.delayed_evaluator import DelayedEvaluator
from superlinked.framework.common.exception import NotImplementedException
from superlinked.framework.common.settings import settings
from superlinked.framework.common.space.embedding.model_based.embedding_input import (
    ModelEmbeddingInput,
)
from superlinked.framework.common.space.embedding.model_based.engine.embedding_engine import (
    EmbeddingEngine,
)
from superlinked.framework.common.space.embedding.model_based.engine.embedding_engine_config import (
    EmbeddingEngineConfig,
)
from superlinked.framework.common.space.embedding.model_based.engine.minimax_engine import (
    MiniMaxEngine,
)
from superlinked.framework.common.space.embedding.model_based.engine.modal_engine import (
    ModalEngine,
)
from superlinked.framework.common.space.embedding.model_based.engine.open_clip_engine import (
    OpenCLIPEngine,
)
from superlinked.framework.common.space.embedding.model_based.engine.sentence_transformers_engine import (
    SentenceTransformersEngine,
)
from superlinked.framework.common.space.embedding.model_based.model_dimension_cache import (
    MODEL_DIMENSION_BY_NAME,
)
from superlinked.framework.common.space.embedding.model_based.model_handler import (
    ModelHandler,
    ModelHandlerType,
    TextModelHandler,
)
from superlinked.framework.common.telemetry.telemetry_registry import telemetry

ENGINE_BY_HANDLER: Mapping[ModelHandlerType, type[EmbeddingEngine]] = {
    TextModelHandler.SENTENCE_TRANSFORMERS: SentenceTransformersEngine,
    TextModelHandler.MODAL: ModalEngine,
    TextModelHandler.MINIMAX: MiniMaxEngine,
    ModelHandler.SENTENCE_TRANSFORMERS: SentenceTransformersEngine,
    ModelHandler.OPEN_CLIP: OpenCLIPEngine,
    ModelHandler.MODAL: ModalEngine,
    ModelHandler.MINIMAX: MiniMaxEngine,
}

logger = structlog.getLogger()


class EmbeddingEngineManager:
    def __init__(self) -> None:
        self._key_to_engine: dict[str, EmbeddingEngine] = {}
        self._key_to_delayed_evaluator: dict[tuple[str, bool], DelayedEvaluator] = {}

    async def embed(  # pylint: disable=too-many-arguments
        self,
        model_handler: ModelHandlerType,
        model_name: str,
        inputs: Sequence[ModelEmbeddingInput],
        is_query_context: bool,
        model_cache_dir: Path | None,
        config: EmbeddingEngineConfig,
    ) -> list[Vector]:
        if not inputs:
            return []
        labels = {
            "model_name": model_name,
            "handler": model_handler.value,
            "is_query_context": is_query_context,
            "precision": config.precision.value,
            "n_input": len(inputs),
        }
        with telemetry.span("engine.embed", attributes=labels):
            telemetry.record_metric("engine.embed.count", len(inputs), labels=labels)
            engine = self._get_engine(model_handler, model_name, model_cache_dir, config)
            embeddings = await self._get_delayed_evaluator(engine, is_query_context).evaluate(inputs)
            return [Vector(embedding) for embedding in embeddings]

    async def calculate_length(
        self,
        model_handler: ModelHandlerType,
        model_name: str,
        model_cache_dir: Path | None,
        config: EmbeddingEngineConfig,
    ) -> int:
        engine_type = self._get_engine_type(model_handler)
        clean_model_name = engine_type.calculate_key(model_name, model_cache_dir, config)
        if not settings.MODEL_WARMUP and clean_model_name in MODEL_DIMENSION_BY_NAME:
            return MODEL_DIMENSION_BY_NAME[clean_model_name]
        length = await self._get_engine(model_handler, model_name, model_cache_dir, config).length
        if model_handler not in [ModelHandler.MODAL, TextModelHandler.MODAL, ModelHandler.MINIMAX, TextModelHandler.MINIMAX]:
            logger.info("Consider caching model dimension.", model_name=clean_model_name, dimension=length)
        return length

    def clear_engines(self) -> None:
        self._key_to_engine.clear()

    def _get_engine(
        self,
        model_handler: ModelHandlerType,
        model_name: str,
        model_cache_dir: Path | None,
        config: EmbeddingEngineConfig,
    ) -> EmbeddingEngine:
        engine_type = self._get_engine_type(model_handler)
        engine_key = engine_type.calculate_key(model_name, model_cache_dir, config)
        if engine_key not in self._key_to_engine:
            engine = engine_type(model_name, model_cache_dir, config)
            self._key_to_engine[engine_key] = engine
            self._create_delayed_evaluators(engine, engine_key)
        return self._key_to_engine[engine_key]

    def _create_delayed_evaluators(self, engine: EmbeddingEngine, engine_key: str) -> None:
        delay_ms = settings.BATCHED_EMBEDDING_WAIT_TIME_MS
        for is_query in [False, True]:
            embed_fn = self._create_engine_embed_fn(engine, is_query)
            self._key_to_delayed_evaluator[(engine_key, is_query)] = DelayedEvaluator(
                delay_ms=delay_ms, eval_fn=embed_fn, task_name=f"{engine._model_name} embed"
            )

    def _create_engine_embed_fn(
        self, engine: EmbeddingEngine, is_query: bool
    ) -> Callable[[Sequence[ModelEmbeddingInput]], Awaitable[list[list[float]]]]:
        async def embed_fn(inputs: Sequence[ModelEmbeddingInput]) -> list[list[float]]:
            return await engine.embed(inputs, is_query)

        return embed_fn

    def _get_delayed_evaluator(self, engine: EmbeddingEngine, is_query_context: bool) -> DelayedEvaluator:
        key = (engine.key, is_query_context and engine.is_query_prompt_supported())
        if (delayed_evaluator := self._key_to_delayed_evaluator.get(key)) is not None:
            return delayed_evaluator
        raise NotImplementedException("No delayed evaluator found.", engine_key=key)

    def _get_engine_type(self, model_handler: ModelHandlerType) -> type[EmbeddingEngine]:
        if (engine_type := ENGINE_BY_HANDLER.get(model_handler)) is not None:
            return engine_type
        raise NotImplementedException("Unsupported model handler.", model_handler=model_handler.value)
