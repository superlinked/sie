import asyncio
from typing import Any, cast

from sie_server.core.inference_output import EncodeOutput
from sie_server.core.prepared import ImagePayload, PreparedBatch, PreparedItem
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker.handlers.encode import EncodeHandler
from sie_server.types.inputs import Item


class EncodePipeline:
    @classmethod
    async def run_encode(
        cls,
        registry: ModelRegistry,
        model: str,
        items: list[Item],
        output_types: list[str],
        instruction: str | None,
        config: Any,
        is_query: bool,
        options: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], RequestTiming]:
        """Main entry point: preprocess then execute encoding.

        This is the unified encode path that handles text, image, and direct modes.
        """
        timing = RequestTiming()

        prepared_batch = await cls._prepare_batch(registry, model, items, config, is_query, timing)

        if prepared_batch is not None:
            # Batched worker path
            worker = await registry.start_worker(model)
            future = await worker.submit(
                prepared_items=prepared_batch.items,
                items=items,
                output_types=output_types,
                instruction=instruction,
                is_query=is_query,
                options=options,
                timing=timing,
            )
            worker_output = await future
            encode_output = cast("EncodeOutput", worker_output.output)
        else:
            # Direct adapter call (no batching) - run in thread to avoid blocking event loop
            encode_handler = EncodeHandler(model, registry.postprocessor_registry)
            adapter = registry.get(model)
            timing.start_inference()
            encode_output = await asyncio.to_thread(
                encode_handler.encode,
                adapter=adapter,
                items=items,
                output_types=output_types,
                is_query=is_query,
                options=options,
                instruction=instruction,
                prepared_items=None,
            )
            timing.end_inference()
            postprocess_ms = await asyncio.to_thread(
                encode_handler.post_process, is_query=is_query, options=options, encode_output=encode_output
            )
            timing.add_postprocessing_ms(postprocess_ms)

        formatted_output = EncodeHandler.format_output(encode_output, output_types=output_types)
        return formatted_output, timing

    @classmethod
    async def _prepare_batch(
        cls,
        registry: ModelRegistry,
        model: str,
        items: list[Item],
        config: Any,
        is_query: bool,
        timing: RequestTiming,
    ) -> PreparedBatch | None:
        """Run CPU preprocessing (tokenization/image processing) if a preprocessor exists.

        Returns None if no preprocessor is registered (direct adapter call path).
        """
        preprocessor_registry = registry.preprocessor_registry
        has_image_input = config.inputs is not None and config.inputs.image
        all_items_have_text = all(item.text is not None for item in items)
        any_items_have_images = any(item.images is not None and len(item.images) > 0 for item in items)

        # Text-only path: use text preprocessor
        if preprocessor_registry.has_preprocessor(model, "text") and all_items_have_text and not any_items_have_images:
            timing.start_tokenization()
            prepared_batch = await preprocessor_registry.prepare(model, items, config, is_query=is_query)
            timing.end_tokenization()
            return prepared_batch

        # Image path: use image preprocessor if available
        if has_image_input and any_items_have_images:
            timing.start_tokenization()
            if preprocessor_registry.has_preprocessor(model, "image"):
                prepared_batch = await preprocessor_registry.prepare(model, items, config, is_query=is_query)
            else:
                # Fallback: create passthrough prepared items for images
                prepared_items = []
                for i, item in enumerate(items):
                    images = item.images
                    image_count = len(images) if images else 1
                    prepared = PreparedItem(
                        payload=ImagePayload(pixel_values=None, original_size=(0, 0)),
                        cost=image_count,
                        original_index=i,
                    )
                    prepared_items.append(prepared)
                total_cost = sum(p.cost for p in prepared_items)
                prepared_batch = PreparedBatch(items=prepared_items, total_cost=total_cost, modality="image")
            timing.end_tokenization()
            return prepared_batch

        # No preprocessor available - return None to signal direct adapter call
        return None
