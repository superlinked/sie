from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from sie_server.core.inference_output import EncodeOutput
from sie_server.core.prepared import ImagePayload, PreparedBatch, PreparedItem
from sie_server.core.preprocessor.text import TextPreprocessor
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker.handlers.encode import EncodeHandler
from sie_server.types.inputs import InvalidInputError, Item

if TYPE_CHECKING:
    from sie_server.config.model import ModelConfig, ResolvedProfile
    from sie_server.core.preprocessor_registry import PreprocessorRegistry
    from sie_server.ipc_types import PreparedTokens


_ENCODE_OUTPUT_TYPES = frozenset({"dense", "sparse", "multivector"})


def _validated_encode_output_types(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise InvalidInputError("'output_types' must be a non-empty array")
    validated_output_types: list[str] = []
    for output_type in value:
        if not isinstance(output_type, str) or output_type not in _ENCODE_OUTPUT_TYPES:
            raise InvalidInputError(
                f"'output_types' entries must be one of {sorted(_ENCODE_OUTPUT_TYPES)}",
            )
        validated_output_types.append(output_type)
    return validated_output_types


def resolve_encode_output_types(
    config: ModelConfig,
    request_output_types: list[str] | None,
    selected_profile: ResolvedProfile,
    effective_options: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Resolve adapter and response output types for every encode ingress.

    Profiles may expose a postprocessed output that the adapter does not emit
    directly. MuVERA is the canonical example: the public response is dense,
    while the adapter must first produce multivectors. Keeping capability
    validation and that translation here prevents the HTTP and managed queue
    paths from drifting apart.

    Returns:
        ``(adapter_output_types, response_output_types)``.

    Raises:
        InvalidInputError: If the requested response includes an output not
            declared by the model or the selected profile.
    """
    if "output_types" in effective_options:
        requested_outputs: object = effective_options["output_types"]
    elif request_output_types is not None:
        requested_outputs = request_output_types
    else:
        requested_outputs = ["dense"]
    response_output_types = _validated_encode_output_types(requested_outputs)

    profile_output_types = selected_profile.runtime.get("output_types")
    if profile_output_types is not None:
        supported_outputs = set(_validated_encode_output_types(profile_output_types))
    else:
        supported_outputs = set(config.outputs) & _ENCODE_OUTPUT_TYPES

    unsupported = set(response_output_types) - supported_outputs
    if unsupported:
        msg = f"Model '{config.sie_id}' does not support output types: {unsupported}. Supported: {supported_outputs}"
        raise InvalidInputError(msg)

    adapter_output_types = list(response_output_types)
    if effective_options.get("muvera") is not None and "dense" in response_output_types:
        adapter_output_types = [output_type for output_type in response_output_types if output_type != "dense"]
        if "multivector" not in adapter_output_types:
            adapter_output_types.append("multivector")

    return adapter_output_types, response_output_types


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
        prepared_tokens_per_item: list[PreparedTokens | None] | None = None,
        response_output_types: list[str] | None = None,
        preformed_batch: bool = False,
    ) -> tuple[list[dict[str, Any]], RequestTiming]:
        """Main entry point: preprocess then execute encoding.

        This is the unified encode path that handles text, image, and direct modes.

        ``prepared_tokens_per_item`` is the worker-sidecar's fast-path
        token payload, aligned 1:1 with ``items``. When supplied,
        the text preprocessor skips its own tokenisation iff the
        tokenizer_id matches (see ``TextPreprocessor.try_prepare_from_prepared_tokens``).
        Absent / mismatched → Python tokenises exactly like today.

        ``response_output_types`` filters the final response. It differs from
        ``output_types`` only when the caller translated the adapter request
        (e.g. muvera asks the adapter for ``multivector`` while the postprocessor
        adds ``dense``); the response must then be filtered by the user-requested
        types, not the translated adapter types. Defaults to ``output_types``.

        ``preformed_batch=True`` is the worker-sidecar IPC path: Rust has
        already formed the batch, so the Python worker must execute it directly
        rather than submit it to the local BatchFormer again. Direct HTTP leaves
        this false and keeps Python-side batching for single-instance serving.
        """
        timing = RequestTiming()

        prepared_batch = await cls._prepare_batch(
            registry,
            model,
            items,
            config,
            is_query,
            timing,
            prepared_tokens_per_item=prepared_tokens_per_item,
        )

        if prepared_batch is not None:
            # Batched worker path
            worker = await registry.start_worker(model)
            submit = worker.submit_preformed if preformed_batch else worker.submit
            future = await submit(
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
            registry.touch_lru(model)
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

        # Unit-meter fallback: adapters that own their tokenization (flash
        # packing — the registry preprocessor is a char-count estimator
        # there) expose real per-item counts via ``EncodeOutput.extra``.
        # The preprocessor-recorded counts (authoritative too) win when both
        # exist; malformed/misaligned values are dropped rather than
        # mis-attributed — metering falls back to its reserve estimate.
        if timing.input_token_counts is None:
            extra_counts = encode_output.extra.get("input_token_counts")
            if (
                isinstance(extra_counts, list)
                and len(extra_counts) == len(items)
                and all(isinstance(count, int) and not isinstance(count, bool) for count in extra_counts)
            ):
                timing.input_token_counts = [int(count) for count in extra_counts]

        # Shared metering seam: adapters that own their tokenization but do not
        # pre-stamp ``extra`` (every flash text encoder — e5/bert_flash,
        # ColBERT, …) still expose real counts through the base
        # ``count_input_tokens`` hook, which re-tokenizes ``items`` with the
        # adapter's own tokenizer (the §P3.5 ground-truth basis). This is a
        # pure fallback: it never runs when the preprocessor or ``extra``
        # already recorded counts, so bge-m3(-flash) keep their exact values.
        # ``None`` (server-backed / image adapters) leaves the meter on its
        # reserve estimate rather than billing an approximation.
        if timing.input_token_counts is None:
            try:
                adapter = registry.get(model)
            except KeyError:
                adapter = None
            if adapter is not None:
                counts = await asyncio.to_thread(adapter.count_input_tokens, items)
                if (
                    isinstance(counts, list)
                    and len(counts) == len(items)
                    and all(isinstance(count, int) and not isinstance(count, bool) for count in counts)
                ):
                    timing.input_token_counts = [int(count) for count in counts]

        formatted_output = EncodeHandler.format_output(
            encode_output,
            output_types=response_output_types if response_output_types is not None else output_types,
        )
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
        *,
        prepared_tokens_per_item: list[PreparedTokens | None] | None = None,
    ) -> PreparedBatch | None:
        """Run CPU preprocessing (tokenization/image processing) if a preprocessor exists.

        Returns None if no preprocessor is registered (direct adapter call path).

        When ``prepared_tokens_per_item`` is supplied, the text path tries the
        Rust-tokenise fast path first via
        ``TextPreprocessor.try_prepare_from_prepared_tokens``; any rejection
        (mismatch, missing, drift, etc.) transparently falls back to the
        Python tokenizer so correctness is never at risk.
        """
        preprocessor_registry = registry.preprocessor_registry
        has_image_input = config.inputs is not None and config.inputs.image
        all_items_have_text = all(item.text is not None for item in items)
        any_items_have_images = any(item.images is not None and len(item.images) > 0 for item in items)

        # Text-only path: use text preprocessor
        if preprocessor_registry.has_preprocessor(model, "text") and all_items_have_text and not any_items_have_images:
            timing.start_tokenization()
            # Try the Rust-tokenise fast path first. Only the in-tree
            # `TextPreprocessor` implements it; other preprocessors
            # (e.g. `CharCountPreprocessor` for library-wrapped
            # adapters) return None and we fall through to the
            # normal path.
            if prepared_tokens_per_item is not None:
                fast_path = await cls._try_fast_path(
                    preprocessor_registry, model, items, prepared_tokens_per_item, config=config
                )
                if fast_path is not None:
                    timing.end_tokenization()
                    cls._record_input_token_counts(preprocessor_registry, model, fast_path, timing)
                    return fast_path

            prepared_batch = await preprocessor_registry.prepare(model, items, config, is_query=is_query)
            timing.end_tokenization()
            cls._record_input_token_counts(preprocessor_registry, model, prepared_batch, timing)
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

    @classmethod
    def _record_input_token_counts(
        cls,
        preprocessor_registry: PreprocessorRegistry,
        model: str,
        prepared_batch: PreparedBatch | None,
        timing: RequestTiming,
    ) -> None:
        """Record authoritative per-item input-token counts on ``timing``.

        Only when the model's text preprocessor is the real
        :class:`TextPreprocessor` — its ``PreparedItem.cost`` is
        ``len(input_ids)`` from the actual tokenizer. Char-count estimators
        (``CharCountPreprocessor`` for library-wrapped adapters) are skipped:
        the unit meter counts, it never approximates, so estimated costs
        must not masquerade as authoritative counts on the result path.
        """
        if prepared_batch is None or not prepared_batch.items:
            return
        try:
            preprocessor = preprocessor_registry.get_preprocessor(model, "text")
        except Exception:  # noqa: BLE001 — descriptor lookup must never fail the request
            return
        if not isinstance(preprocessor, TextPreprocessor):
            return
        counts = [0] * len(prepared_batch.items)
        for prepared in prepared_batch.items:
            index = prepared.original_index
            if not isinstance(index, int) or not 0 <= index < len(counts):
                return  # malformed batch — leave counts unset rather than mis-attribute
            counts[index] = int(prepared.cost)
        timing.input_token_counts = counts

    @classmethod
    async def _try_fast_path(
        cls,
        preprocessor_registry: PreprocessorRegistry,
        model: str,
        items: list[Item],
        prepared_tokens_per_item: list[PreparedTokens | None],
        *,
        config: Any,
    ) -> PreparedBatch | None:
        """Attempt the Rust-tokenise fast path. Returns ``None`` if the
        preprocessor for ``model`` isn't a plain ``TextPreprocessor``
        (e.g. ``CharCountPreprocessor`` for library-wrapped adapters),
        or if the fast path rejects the batch for any reason.

        Runs synchronously — the fast path is pure Python list
        manipulation, no tokenizer call. Skipping the ``to_thread``
        hop saves ~1.5 ms of scheduling overhead for the common case
        where every item hits the fast path.
        """
        preprocessor = preprocessor_registry.get_preprocessor(model, "text")
        if not isinstance(preprocessor, TextPreprocessor):
            return None
        return preprocessor.try_prepare_from_prepared_tokens(items, prepared_tokens_per_item, config=config)
