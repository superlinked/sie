"""Per-item batching cost for the extract path.

The cost surface is consumed by ``BatchFormer`` (see
``core/batcher.py``) when deciding how many extract items can pack
into one forward pass. Two distinct adapter shapes feed this:

1. **Encoder-only** (GLiNER, GLiClass, …) — runtime is dominated by
   the input pass. Char/byte count is a faithful proxy.

2. **Decoder OCR** (LightOnOCR, GLM-OCR, PaddleOCR-VL, …) — runtime
   grows with both input *and* generated output, because KV cache
   inflates as tokens are decoded. Char/byte count under-counts the
   output side, so the batcher over-packs and the GPU stalls or
   OOMs. This is the bug behind issue #33.

The fix: callers operating on decoder-OCR adapters pass
``decoder_max_output_tokens`` so the cost reflects the worst-case
KV growth during decode. Encoder-only callers pass nothing — default
zero preserves the existing behaviour exactly.

The output-token uplift is intentionally additive (not multiplicative)
on the input cost. The two quantities have different units in this
function (byte count for documents, tokens for the decoder output);
adding them is approximation, not exact accounting. It is good
*enough* for the BatchFormer's packing decision — the goal is to stop
the batcher from packing 64 large images into one batch, not to
predict GPU runtime to the millisecond. A future refinement could
introduce per-adapter cost calibration; the current surface
deliberately keeps the API simple so all decoder-OCR adapters can
adopt it without coordinating on a calibration table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from sie_server.core.prepared import ExtractPreparedItem
from sie_server.types.inputs import is_document_input

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

# Upper bound on the number of entity labels a single extract request may carry.
# GLiNER-family adapters run one forward pass per label, so an unbounded label
# list is an uncapped compute vector on a public endpoint. Mirrors the score /
# rerank candidate cap (``MAX_SCORE_ITEMS``) in magnitude and named-400 shape.
MAX_EXTRACT_LABELS: Final[int] = 1000

# Bounds on an extract ``output_schema``'s nesting and size, checked at request
# ingress. The worker builds its batching key and adapters compile the schema by
# walking it recursively, so a pathologically deep schema would otherwise fail
# with a RecursionError deep inside the worker (or strand the other requests
# batched with it) instead of a 400 for the request that sent it. Nesting counts
# JSON objects and arrays; a JSON Schema object property adds two levels.
MAX_OUTPUT_SCHEMA_DEPTH: Final[int] = 128
MAX_OUTPUT_SCHEMA_VALUES: Final[int] = 100_000


def output_schema_shape_error(schema: Any) -> str | None:
    """Return why ``schema`` is too deeply nested or too large, or ``None``.

    Walks the value iteratively, so the check itself cannot hit Python's
    recursion limit.
    """
    stack: list[tuple[Any, int]] = [(schema, 1)]
    values = 0
    while stack:
        value, depth = stack.pop()
        values += 1
        if values > MAX_OUTPUT_SCHEMA_VALUES:
            return f"'output_schema' must contain at most {MAX_OUTPUT_SCHEMA_VALUES} values"
        if isinstance(value, dict):
            children = list(value.values())
        elif isinstance(value, list):
            children = value
        else:
            continue
        if depth > MAX_OUTPUT_SCHEMA_DEPTH:
            return f"'output_schema' must nest objects and arrays at most {MAX_OUTPUT_SCHEMA_DEPTH} levels deep"
        stack.extend((child, depth + 1) for child in children)
    return None


def extract_item_cost(
    item: Item,
    *,
    decoder_max_output_tokens: int = 0,
) -> int:
    """Return the batching cost for a single extract item.

    For document items the input cost is the byte size of the raw
    document so that very large PDFs/DOCX inputs do not get bundled
    into a single batch with other heavy items. For text items the
    input cost is the character count, which matches the historical
    behavior used by GLiNER/GLiClass adapters.

    Args:
        item: The extract item.
        decoder_max_output_tokens: Worst-case generated-output token
            count for decoder-OCR adapters. Added to the input cost to
            reflect KV-cache growth during decode. Encoder-only callers
            (GLiNER, GLiClass) leave this at 0 — backward-compatible
            with the pre-#33 behaviour. Decoder-OCR callers should
            pass the model's configured ``max_output_tokens`` (typically
            from ``model_config.tasks.extract.max_output_tokens`` or
            the equivalent generate-task field).
    """
    document = item.document
    if is_document_input(document):
        input_cost = len(document["data"])
    elif item.text:
        input_cost = len(item.text)
    else:
        input_cost = 0
    # ``decoder_max_output_tokens`` is in tokens; ``input_cost`` is in
    # bytes/chars. Adding them is intentional approximation — see the
    # module docstring. Negative values are silently clamped to zero so
    # a config typo doesn't subtract from the input cost.
    return input_cost + max(decoder_max_output_tokens, 0)


def build_extract_prepared_items(
    items: list[Item],
    *,
    decoder_max_output_tokens: int = 0,
    item_costs: list[int] | None = None,
) -> list[ExtractPreparedItem]:
    """Build PreparedItems for a batch of extract items.

    ``decoder_max_output_tokens`` is forwarded to :func:`extract_item_cost`
    for each item — see that function's docstring for the rationale.
    Decoder-OCR call sites must pass this; encoder-only call sites
    leave it at the default and get the pre-#33 byte-count behaviour.

    ``item_costs`` replaces the per-item input cost when an adapter reports
    its own (``ModelAdapter.extract_item_costs``), e.g. because it runs one
    forward row per (item, question) pair. It is ignored unless it holds one
    non-negative integer per item, so a malformed estimate falls back to the
    character count rather than mis-sizing a batch.
    """
    usable_costs = _usable_item_costs(item_costs, len(items))
    prepared: list[ExtractPreparedItem] = []
    for i, item in enumerate(items):
        if usable_costs is None:
            cost = extract_item_cost(item, decoder_max_output_tokens=decoder_max_output_tokens)
        else:
            cost = usable_costs[i] + max(decoder_max_output_tokens, 0)
        prepared.append(ExtractPreparedItem(cost=cost, original_index=i))
    return prepared


def adapter_extract_item_costs(
    adapter: object,
    items: list[Item],
    *,
    labels: list[str] | None = None,
    output_schema: dict[str, Any] | None = None,
    instruction: str | None = None,
    options: dict[str, Any] | None = None,
) -> list[int] | None:
    """Ask ``adapter`` for per-item extract costs; ``None`` when it has none or fails."""
    hook = getattr(adapter, "extract_item_costs", None)
    if not callable(hook):
        return None
    try:
        costs = hook(items, labels=labels, output_schema=output_schema, instruction=instruction, options=options)
    except Exception:  # noqa: BLE001 — a cost estimate must never fail the request
        return None
    return _usable_item_costs(costs, len(items))


def _usable_item_costs(costs: object, expected_len: int) -> list[int] | None:
    if not isinstance(costs, list) or len(costs) != expected_len:
        return None
    usable: list[int] = []
    for cost in costs:
        if not isinstance(cost, int) or isinstance(cost, bool) or cost < 0:
            return None
        usable.append(cost)
    return usable
