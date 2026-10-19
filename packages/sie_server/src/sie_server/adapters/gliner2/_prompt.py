"""Count the inference prompt with GLiNER2's own schema compiler."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy, deepcopy
from typing import Any


def prompt_counter(processor: Any) -> Callable[[Any], int]:
    """An independent inference compiler using the loaded tokenizer's settings.

    Tokenizers and word caches are mutable, so make a private tokenizer once
    per load, not per request. Keep the serving processor's mode and caches
    untouched. The copied processor's tokenizer cache is rebound explicitly:
    deepcopy of an LRU callable would still refer to the serving tokenizer.
    """
    inference = copy(processor)
    inference.tokenizer = deepcopy(processor.tokenizer)
    inference.word_splitter = deepcopy(processor.word_splitter)
    inference.sampling_config = deepcopy(processor.sampling_config)
    inference._token_cache = deepcopy(processor._token_cache)
    inference._tokenize_cached = inference.tokenizer.tokenize
    inference.change_mode(is_training=False)

    def count(schema: Any) -> int:
        # The public single-record entry point deep-copies the schema and
        # propagates errors, unlike 1.x's batch-collator fallback. Empty text
        # adds no synthetic period: all ids are prompt markers, schema strings,
        # separator and (for structured choices) the document-side prefix.
        compiled_schema = deepcopy(schema.build())
        if compiled_schema.get("classifications"):
            return _classification_tokens(inference, compiled_schema)
        record = inference.transform_and_format("", compiled_schema)
        return len(record.input_ids)

    return count


def _classification_tokens(processor: Any, schema: dict[str, Any]) -> int:
    """Render the plain single-task schema accepted by the generic adapter.

    The upstream inference compiler selects ``random.choice(["both"])`` even
    though there is only one possible mode. Call its rendering/encoding stages
    directly for this schema so counting does not advance process-global RNG.
    These options affect classification postprocessing, not its prompt. Refuse
    extensions here rather than silently ignoring new prompt-bearing fields.
    """
    tasks = schema["classifications"]
    if len(tasks) != 1 or any(schema.get(key) for key in ("entities", "relations", "json_structures")):
        raise RuntimeError("GLiNER2 prompt counter requires a single classification task")
    task = tasks[0]
    if set(task) - {"task", "labels", "multi_label", "cls_threshold", "true_label"}:
        raise RuntimeError("GLiNER2 classification prompt schema changed")
    tokens = processor._transform_schema(task["task"], task["labels"], processor.L_TOKEN, example_mode="both")
    return len(processor._format_input_with_mapping([tokens], [])["input_ids"])


def structure_schema(model: Any, structures: dict[str, list[str]]) -> Any:
    """Use the same field parser and schema builder as ``batch_extract_json``."""
    schema = model.create_schema()
    for parent, fields in structures.items():
        builder = schema.structure(parent)
        for spec in fields:
            name, dtype, choices, description = model._parse_field_spec(spec)
            builder.field(name, dtype=dtype, choices=choices, description=description)
    return schema
