"""Native causal scoring contracts with mocked loaders and tiny CPU tensors."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
import sentence_transformers
import torch
from jinja2.sandbox import ImmutableSandboxedEnvironment
from sentence_transformers.base import modules as native_modules
from sentence_transformers.cross_encoder.modules import LogitScore
from sie_server.adapters.errors import InputTooLongError
from sie_server.adapters.native_causal_cross_encoder import adapter as adapter_module
from sie_server.adapters.native_causal_cross_encoder.adapter import (
    NativeCausalCrossEncoderAdapter,
    _NativeInputTooLongError,
    _NativeRows,
)
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.oom import OomRecoveryConfig, OomRecoveryStats
from sie_server.core.worker.handlers.score import ScoreHandler
from sie_server.core.worker.oom_recovery import BatchExecutor, ConfigGroup
from sie_server.queue_executor import _inference_exception_outcome
from sie_server.types.inputs import AudioInput, InvalidInputError, Item

# The query/document branch of the pinned publisher's chat_template.jinja.
_PAIR_TEMPLATE = """{%- set query = messages | selectattr('role', 'eq', 'query') | map(attribute='content') | list -%}
{%- set document = messages | selectattr('role', 'eq', 'document') | map(attribute='content') | list -%}
{{- '<|im_start|>system\\n' + (query | first) + '<|im_end|>\\n' -}}
{{- '<|im_start|>user\\n' + (document | first) + '<|im_end|>\\n' -}}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}
"""


class ToyTokenizer:
    padding_side = "left"

    def __init__(self) -> None:
        self.chat_template: str | None = _PAIR_TEMPLATE

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        assert kwargs.get("tokenize") is False
        return (
            ImmutableSandboxedEnvironment()
            .from_string(self.chat_template or "")
            .render(messages=messages, add_generation_prompt=kwargs["add_generation_prompt"])
        )

    def row(self, query: str, document: str) -> list[int]:
        rendered = self.apply_chat_template(
            [{"role": "query", "content": query}, {"role": "document", "content": document}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return [ord(character) + 1 for character in rendered]


class ToyTransformer:
    def __init__(self) -> None:
        self.transformer_task = "text-generation"
        self.module_output_name = "causal_logits"
        self.can_flatten_inputs = False
        self.do_lower_case = False
        self.modality_config = {
            "text": {"method": "forward", "method_output_name": "logits"},
            "message": {"method": "forward", "method_output_name": "logits", "format": "flat"},
        }
        self.processing_kwargs = {"chat_template": {"add_generation_prompt": True}}
        self.tokenizer = ToyTokenizer()
        self.processor = self.tokenizer


class ToyCrossEncoder:
    """Only nested processing_kwargs controls this fake native processor."""

    def __init__(self) -> None:
        self.transformer = ToyTransformer()
        self.scorer = LogitScore(true_token_id=9454, false_token_id=None)
        self.activation_fn: Any = torch.nn.Identity()
        self.extra_modules: list[Any] = []
        self.max_length = 32768
        self.preprocess_calls: list[tuple[list[tuple[str, str]], dict[str, Any]]] = []
        self.forward_calls: list[dict[str, Any]] = []
        self.values: dict[str, float] = {}
        self.output: Any = None
        self.mutate_batch: str | None = None
        self.malformed: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self.eval = Mock()

    def children(self) -> Iterator[Any]:
        yield self.transformer
        yield self.scorer
        yield from self.extra_modules

    def preprocess(self, pairs: list[tuple[str, str]], **kwargs: Any) -> dict[str, Any]:
        self.preprocess_calls.append((list(pairs), kwargs))
        truncate = self.transformer.processing_kwargs.get("text", {}).get("truncation", True)
        rows = [self.transformer.tokenizer.row(query, document) for query, document in pairs]
        if truncate:
            rows = [row[: self.max_length] for row in rows]
        width = max(map(len, rows))
        ids = [[0] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        features = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "logits_to_keep": 1,
            "modality": "message",
            "_pairs": pairs,
        }
        if len(pairs) > 1 and self.mutate_batch == "ids":
            features["input_ids"][0, -1] += 1
        elif len(pairs) > 1 and self.mutate_batch == "counts":
            last_padding = (features["attention_mask"][0] == 0).nonzero()[-1, 0]
            features["attention_mask"][0, last_padding] = 1
        return self.malformed(features) if self.malformed is not None else features

    def forward(self, features: dict[str, Any]) -> dict[str, Any]:
        self.forward_calls.append(features)
        scores = self.output
        if scores is None:
            scores = torch.tensor([[self.values.get(document, 5.25)] for _, document in features["_pairs"]])
        return {"scores": scores}


@pytest.fixture
def native_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[ToyCrossEncoder, Mock, Mock, Path]:
    model = ToyCrossEncoder()

    def construct(*args: Any, **kwargs: Any) -> ToyCrossEncoder:
        model.max_length = kwargs["max_length"]
        return model

    loader = Mock(side_effect=construct)
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", loader)
    monkeypatch.setattr(native_modules, "Transformer", ToyTransformer)
    modules_file = tmp_path / "modules.json"
    modules_file.write_text(
        json.dumps(
            [
                {"idx": 0, "name": "0", "path": "", "type": adapter_module._MODULE_TYPES[0]},
                {"idx": 1, "name": "1", "path": "1_LogitScore", "type": adapter_module._MODULE_TYPES[1]},
            ]
        )
    )
    download = Mock(return_value=str(modules_file))
    monkeypatch.setattr(adapter_module, "hf_hub_download", download)
    return model, loader, download, modules_file


@pytest.fixture
def loaded(
    native_loader: tuple[ToyCrossEncoder, Mock, Mock, Path],
) -> tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder]:
    adapter = NativeCausalCrossEncoderAdapter("zeroentropy/zerank-2-reranker", revision="a" * 40)
    adapter.load("cpu")
    return adapter, native_loader[0]


@pytest.mark.parametrize(
    ("device", "precision", "dtype"),
    [
        ("cpu", None, torch.float32),
        ("cuda:0", None, torch.bfloat16),
        ("cpu", "float16", torch.float16),
        ("cpu", "bfloat16", torch.bfloat16),
        ("cpu", "float32", torch.float32),
    ],
)
def test_native_constructor_and_saved_processing(
    native_loader: tuple[ToyCrossEncoder, Mock, Mock, Path], device: str, precision: Any, dtype: torch.dtype
) -> None:
    model, loader, download, _ = native_loader
    saved = copy.deepcopy(model.transformer.processing_kwargs)
    adapter = NativeCausalCrossEncoderAdapter(
        "zeroentropy/zerank-2-reranker", revision="b" * 40, compute_precision=precision, max_seq_length=32768
    )
    adapter.load(device)
    download.assert_called_once_with("zeroentropy/zerank-2-reranker", "modules.json", revision="b" * 40)
    loader.assert_called_once_with(
        "zeroentropy/zerank-2-reranker",
        revision="b" * 40,
        device=device,
        max_length=32768,
        trust_remote_code=False,
        model_kwargs={"dtype": dtype, "attn_implementation": "sdpa"},
    )
    assert model.transformer.processing_kwargs == {**saved, "text": {"truncation": False}}
    model.eval.assert_called_once()


def test_local_native_modules_are_required_before_loading(
    native_loader: tuple[ToyCrossEncoder, Mock, Mock, Path], tmp_path: Path
) -> None:
    _, loader, download, modules_file = native_loader
    adapter = NativeCausalCrossEncoderAdapter(tmp_path, revision="b" * 40)
    modules_file.unlink()
    with pytest.raises(RuntimeError, match=r"requires saved modules\.json"):
        adapter.load("cpu")
    loader.assert_not_called()
    download.assert_not_called()


@pytest.mark.parametrize(
    "metadata",
    [None, {}, [], [{"idx": 0, "type": "fallback"}], [{"idx": 0, "type": "fallback"}, {"idx": 1, "type": "fallback"}]],
)
def test_missing_or_altered_module_metadata_never_uses_fallback(
    native_loader: tuple[ToyCrossEncoder, Mock, Mock, Path], metadata: Any
) -> None:
    _, loader, _, modules_file = native_loader
    modules_file.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="saved Transformer and LogitScore"):
        NativeCausalCrossEncoderAdapter("publisher/model").load("cpu")
    loader.assert_not_called()


@pytest.mark.parametrize(
    ("target", "attribute", "value"),
    [
        ("transformer", "transformer_task", "sequence-classification"),
        ("transformer", "module_output_name", "scores"),
        ("transformer", "can_flatten_inputs", True),
        ("transformer", "do_lower_case", True),
        ("scorer", "true_token_id", 1),
        ("scorer", "true_token_id", True),
        ("scorer", "false_token_id", 442),
        ("scorer", "module_input_name", "logits"),
        ("model", "activation_fn", torch.nn.Sigmoid()),
    ],
)
def test_native_scorer_and_generation_identity_are_validated(
    native_loader: tuple[ToyCrossEncoder, Mock, Mock, Path], target: str, attribute: str, value: Any
) -> None:
    model = native_loader[0]
    owner = model if target == "model" else getattr(model, target)
    setattr(owner, attribute, value)
    with pytest.raises(RuntimeError, match="raw positive-token scorer"):
        NativeCausalCrossEncoderAdapter("publisher/model").load("cpu")
    assert not model.forward_calls


def test_extra_native_modules_are_rejected(native_loader: tuple[ToyCrossEncoder, Mock, Mock, Path]) -> None:
    native_loader[0].extra_modules.append(torch.nn.Identity())
    with pytest.raises(RuntimeError, match="module classes"):
        NativeCausalCrossEncoderAdapter("publisher/model").load("cpu")


def test_wrong_native_module_class_is_rejected(native_loader: tuple[ToyCrossEncoder, Mock, Mock, Path]) -> None:
    native_loader[0].scorer = torch.nn.Identity()
    with pytest.raises(RuntimeError, match="module classes"):
        NativeCausalCrossEncoderAdapter("publisher/model").load("cpu")


@pytest.mark.parametrize("change", ["format", "padding", "template", "generation", "thinking"])
def test_native_format_padding_and_template_are_validated(
    native_loader: tuple[ToyCrossEncoder, Mock, Mock, Path], change: str
) -> None:
    model = native_loader[0]
    if change == "format":
        model.transformer.modality_config["message"]["format"] = "nested"
    elif change == "padding":
        model.transformer.tokenizer.padding_side = "right"
    elif change == "template":
        model.transformer.tokenizer.chat_template = None
    elif change == "generation":
        model.transformer.processing_kwargs = {"chat_template": {"add_generation_prompt": False}}
    else:
        model.transformer.tokenizer.chat_template += "<think>"
    with pytest.raises(RuntimeError, match=r"formatter|left padding|template|generation-prefix|publisher format"):
        NativeCausalCrossEncoderAdapter("publisher/model").load("cpu")


def test_real_native_logit_score_preserves_raw_positive_logits() -> None:
    # Exercise the scoring module only; no causal model or checkpoint is loaded.
    logits = torch.zeros((2, 1, 9455))
    logits[:, 0, 0] = 1000
    logits[:, 0, 9454] = torch.tensor([5.25, -3.5])
    output = LogitScore(true_token_id=9454, false_token_id=None)({"causal_logits": logits})
    torch.testing.assert_close(output["scores"], torch.tensor([[5.25], [-3.5]]))


def test_publisher_template_preserves_unicode_newlines_and_source_tail() -> None:
    query, document = "Question\n東京", "Élodie\nAnswer at source-tail-✓"
    rendered = ToyTokenizer().apply_chat_template(
        [{"role": "query", "content": query}, {"role": "document", "content": document}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered == (
        f"<|im_start|>system\n{query}<|im_end|>\n<|im_start|>user\n{document}<|im_end|>\n<|im_start|>assistant\n"
    )
    assert "<think>" not in rendered


def test_full_native_rows_raw_scores_and_exact_usage(
    loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder],
) -> None:
    adapter, model = loaded
    query = "q\n東京"
    docs = ["short", "longer document ends at source-tail-✓"]
    model.values = {docs[0]: -3.5, docs[1]: 5.25}
    output = adapter.score_pairs([Item(text=query)] * 2, [Item(text=document) for document in docs])
    np.testing.assert_array_equal(output.scores, [-3.5, 5.25])
    assert output.scores.dtype == np.float32
    assert output.input_token_counts == [len(model.transformer.tokenizer.row(query, document)) for document in docs]
    assert output.content_token_counts is None
    assert all(kwargs == {} for _, kwargs in model.preprocess_calls)
    assert all(features["logits_to_keep"] == 1 for features in model.forward_calls)
    assert model.forward_calls[0]["attention_mask"][0, 0] == 0
    assert model.forward_calls[0]["_pairs"][1][1].endswith("source-tail-✓")


def test_exact_limit_and_one_token_overflow(loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder]) -> None:
    adapter, model = loaded
    adapter._max_seq_length = len(model.transformer.tokenizer.row("q", "doc"))
    output = adapter.score_pairs([Item(text="q")], [Item(text="doc")])
    assert output.input_token_counts == [adapter._max_seq_length]
    model.forward_calls.clear()
    with pytest.raises(_NativeInputTooLongError, match="including template"):
        adapter.score_pairs([Item(text="q")], [Item(text="docx")])
    assert not model.forward_calls


def test_large_query_and_small_doc_admit_complete_template_first(
    loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder],
) -> None:
    adapter, model = loaded
    adapter._max_seq_length = 100
    with pytest.raises(_NativeInputTooLongError):
        adapter.score_pairs([Item(text="q" * 80)], [Item(text="d")])
    assert not model.forward_calls
    assert model.transformer.processing_kwargs["text"]["truncation"] is False


def test_any_overlength_pair_prevents_all_forwards_in_its_request(
    loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder],
) -> None:
    adapter, model = loaded
    adapter._max_seq_length = 100
    with pytest.raises(_NativeInputTooLongError):
        adapter.score_pairs([Item(text="q"), Item(text="q")], [Item(text="d"), Item(text="x" * 100)])
    assert not model.forward_calls


@pytest.mark.parametrize("limit", [True, False, None, 0, -1, 1.5, "100", 32769])
def test_runtime_limit_only_reduces_context(
    loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder], limit: Any
) -> None:
    adapter, model = loaded
    with pytest.raises(InvalidInputError, match="max_seq_length"):
        adapter.score_pairs([Item(text="q")], [Item(text="d")], options={"max_seq_length": limit})
    assert not model.preprocess_calls


def test_runtime_limit_does_not_mutate_native_processing(
    loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder],
) -> None:
    adapter, model = loaded
    saved = copy.deepcopy(model.transformer.processing_kwargs)
    count = len(model.transformer.tokenizer.row("q", "d"))
    with pytest.raises(_NativeInputTooLongError):
        adapter.score_pairs([Item(text="q")], [Item(text="d")], options={"max_seq_length": count - 1})
    assert adapter.score_pairs([Item(text="q")], [Item(text="d")]).input_token_counts == [count]
    assert model.transformer.processing_kwargs == saved
    assert model.max_length == 32768


@pytest.mark.parametrize(
    "option", [{"truncation": False}, {"prompt": "x"}, {"batch_size": 100}, {"activation_fn": "Identity"}]
)
def test_unknown_options_are_rejected(
    loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder], option: dict[str, Any]
) -> None:
    adapter, model = loaded
    with pytest.raises(InvalidInputError, match="only the max_seq_length"):
        adapter.score_pairs([Item(text="q")], [Item(text="d")], options=option)
    assert not model.preprocess_calls


@pytest.mark.parametrize("instruction", ["condition", ""])
def test_separate_instruction_is_explicitly_unsupported(
    loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder], instruction: str
) -> None:
    adapter, model = loaded
    with pytest.raises(InvalidInputError, match="conditions belong in the query"):
        adapter.score_pairs([Item(text="q")], [Item(text="d")], instruction=instruction)
    assert not model.preprocess_calls


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("images", []),
        ("images", [{"data": b"x"}]),
        ("audio", AudioInput(data=b"x")),
        ("video", {"data": b"x"}),
        ("document", {"data": b"x"}),
        ("text", None),
        ("text", 1),
    ],
)
@pytest.mark.parametrize("side", ["query", "document"])
def test_non_text_pairs_and_attached_media_are_rejected(
    loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder], field: str, value: Any, side: str
) -> None:
    adapter, model = loaded
    altered = Item(**{"text": "text", field: value})
    query, document = (altered, Item(text="d")) if side == "query" else (Item(text="q"), altered)
    with pytest.raises(InvalidInputError, match="text-only"):
        adapter.score_pairs([query], [document])
    assert not model.preprocess_calls


def test_parallel_cardinality_and_empty_inputs(loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder]) -> None:
    adapter, model = loaded
    with pytest.raises(InvalidInputError, match="equal numbers"):
        adapter.score_pairs([Item(text="q")], [])
    output = adapter.score_pairs([], [])
    assert output.scores.shape == (0,)
    assert output.scores.dtype == np.float32
    assert output.input_token_counts == []
    assert not model.preprocess_calls
    assert not model.forward_calls


@pytest.mark.parametrize(
    ("value", "size"),
    [
        (torch.tensor(5.25), 1),
        (torch.tensor([5.25]), 1),
        (torch.tensor([[5.25]]), 1),
        (torch.tensor([5.25, -3.5]), 2),
        (torch.tensor([[5.25], [-3.5]]), 2),
    ],
)
def test_native_singleton_and_batch_score_shapes(value: torch.Tensor, size: int) -> None:
    result = NativeCausalCrossEncoderAdapter._score_vector(value, size)
    assert result.shape == (size,)
    assert result.dtype == np.float32
    assert result[0] == 5.25


@pytest.mark.parametrize(
    ("value", "size"),
    [
        (torch.tensor(1.0), 2),
        (torch.tensor([1.0]), 2),
        (torch.tensor([1.0, 2.0]), 1),
        (torch.ones(2, 2), 2),
        (torch.ones(2, 1, 1), 2),
        (torch.tensor([1]), 1),
        (torch.tensor([float("nan")]), 1),
        (torch.tensor([float("inf")]), 1),
        (torch.tensor([1e100], dtype=torch.float64), 1),
        (np.array([1.0]), 1),
    ],
)
def test_invalid_cardinality_dimensions_and_nonfinite_scores(value: Any, size: int) -> None:
    with pytest.raises(RuntimeError, match="invalid scores"):
        NativeCausalCrossEncoderAdapter._score_vector(value, size)


def test_bad_native_output_is_internal_error(loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder]) -> None:
    adapter, model = loaded
    model.output = torch.ones(1, 2)
    with pytest.raises(RuntimeError, match="invalid scores") as error:
        adapter.score_pairs([Item(text="q")], [Item(text="d")])
    assert not isinstance(error.value, InvalidInputError)


@pytest.mark.parametrize("mutation", ["ids", "counts"])
def test_admitted_ids_and_counts_must_match_chunk_preprocessing(
    loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder], mutation: str
) -> None:
    adapter, model = loaded
    model.mutate_batch = mutation
    with pytest.raises(RuntimeError, match="changed after admission"):
        adapter.score_pairs([Item(text="q")] * 2, [Item(text="d"), Item(text="longer document")])
    assert not model.forward_calls


@pytest.mark.parametrize(
    "bad_features",
    [
        {},
        {"input_ids": torch.ones(1)},
        {"input_ids": torch.ones(1, 1), "attention_mask": torch.ones(1, 1)},
        {"input_ids": torch.zeros(1, 0, dtype=torch.long), "attention_mask": torch.zeros(1, 0, dtype=torch.long)},
        {"input_ids": torch.ones(1, 3, dtype=torch.long), "attention_mask": torch.tensor([[1, 0, 1]])},
        {"input_ids": torch.ones(1, 3, dtype=torch.long), "attention_mask": torch.tensor([[1, 1, 0]])},
        {"input_ids": torch.ones(1, 1, dtype=torch.long), "attention_mask": torch.tensor([[0]])},
        {"input_ids": torch.ones(1, 1, dtype=torch.long), "attention_mask": torch.tensor([[2]])},
        {"input_ids": torch.ones(1, 1, dtype=torch.long), "attention_mask": torch.tensor([[float("nan")]])},
        {"input_ids": torch.tensor([[-1]]), "attention_mask": torch.tensor([[1]])},
        {"logits_to_keep": 1.0},
        {"logits_to_keep": True},
        {"logits_to_keep": 2},
    ],
)
def test_malformed_native_rows_do_not_forward(
    loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder], bad_features: dict[str, Any]
) -> None:
    adapter, model = loaded
    model.malformed = lambda features: {**features, **bad_features} if bad_features else {}
    with pytest.raises(RuntimeError, match="invalid complete-row features"):
        adapter.score_pairs([Item(text="q")], [Item(text="d")])
    assert not model.forward_calls


def test_padded_budget_uses_rows_times_width_and_restores_original_order(
    loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder],
) -> None:
    adapter, model = loaded
    docs = ["x" * 100, "d", "y" * 50, "dd"]
    model.values = dict(zip(docs, [1.0, -3.5, 8.0, 5.25], strict=True))
    adapter._forward_token_budget = 240
    output = adapter.score_pairs([Item(text="q")] * 4, [Item(text=document) for document in docs])
    np.testing.assert_array_equal(output.scores, [1.0, -3.5, 8.0, 5.25])
    assert output.input_token_counts == [len(model.transformer.tokenizer.row("q", document)) for document in docs]
    assert len(model.forward_calls) == 3
    for features in model.forward_calls:
        assert features["input_ids"].numel() <= adapter._forward_token_budget
        assert features["attention_mask"][:, -1].all()
    assert [document for features in model.forward_calls for _, document in features["_pairs"]] != docs


def test_actual_native_padding_cannot_exceed_private_forward_budget(
    loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder],
) -> None:
    adapter, model = loaded
    adapter._forward_token_budget = 240

    def extra_padding(features: dict[str, Any]) -> dict[str, Any]:
        for key in ("input_ids", "attention_mask"):
            features[key] = torch.nn.functional.pad(features[key], (300, 0))
        return features

    model.malformed = extra_padding
    with pytest.raises(RuntimeError, match="exceeded the padded forward budget"):
        adapter.score_pairs([Item(text="q")], [Item(text="d")])
    assert not model.forward_calls


@pytest.mark.parametrize("change", ["count", "cardinality", "index", "empty_ids", "boolean_index"])
def test_native_rows_alignment_is_checked(change: str) -> None:
    values: dict[str, Any] = {"pairs": (("q", "d"),), "ids": ((1,),), "counts": (1,), "indices": (0,)}
    if change == "count":
        values["counts"] = (2,)
    elif change == "cardinality":
        values["counts"] = ()
    elif change == "index":
        values["indices"] = (1,)
    elif change == "empty_ids":
        values["ids"] = ((),)
    else:
        values["indices"] = (False,)
    with pytest.raises(RuntimeError, match="misaligned"):
        _NativeRows(**values)


def test_score_and_count_hook_use_same_native_path(
    loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder],
) -> None:
    adapter, model = loaded
    query = Item(text="q")
    docs = [Item(text="d"), Item(text="longer")]
    assert adapter.score(query, docs) == [5.25, 5.25]
    forwards = len(model.forward_calls)
    assert adapter.count_pair_input_tokens(query, docs) == [
        len(model.transformer.tokenizer.row("q", item.text)) for item in docs
    ]
    assert len(model.forward_calls) == forwards
    assert adapter.count_pair_input_tokens(query, docs, instruction="unsupported") is None
    adapter._max_seq_length = 1
    assert adapter.count_pair_input_tokens(query, docs) is None
    assert adapter.count_input_tokens([query]) is None


def test_score_handler_preserves_exact_positional_counts() -> None:
    handler = ScoreHandler()
    output = ScoreOutput(scores=np.array([5.25, -3.5], dtype=np.float32), input_token_counts=[73, 100])
    parts = {1: handler.slice_output(output, 1), 0: handler.slice_output(output, 0)}
    restored = handler.assemble_output(parts, 2)
    np.testing.assert_array_equal(restored.scores, output.scores)
    assert restored.input_token_counts == [73, 100]


@pytest.mark.asyncio
async def test_overlength_native_pair_is_isolated_and_maps_to_input_too_long(
    loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder],
) -> None:
    adapter, model = loaded
    adapter._max_seq_length = 100
    loop = asyncio.get_running_loop()
    bad, good = [
        SimpleNamespace(
            future=loop.create_future(),
            _partial_results=None,
            items=[0],
            query=Item(text="q"),
            instruction=None,
            options=None,
        )
        for _ in range(2)
    ]
    group = cast("ConfigGroup", ([Item(text="x" * 100), Item(text="d")], [bad, good], [0, 0], [0, 1]))
    handler = ScoreHandler()
    executor = BatchExecutor(
        model_name="native", registry=None, config=OomRecoveryConfig(enabled=False), stats=OomRecoveryStats()
    )
    seen_sizes: list[int] = []

    async def dispatch(operation: Any, candidate: ConfigGroup) -> ScoreOutput:
        seen_sizes.append(len(candidate[0]))
        return operation.run_inference(adapter, candidate[0], (None, None), None, candidate[1])

    await executor.run(handler, group, dispatch)
    error = bad.future.exception()
    assert isinstance(error, _NativeInputTooLongError)
    assert isinstance(error, InvalidInputError)
    assert isinstance(error, InputTooLongError)
    assert seen_sizes == [2, 1, 1]
    assert good._partial_results[0].input_token_counts == [len(model.transformer.tokenizer.row("q", "d"))]
    assert not good.future.done()
    assert [pair for features in model.forward_calls for pair in features["_pairs"]] == [("q", "d")]
    batch_item = SimpleNamespace(work_item_id="w", request_id="r", item_index=0)
    outcome = _inference_exception_outcome(batch_item, error)
    assert outcome.error_code == "INPUT_TOO_LONG"
    assert outcome.units is None


@pytest.mark.parametrize("value", [True, 0, -1, 32769, 1.5, "32768"])
def test_invalid_constructor_context_is_rejected(value: Any) -> None:
    with pytest.raises(InvalidInputError, match="max_seq_length"):
        NativeCausalCrossEncoderAdapter("publisher/model", max_seq_length=value)


def test_constructor_rejects_imagined_options_and_precision() -> None:
    unknown: dict[str, Any] = {"truncation": False}
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        NativeCausalCrossEncoderAdapter("publisher/model", **unknown)
    with pytest.raises(ValueError, match="compute_precision"):
        NativeCausalCrossEncoderAdapter("publisher/model", compute_precision="auto")  # ty: ignore[invalid-argument-type]


def test_unload_clears_native_state(loaded: tuple[NativeCausalCrossEncoderAdapter, ToyCrossEncoder]) -> None:
    adapter, _ = loaded
    adapter.unload()
    assert all(getattr(adapter, field) is None for field in adapter.spec.unload_fields)
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.score_pairs([], [])
