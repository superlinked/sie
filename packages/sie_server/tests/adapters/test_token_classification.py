"""Offline OntoNotes contracts with constructed tokenizers and synthetic logits."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import msgpack
import msgspec
import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_sdk import SIEClient
from sie_server.adapters.token_classification import adapter as module
from sie_server.adapters.token_classification.adapter import OntoNotesTokenClassificationAdapter, _Span, _Window
from sie_server.api import extract as api
from sie_server.core.extract_cost import build_extract_prepared_items
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.loader import load_model_configs
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import WorkerResult
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.core.worker.types import RequestMetadata
from sie_server.ipc_types import ExtractBatchItem, ItemOutcome
from sie_server.queue_executor import _extract_success_outcome
from sie_server.types.inputs import InvalidInputError, Item
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import RobertaTokenizerFast
from transformers.pipelines.token_classification import AggregationStrategy, TokenClassificationPipeline

TYPES = (
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
)
HEAD = dict(enumerate(["O", *[tag for name in TYPES for tag in (f"B-{name}", f"I-{name}")]]))
LABEL_IDS = {value: key for key, value in HEAD.items()}


class ToyTokenizer:
    """One source character per token, including whitespace; no hidden truncation."""

    def __init__(self):
        self.is_fast = True
        self.model_max_length = 512
        self.add_prefix_space = True
        self.init_kwargs = {"trim_offsets": True}
        self.model_input_names = ["input_ids", "attention_mask"]
        self.pad_token_id = 1
        self.calls = []
        self.pad_rows = False
        self.corrupt = None

    def num_special_tokens_to_add(self, *, pair=False):
        assert pair is False
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [0, *ids, 2]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def __call__(self, text, **kwargs):
        self.calls.append((text, deepcopy(kwargs)))
        ids = [ord(character) + 10 for character in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        if kwargs.get("add_special_tokens") is False:
            assert kwargs["truncation"] is False
            output = {"input_ids": ids, "offset_mapping": offsets}
        else:
            assert kwargs["truncation"] is True
            assert kwargs["max_length"] == 512
            assert kwargs["stride"] == 128
            output = {
                "input_ids": [],
                "attention_mask": [],
                "offset_mapping": [],
                "special_tokens_mask": [],
                "overflow_to_sample_mapping": [],
            }
            start = 0
            while True:
                end = min(start + 510, len(ids))
                row = self.build_inputs_with_special_tokens(ids[start:end])
                attention = [1] * len(row)
                special = [1, *([0] * (end - start)), 1]
                row_offsets = [(0, 0), *offsets[start:end], (0, 0)]
                if self.pad_rows:
                    padding = 512 - len(row)
                    row += [1] * padding
                    attention += [0] * padding
                    special += [1] * padding
                    row_offsets += [(0, 0)] * padding
                output["input_ids"].append(row)
                output["attention_mask"].append(attention)
                output["offset_mapping"].append(row_offsets)
                output["special_tokens_mask"].append(special)
                output["overflow_to_sample_mapping"].append(0)
                if end == len(ids):
                    break
                start = end - 128
        if self.corrupt is not None:
            self.corrupt(output, kwargs)
        return output


class ToyModel:
    def __init__(self):
        self.config = SimpleNamespace(
            model_type="roberta",
            architectures=["RobertaForTokenClassification"],
            num_labels=37,
            id2label=HEAD.copy(),
            label2id=LABEL_IDS.copy(),
            max_position_embeddings=514,
        )
        self.to = Mock()
        self.eval = Mock()
        self.calls = []
        self.labels = {}
        self.rows = []
        self.failure_at = None
        self.bad_logits = None

    def __call__(self, **inputs):
        assert set(inputs) <= {"input_ids", "attention_mask", "token_type_ids"}
        assert not torch.is_grad_enabled()
        index = len(self.calls)
        self.calls.append(inputs)
        if index == self.failure_at:
            raise RuntimeError("synthetic forward failure")
        ids = inputs["input_ids"][0].tolist()
        tags = self.rows[index] if index < len(self.rows) else [self.labels.get(token, "O") for token in ids]
        logits = torch.full((1, len(ids), 37), -5.0)
        for position, tag in enumerate(tags):
            logits[0, position, LABEL_IDS[tag]] = 5.0
        if self.bad_logits is not None:
            logits = self.bad_logits(logits)
        return SimpleNamespace(logits=logits)


@pytest.fixture
def loader(monkeypatch, tmp_path):
    tokenizer, model = ToyTokenizer(), ToyModel()
    snapshot = Mock(return_value=str(tmp_path))
    tokenizer_loader, model_loader = Mock(return_value=tokenizer), Mock(return_value=model)
    monkeypatch.setattr(module, "snapshot_download", snapshot)
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(module.AutoModelForTokenClassification, "from_pretrained", model_loader)
    monkeypatch.setattr(module, "RobertaTokenizerFast", ToyTokenizer)
    return tokenizer, model, snapshot, tokenizer_loader, model_loader


@pytest.fixture
def loaded(loader):
    adapter = OntoNotesTokenClassificationAdapter("publisher/model", revision="a" * 40)
    adapter.load("cpu")
    return adapter, loader[0], loader[1]


def test_pinned_single_snapshot_fast_builtin_fp32_and_lifecycle(loader):
    tokenizer, model, snapshot, tokenizer_loader, model_loader = loader
    adapter = OntoNotesTokenClassificationAdapter("learnrr/roberta-large-ontonotes5-ner", revision="b" * 40)
    adapter.load("cpu")
    snapshot.assert_called_once_with(
        repo_id="learnrr/roberta-large-ontonotes5-ner",
        revision="b" * 40,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ],
    )
    tokenizer_loader.assert_called_once_with(snapshot.return_value, use_fast=True, trust_remote_code=False)
    model_loader.assert_called_once_with(
        snapshot.return_value,
        use_safetensors=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    )
    model.to.assert_called_once_with(device="cpu", dtype=torch.float32)
    model.eval.assert_called_once()
    assert adapter._tokenizer is tokenizer
    assert adapter.capabilities.inputs == ["text"]
    assert adapter.capabilities.outputs == ["json"]
    adapter.unload()
    assert adapter._model is adapter._tokenizer is adapter._device is None


def test_local_snapshot_does_not_resolve_a_second_model(loader, tmp_path):
    adapter = OntoNotesTokenClassificationAdapter(tmp_path)
    adapter.load("cpu")
    loader[2].assert_not_called()
    assert loader[3].call_args.args == loader[4].call_args.args == (str(tmp_path),)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "bert"),
        ("architectures", ["RobertaForSequenceClassification"]),
        ("num_labels", 36),
        ("id2label", {0: "O"}),
        ("label2id", {"O": 0}),
        ("max_position_embeddings", 512),
        ("max_position_embeddings", float("inf")),
    ],
)
def test_incompatible_head_or_context_is_rejected(loader, field, value):
    setattr(loader[1].config, field, value)
    adapter = OntoNotesTokenClassificationAdapter("publisher/model")
    with pytest.raises(RuntimeError, match="37-label"):
        adapter.load("cpu")
    assert adapter._model is None
    loader[1].to.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("is_fast", False),
        ("model_max_length", 513),
        ("model_max_length", 10**30),
        ("add_prefix_space", False),
        ("init_kwargs", {"trim_offsets": False}),
        ("model_input_names", ["input_ids", "offset_mapping"]),
    ],
)
def test_incompatible_tokenizer_is_rejected_before_weights(loader, field, value):
    setattr(loader[0], field, value)
    with pytest.raises(RuntimeError, match="fast RoBERTa"):
        OntoNotesTokenClassificationAdapter("publisher/model").load("cpu")
    loader[4].assert_not_called()


def test_different_fast_tokenizer_class_is_not_admitted(loader):
    loader[3].return_value = SimpleNamespace(is_fast=True)
    with pytest.raises(RuntimeError, match="fast RoBERTa"):
        OntoNotesTokenClassificationAdapter("publisher/model").load("cpu")
    loader[4].assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_seq_length": 511},
        {"max_seq_length": True},
        {"window_overlap": 127},
        {"window_overlap": True},
        {"max_document_tokens": 0},
        {"max_document_tokens": 16385},
        {"max_document_tokens": True},
        {"compute_precision": "float16"},
        {"compute_precision": "bfloat16"},
    ],
)
def test_loadtime_contract_is_explicit(kwargs):
    with pytest.raises(ValueError, match=r"requires|must be"):
        OntoNotesTokenClassificationAdapter("publisher/model", **kwargs)


@pytest.mark.parametrize("labels", [[], ["PERSON"], ["custom"], ["person", "person"], [None], "person"])
def test_labels_are_exact_unique_nonempty_filters(loaded, labels):
    adapter, tokenizer, model = loaded
    with pytest.raises(InvalidInputError, match="subset"):
        adapter.extract([Item(text="Alice")], labels=labels)
    assert tokenizer.calls == model.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"instruction": ""},
        {"instruction": "extract"},
        {"output_schema": {}},
        {"options": {"threshold": 0.85}},
        {"options": {"repair_boundaries": True}},
    ],
)
def test_unsupported_requests_are_not_silently_ignored(loaded, kwargs):
    adapter, tokenizer, model = loaded
    with pytest.raises(InvalidInputError, match="only labels"):
        adapter.extract([Item(text="Alice")], **kwargs)
    assert tokenizer.calls == model.calls == []


def _window(tags, offsets=None):
    size = len(tags)
    offsets = offsets if offsets is not None else [(index, index + 1) for index in range(size)]
    logits = torch.full((1, size + 2, 37), -1.0)
    for index, tag in enumerate(["O", *tags, "O"]):
        logits[0, index, LABEL_IDS[tag]] = 1.0 + index / 100
    return logits, _Window(
        inputs={"input_ids": [0, *range(10, size + 10), 2], "attention_mask": [1] * (size + 2)},
        offsets=[(0, 0), *offsets, (0, 0)],
        special=[1, *([0] * size), 1],
        token_count=size + 2,
    )


@pytest.mark.parametrize(
    "tags",
    [
        ["B-PERSON", "I-PERSON", "O", "B-ORG"],
        ["B-PERSON", "B-PERSON", "I-PERSON"],
        ["I-PERSON", "I-PERSON", "O", "I-ORG"],
        ["B-GPE", "I-GPE", "B-FAC", "I-LOC"],
        ["B-PRODUCT", "I-PRODUCT", "B-NORP", "O"],
        ["O", "O"],
    ],
)
def test_local_simple_bio_matches_pinned_transformers_grouping(tags):
    logits, window = _window(tags)
    spans = OntoNotesTokenClassificationAdapter._decode(logits, window)
    pipeline = object.__new__(TokenClassificationPipeline)
    pipeline.model = SimpleNamespace(config=SimpleNamespace(id2label=HEAD))
    pipeline.tokenizer = ToyTokenizer()
    scores = logits[0].softmax(-1).numpy()
    entities = pipeline.aggregate(
        [
            {"scores": scores[index + 1], "index": index + 1, "word": str(index), "start": index, "end": index + 1}
            for index in range(len(tags))
        ],
        AggregationStrategy.SIMPLE,
    )
    expected = [entity for entity in entities if entity["entity_group"] != "O"]
    assert [(s.start, s.end, s.tag) for s in spans] == [
        (entity["start"], entity["end"], entity["entity_group"]) for entity in expected
    ]
    assert [s.score for s in spans] == pytest.approx([entity["score"] for entity in expected])


def test_no_retained_type_renormalization_or_confidence_cutoff(loaded):
    adapter, _, model = loaded

    def logits(values):
        values[:] = 0
        values[:, :, 1] = 0.01
        return values

    model.bad_logits = logits
    result = adapter.extract([Item(text="A")])
    assert result.errors is None
    assert result.entities[0][0]["label"] == "person"
    assert 0.02 < result.entities[0][0]["score"] < 0.04


def test_ontology_projection_happens_after_grouping_and_labels_filter(loaded):
    adapter, _, model = loaded
    model.rows = [["O", "B-GPE", "I-GPE", "B-LOC", "B-FAC", "B-PRODUCT", "B-PERSON", "B-ORG", "O"]]
    result = adapter.extract([Item(text="abcdefg")])
    assert [(e["text"], e["label"]) for e in result.entities[0]] == [
        ("ab", "location"),
        ("c", "location"),
        ("d", "location"),
        ("f", "person"),
        ("g", "organization"),
    ]
    model.labels = {ord("a") + 10: "B-PERSON", ord("b") + 10: "B-ORG"}
    result = adapter.extract([Item(text="ab")], labels=["organization"])
    assert [(e["text"], e["label"]) for e in result.entities[0]] == [("b", "organization")]


def test_literal_unicode_subword_repeated_and_punctuation_offsets(loaded):
    adapter, tokenizer, model = loaded
    text = "  The Ada's, 🐍Ada."
    tags = ["O"] * len(text)
    tags[6:11] = ["B-PERSON", *(["I-PERSON"] * 4)]
    tags[14:17] = ["B-PERSON", "I-PERSON", "I-PERSON"]
    model.rows = [["O", *tags, "O"]]
    result = adapter.extract([Item(text=text)])
    assert [(e["text"], e["start"], e["end"]) for e in result.entities[0]] == [("Ada's", 6, 11), ("Ada", 14, 17)]
    assert all(call[0] == text for call in tokenizer.calls)
    assert result.input_token_counts == [len(text) + 2]


def test_zero_width_tokens_are_skipped_without_word_repair():
    logits, window = _window(["B-PERSON", "B-ORG", "I-PERSON"], [(0, 1), (1, 1), (1, 2)])
    spans = OntoNotesTokenClassificationAdapter._decode(logits, window)
    assert [(s.start, s.end, s.tag) for s in spans] == [(0, 2, "PERSON")]


def test_last_window_entity_full_coverage_and_repeated_work_counts(loaded):
    adapter, tokenizer, model = loaded
    text = "x" * 650 + "Y" + "x" * 49
    model.labels = {ord("Y") + 10: "B-PERSON"}
    result = adapter.extract([Item(text=text)], prepared_items=[SimpleNamespace(input_ids=[10])])
    assert [(e["text"], e["start"], e["end"]) for e in result.entities[0]] == [("Y", 650, 651)]
    assert [len(call["input_ids"][0]) for call in model.calls] == [512, 320]
    assert result.input_token_counts == [700 + 128 + 4]
    assert len(tokenizer.calls) == 2
    assert adapter.count_input_tokens([Item(text=text)]) is None


def test_padding_is_not_forwarded_token_usage(loaded):
    adapter, tokenizer, model = loaded
    tokenizer.pad_rows = True
    result = adapter.extract([Item(text="Alice")])
    assert result.errors is None
    assert model.calls[0]["input_ids"].shape == (1, 512)
    assert result.input_token_counts == [7]


@pytest.mark.parametrize(
    "corruption", ["first_only", "offset", "id", "overflow_owner", "extra_row", "full_offset", "mask", "missing_input"]
)
def test_malformed_or_incomplete_coverage_fails_before_any_forward(loaded, corruption):
    adapter, tokenizer, model = loaded

    def corrupt(output, kwargs):
        if kwargs.get("add_special_tokens") is False:
            if corruption == "full_offset":
                output["offset_mapping"][-1] = (99999, 100000)
            return
        if corruption == "first_only":
            for name in output:
                output[name] = output[name][:1]
        elif corruption == "offset":
            output["offset_mapping"][1][1] = (0, 1)
        elif corruption == "id":
            output["input_ids"][1][1] += 1
        elif corruption == "overflow_owner":
            output["overflow_to_sample_mapping"][1] = 1
        elif corruption == "extra_row":
            for rows in output.values():
                rows.append(deepcopy(rows[-1]))
        elif corruption == "mask":
            output["special_tokens_mask"][0][1] = 1
        elif corruption == "missing_input":
            output.pop("attention_mask")

    tokenizer.corrupt = corrupt
    result = adapter.extract([Item(text="x" * 700)])
    assert result.errors[0].code == "INFERENCE_ERROR"
    assert result.input_token_counts == [0]
    assert result.entities == [[]]
    assert model.calls == []


def test_document_cap_rejects_complete_item_without_overflow_tokenization(loader):
    adapter = OntoNotesTokenClassificationAdapter("publisher/model", max_document_tokens=16)
    adapter.load("cpu")
    result = adapter.extract([Item(text="x" * 17), Item(text="x" * 16)])
    assert result.errors[0].code == "INPUT_TOO_LONG"
    assert result.errors[1] is None
    assert result.input_token_counts == [0, 18]
    assert len(loader[0].calls) == 3
    assert len(loader[1].calls) == 1


@pytest.mark.parametrize("bad", ["shape", "labels", "nan", "inf", "integer"])
def test_malformed_model_output_is_not_a_successful_partial_item(loaded, bad):
    adapter, _, model = loaded
    model.bad_logits = {
        "shape": lambda logits: logits[0],
        "labels": lambda logits: logits[:, :, :36],
        "nan": lambda logits: logits * float("nan"),
        "inf": lambda logits: logits + float("inf"),
        "integer": lambda logits: logits.to(torch.long),
    }[bad]
    result = adapter.extract([Item(text="Alice")])
    assert result.entities == [[]]
    assert result.errors[0].code == "INFERENCE_ERROR"
    assert result.input_token_counts == [7]


def test_later_forward_failure_has_no_prefix_success_and_preserves_usage(loaded):
    adapter, _, model = loaded
    model.labels = {ord("A") + 10: "B-PERSON"}
    model.failure_at = 1
    result = adapter.extract([Item(text="A" + "x" * 699), Item(text="A")])
    assert result.entities[0] == []
    assert result.errors[0].code == "INFERENCE_ERROR"
    assert result.errors[1] is None
    assert result.input_token_counts == [832, 3]
    assert result.entities[1][0]["text"] == "A"


def test_mixed_blank_missing_and_valid_positions(loaded):
    adapter, tokenizer, model = loaded
    model.labels = {ord("A") + 10: "B-PERSON"}
    result = adapter.extract([Item(), Item(text="A"), Item(text=" \t"), Item(text="A")])
    assert [error.code if error else None for error in result.errors] == ["INVALID_INPUT", None, "INVALID_INPUT", None]
    assert result.input_token_counts == [0, 3, 0, 3]
    assert [bool(entities) for entities in result.entities] == [False, True, False, True]
    assert len(tokenizer.calls) == 4


def test_empty_batch_preprocessor_estimation_and_warmup(loaded):
    adapter, _, model = loaded
    assert adapter.extract([]) == ExtractOutput(entities=[], input_token_counts=[])
    assert isinstance(adapter.get_preprocessor(), CharCountPreprocessor)
    adapter.warmup()
    assert model.calls == []
    assert adapter.extract([Item(text="first caller")]).errors is None
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    "spans",
    [
        [_Span(0, 3, "PERSON", 0.9), _Span(1, 5, "ORG", 0.8)],
        [_Span(0, 3, "PERSON", 0.7), _Span(0, 3, "ORG", 0.9)],
        [_Span(0, 3, "PERSON", 0.9), _Span(0, 3, "ORG", 0.9)],
        [_Span(0, 3, "PRODUCT", 0.9), _Span(1, 3, "PERSON", 1), _Span(3, 5, "PERSON", 0.5)],
        [_Span(1, 4, "PERSON", 0.9), _Span(0, 5, "ORG", 0.6)],
        [],
    ],
)
def test_window_overlap_matches_pinned_library_length_score_tie_policy(spans):
    pipeline = object.__new__(TokenClassificationPipeline)
    expected = pipeline.aggregate_overlapping_entities(
        [{"start": span.start, "end": span.end, "entity_group": span.tag, "score": span.score} for span in spans]
    )
    assert OntoNotesTokenClassificationAdapter._overlaps(spans) == [
        _Span(entity["start"], entity["end"], entity["entity_group"], entity["score"]) for entity in expected
    ]


def test_real_fast_bytelevel_offsets_and_overflows_without_pretrained_assets(loader, monkeypatch):
    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        **{character: index + 4 for index, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))},
    }
    unknown = "<unk>"
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=unknown))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    backend.post_processor = processors.RobertaProcessing(
        sep=("</s>", 2),
        cls=("<s>", 0),
        trim_offsets=True,
        add_prefix_space=True,
    )
    tokenizer = RobertaTokenizerFast(
        tokenizer_object=backend,
        add_prefix_space=True,
        trim_offsets=True,
        model_max_length=512,
    )
    loader[3].return_value = tokenizer
    monkeypatch.setattr(module, "RobertaTokenizerFast", RobertaTokenizerFast)
    adapter = OntoNotesTokenClassificationAdapter("publisher/model")
    adapter.load("cpu")
    text = "  Ada 🐍 café. " * 75
    windows = adapter._windows(text)
    assert len(windows) > 1
    assert sum(w.token_count for w in windows) > len(tokenizer(text, add_special_tokens=False)["input_ids"])
    assert all(0 <= start <= end <= len(text) for window in windows for start, end in window.offsets)
    result = adapter.extract([Item(text=text)])
    assert result.errors is None
    assert result.input_token_counts == [sum(window.token_count for window in windows)]


@pytest.mark.parametrize("profile_options", [{}, {"profile": "default"}])
@pytest.mark.parametrize("public_labels", [None, [], ["person", "organization", "location"], ["person"]])
@pytest.mark.parametrize("single_item", [False, True])
def test_handler_http_ipc_and_public_sdk_transport_keep_positions_offsets_and_usage(
    loaded, monkeypatch, profile_options, public_labels, single_item
):
    adapter, _, model = loaded
    model.labels = {ord("A") + 10: "B-PERSON", ord("B") + 10: "B-ORG", ord("C") + 10: "B-GPE"}
    expected_labels = public_labels or ["person", "organization", "location"]
    expected_counts = [5] if single_item else [5, 0]
    handler = ExtractHandler()
    physical = []
    raw = []
    ipc = []
    model_name = "learnrr/roberta-large-ontonotes5-ner"
    server = Path(__file__).resolve().parents[2]
    registry = Mock(spec=ModelRegistry)
    registry.has_model.return_value = True
    registry.get_config.return_value = load_model_configs(server / "models")[model_name]

    async def route(*args, **kwargs):
        return SimpleNamespace(key=model_name, headers=dict)

    async def worker(_registry, _model, items, *, labels, output_schema, instruction, options):
        assert labels == public_labels
        # The public profile selector is consumed by the existing API boundary.
        assert options == {}
        assert output_schema is instruction is None
        timing = RequestTiming()
        metadata = RequestMetadata(
            future=asyncio.get_running_loop().create_future(),
            items=items,
            timing=timing,
            operation="extract",
            labels=labels,
            options=options,
        )
        output = handler.run_inference(
            adapter,
            items,
            handler.make_config_key(metadata),
            build_extract_prepared_items(items),
            [metadata],
        )
        partials = {i: handler.slice_output(output, i) for i in range(len(items))}
        output = handler.assemble_output(partials, len(items))
        assert output.input_token_counts == expected_counts
        for index, item in enumerate(items):
            batch_item = ExtractBatchItem(
                work_item_id=f"synthetic-work-{index}",
                request_id="synthetic-request",
                item_index=index,
                total_items=len(items),
                timestamp=0,
                item={"text": item.text},
                labels=labels,
            )
            outcome = _extract_success_outcome(
                adapter,
                batch_item,
                item,
                WorkerResult(output=partials[index], timing=timing),
            )
            ipc.append(msgspec.msgpack.decode(msgspec.msgpack.encode(outcome), type=ItemOutcome))
        return WorkerResult(output=output, timing=timing)

    monkeypatch.setattr(api, "route_request", route)
    monkeypatch.setattr(api, "_extract_via_worker", worker)
    app = FastAPI()
    app.state.registry = registry
    app.include_router(api.router)

    def transport(request):
        physical.append(request)
        body = msgpack.unpackb(request.content, raw=False)
        expected_params = {"options": profile_options}
        if public_labels is not None:
            expected_params["labels"] = public_labels
        assert body["params"] == expected_params
        assert request.url == "https://inference.example/gateway/v1/extract/learnrr/roberta-large-ontonotes5-ner"
        with TestClient(app) as test_client:
            response = test_client.post(
                request.url.path.removeprefix("/gateway"),
                content=request.content,
                headers={"Content-Type": "application/msgpack", "Accept": "application/msgpack"},
            )
        assert response.status_code == 200, response.text
        raw.append(msgpack.unpackb(response.content, raw=False))
        return httpx.Response(response.status_code, content=response.content, headers=response.headers)

    with httpx.Client(base_url="https://inference.example/gateway", transport=httpx.MockTransport(transport)) as http:
        with SIEClient("https://inference.example/gateway", api_key=None, http_client=http) as client:
            results = client.extract(
                "learnrr/roberta-large-ontonotes5-ner",
                {"text": "ABC"} if single_item else [{"text": "ABC"}, {"text": " "}],
                labels=public_labels,
                options=profile_options,
            )
    assert len(physical) == 1
    assert isinstance(results, dict if single_item else list)
    results_list = [results] if single_item else results
    assert len(raw[0]["items"]) == len(results_list) == len(expected_counts)
    assert [entity["label"] for entity in results_list[0]["entities"]] == expected_labels
    assert [entity["text"] for entity in results_list[0]["entities"]] == (
        ["A"] if public_labels == ["person"] else ["A", "B", "C"]
    )
    assert results_list[0]["entities"][0]["start"] == 0
    assert results_list[0]["entities"][0]["end"] == 1
    if not single_item:
        assert results_list[1]["error"]["code"] == "INVALID_INPUT"
    assert raw[0]["usage"]["input_tokens"] == 5
    assert [outcome.units.input_tokens for outcome in ipc] == expected_counts
    assert msgpack.unpackb(ipc[0].result_msgpack, raw=False)["entities"][0]["text"] == "A"
    if not single_item:
        assert msgpack.unpackb(ipc[1].result_msgpack, raw=False)["error"]["code"] == "INVALID_INPUT"
