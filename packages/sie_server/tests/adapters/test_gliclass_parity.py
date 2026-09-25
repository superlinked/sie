"""GLiClass scores against the gliclass pipeline, on a tiny random model (no downloads).

Before scoring moved into the adapter, a labels request ran exactly
``ZeroShotClassificationPipeline(texts, labels, threshold=0.0,
return_hierarchical=True, prompt=..., examples=...)``. The adapter must still
return the scores that call returns, bit for bit. Separate label groups must
score like one labels request per group.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch
from gliclass import GLiClassModel, GLiClassModelConfig, ZeroShotClassificationPipeline
from sie_server.adapters.gliclass import GLiClassAdapter
from sie_server.types.inputs import InvalidInputError, Item
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import DebertaV2Config, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

_MAX_LENGTH = 64
_TEXTS = [f"the app crashes on startup {'when config file is missing ' * i}".strip() for i in range(11)]
_LABELS = ["billing", "bug report", "feature request"]
_GROUPS = {"topic": _LABELS, "urgency": ["low", "high"], "remote": ["yes", "no", "can exploit it over network"]}
_EXAMPLE_TEXTS = ["charged twice for one order", "please add dark mode"]
_INSTRUCTION = "classify the ticket"
# Every word the tests send, so no word falls back to [UNK].
_WORDS = sorted(
    {
        word
        for text in [
            *_TEXTS,
            *_EXAMPLE_TEXTS,
            _INSTRUCTION,
            *(label for labels in _GROUPS.values() for label in labels),
        ]
        for word in text.split()
    }
)


def _tokenizer() -> PreTrainedTokenizerFast:
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    vocab = {token: index for index, token in enumerate([*specials, *_WORDS, ".", ",", ":"])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))  # noqa: S106 -- a vocabulary entry, not a secret
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",  # noqa: S106 -- vocabulary entries, not secrets
        unk_token="[UNK]",  # noqa: S106
        cls_token="[CLS]",  # noqa: S106
        sep_token="[SEP]",  # noqa: S106
    )
    tokenizer.add_tokens(["<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>"], special_tokens=True)
    tokenizer.model_max_length = _MAX_LENGTH
    return tokenizer


def _model(tokenizer: PreTrainedTokenizerFast, *, prompt_first: bool, scorer: str) -> GLiClassModel:
    torch.manual_seed(0)
    encoder = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        relative_attention=True,
        position_buckets=16,
        pos_att_type=["p2c", "c2p"],
        position_biased_input=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    config = GLiClassModelConfig(
        encoder_config=encoder,
        encoder_model="tiny-gliclass-encoder-not-on-disk",
        class_token_index=tokenizer.convert_tokens_to_ids("<<LABEL>>"),
        text_token_index=tokenizer.convert_tokens_to_ids("<<SEP>>"),
        example_token_index=tokenizer.convert_tokens_to_ids("<<EXAMPLE>>"),
        vocab_size=len(tokenizer),
        scorer_type=scorer,
        scorer_mlp_hidden_size=32,
        prompt_first=prompt_first,
        dropout=0.0,
    )
    return GLiClassModel(config).eval()


class _Rig:
    """A tiny GLiClass model behind both the gliclass pipelines and the adapter."""

    def __init__(
        self,
        *,
        prompt_first: bool,
        scorer: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        cuda_graphs: str = "off",
    ) -> None:
        tokenizer = _tokenizer()
        model = _model(tokenizer, prompt_first=prompt_first, scorer=scorer).to(device, dtype=dtype)
        self.pipelines = {
            classification_type: ZeroShotClassificationPipeline(
                model,
                tokenizer,
                max_length=_MAX_LENGTH,
                classification_type=classification_type,
                device=device,
                progress_bar=False,
            )
            for classification_type in ("single-label", "multi-label")
        }
        self.adapter = GLiClassAdapter("tiny", max_seq_length=_MAX_LENGTH, cuda_graphs=cuda_graphs)
        self.adapter._attach(self.pipelines["single-label"].pipe, tokenizer)

    def pipeline_scores(
        self, texts: list[str], labels: list[str], classification_type: str, **context: Any
    ) -> list[dict[str, float]]:
        """The pre-change adapter's call, verbatim."""
        with torch.inference_mode():
            return self.pipelines[classification_type](
                texts, labels, threshold=0.0, return_hierarchical=True, **context
            )


@pytest.fixture(scope="module", params=[(False, "simple"), (True, "mlp")], ids=["text-first", "labels-first"])
def rig(request: pytest.FixtureRequest) -> _Rig:
    prompt_first, scorer = request.param
    return _Rig(prompt_first=prompt_first, scorer=scorer)


def _scores(output: Any) -> list[dict[str, float]]:
    return [{c["label"]: c["score"] for c in row} for row in output.classifications]


@pytest.mark.parametrize("classification_type", ["single-label", "multi-label"])
@pytest.mark.parametrize(
    "context",
    [
        {},
        {"prompt": _INSTRUCTION},
        {"examples": [{"text": "charged twice for one order", "labels": ["billing"]}]},
    ],
    ids=["plain", "instruction", "examples"],
)
def test_labels_requests_score_bit_for_bit_like_the_pipeline(
    rig: _Rig, classification_type: str, context: dict[str, Any]
) -> None:
    # Eleven texts of different lengths: two padded sub-batches of eight and three.
    options: dict[str, Any] = {"classification_type": classification_type}
    if "examples" in context:
        options["examples"] = context["examples"]

    output = rig.adapter.extract(
        [Item(text=text) for text in _TEXTS], labels=_LABELS, instruction=context.get("prompt"), options=options
    )

    assert _scores(output) == rig.pipeline_scores(_TEXTS, _LABELS, classification_type, **context)


@pytest.mark.parametrize("classification_type", ["single-label", "multi-label"])
def test_separate_groups_score_like_one_labels_request_per_group(rig: _Rig, classification_type: str) -> None:
    items = [Item(text=text) for text in _TEXTS[:4]]
    options = {"classification_type": classification_type}

    grouped = rig.adapter.extract(items, options={**options, "label_groups": _GROUPS})

    assert grouped.data is not None
    usage = [0] * len(items)
    for group, labels in _GROUPS.items():
        alone = rig.adapter.extract(items, labels=labels, options=options)
        assert alone.input_token_counts is not None
        for index, expected in enumerate(_scores(alone)):
            got = grouped.data[index][group]["probabilities"]
            assert got == pytest.approx(expected, abs=1e-6)
            if classification_type == "single-label":
                assert grouped.data[index][group]["choice"] == max(expected, key=expected.__getitem__)
            usage[index] += alone.input_token_counts[index]
    # Every group encodes the document again, and each row is metered.
    assert grouped.input_token_counts == usage


def test_separate_groups_with_examples_score_like_per_group_requests(rig: _Rig) -> None:
    items = [Item(text=_TEXTS[2])]
    examples = [
        {"text": "charged twice for one order", "labels": {"topic": "billing", "urgency": "high"}},
        {"text": "please add dark mode", "labels": ["topic.feature request"]},
    ]

    grouped = rig.adapter.extract(
        items, instruction=_INSTRUCTION, options={"label_groups": _GROUPS, "examples": examples}
    )

    assert grouped.data is not None
    per_group_examples = {
        "topic": [
            {"text": "charged twice for one order", "labels": ["billing"]},
            {"text": "please add dark mode", "labels": ["feature request"]},
        ],
        "urgency": [
            {"text": "charged twice for one order", "labels": ["high"]},
            {"text": "please add dark mode", "labels": []},
        ],
        "remote": [
            {"text": "charged twice for one order", "labels": []},
            {"text": "please add dark mode", "labels": []},
        ],
    }
    for group, labels in _GROUPS.items():
        expected = rig.pipeline_scores(
            [_TEXTS[2]], labels, "single-label", prompt=_INSTRUCTION, examples=per_group_examples[group]
        )[0]
        assert grouped.data[0][group]["probabilities"] == pytest.approx(expected, abs=1e-6)


def test_joint_groups_are_unchanged_by_the_separate_default(rig: _Rig) -> None:
    items = [Item(text=text) for text in _TEXTS[:3]]

    joint = rig.adapter.extract(items, options={"label_groups": _GROUPS, "group_encoding": "joint"})
    separate = rig.adapter.extract(items, options={"label_groups": _GROUPS})

    assert joint.data is not None
    assert separate.data is not None
    for joint_answers, separate_answers in zip(joint.data, separate.data, strict=True):
        assert list(joint_answers) == list(separate_answers) == list(_GROUPS)
        for answer in joint_answers.values():
            assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    # The joint row encodes the document once.
    assert joint.input_token_counts is not None
    assert separate.input_token_counts is not None
    assert [count * len(_GROUPS) for count in joint.input_token_counts] == separate.input_token_counts


# Far longer than the model reads: its window holds about 60 of these words.
_LONG_DOCUMENT = " ".join(f"the app crashes on startup when config file {index} is missing" for index in range(2_000))


def _outputs(output: Any) -> dict[str, Any]:
    errors = None if output.errors is None else [None if e is None else e.code for e in output.errors]
    return {
        "classifications": output.classifications,
        "data": output.data,
        "errors": errors,
        "usage": output.input_token_counts,
    }


@pytest.mark.parametrize("policy", ["default", "truncate_text"])
@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"labels": _LABELS},
        {"labels": _LABELS, "instruction": _INSTRUCTION, "options": {"classification_type": "multi-label"}},
        {"options": {"label_groups": _GROUPS, "group_encoding": "joint"}},
        {"options": {"label_groups": _GROUPS}},
    ],
    ids=["labels", "labels-context", "joint", "separate"],
)
def test_cutting_a_long_document_to_what_the_model_reads_changes_nothing(
    rig: _Rig, policy: str, request_kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [Item(text=_LONG_DOCUMENT), Item(text=_TEXTS[3])]
    kwargs = {**request_kwargs, "options": {**request_kwargs.get("options", {}), "overflow_policy": policy}}

    cut = rig.adapter.extract(items, **kwargs)
    monkeypatch.setattr(rig.adapter, "_visible_tokens", lambda: None)
    whole = rig.adapter.extract(items, **kwargs)

    assert _outputs(cut) == _outputs(whole)
    if "labels" in request_kwargs and policy == "default" and rig.adapter._pipe.model.config.prompt_first:
        # The pipeline reads the whole document.
        expected = rig.pipeline_scores(
            [item.text for item in items],
            _LABELS,
            request_kwargs.get("options", {}).get("classification_type", "single-label"),
            **({"prompt": _INSTRUCTION} if "instruction" in request_kwargs else {}),
        )
        assert _scores(cut) == expected


def _counting_tokenizer_chars(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Characters of every string handed to any tokenizer from now on."""
    seen: list[int] = []
    original = PreTrainedTokenizerBase.__call__

    def counting(self: Any, text: Any = None, *args: Any, **kwargs: Any) -> Any:
        texts = text if isinstance(text, list) else [text]
        seen.append(sum(len(t) for t in texts if isinstance(t, str)))
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(PreTrainedTokenizerBase, "__call__", counting)
    return seen


_MANY_GROUPS = {f"q{index}": ["yes", "no"] for index in range(64)}


@pytest.mark.parametrize("policy", ["default", "truncate_text"])
@pytest.mark.parametrize("size", [10_000, 100_000, 1_000_000, 4_000_000])
def test_characters_tokenized_do_not_grow_with_the_document(
    rig: _Rig, size: int, policy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = (_LONG_DOCUMENT * (size // len(_LONG_DOCUMENT) + 1))[:size]
    seen = _counting_tokenizer_chars(monkeypatch)

    output = rig.adapter.extract(
        [Item(text=document)], options={"label_groups": _MANY_GROUPS, "overflow_policy": policy}
    )

    assert output.data is not None
    # The model reads at most ``window`` document tokens. The document is cut
    # to a prefix of about 8 characters per token it can read, tokenized with
    # twice that once to check the cut, and then once per group.
    window = _MAX_LENGTH - 2
    bound = (len(_MANY_GROUPS) + 3) * (window + 8) * 8 * 2
    assert sum(seen) < bound


def test_a_multi_megabyte_document_with_64_groups_stays_fast(rig: _Rig) -> None:
    document = (_LONG_DOCUMENT * 400)[: 4 * 1024 * 1024]

    start = time.perf_counter()
    output = rig.adapter.extract([Item(text=document)], options={"label_groups": _MANY_GROUPS})
    elapsed = time.perf_counter() - start

    assert output.data is not None
    # Tokenizing the whole document once per group took minutes.
    assert elapsed < 10.0


@pytest.mark.parametrize("document", ["whitespace", "one-word"])
@pytest.mark.parametrize("policy", ["default", "truncate_text"])
def test_a_document_that_cannot_be_cut_costs_what_reading_it_whole_does(
    rig: _Rig, document: str, policy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Neither has a word boundary where the model's window ends: 1 MB of
    # spaces (no tokens) before three words, and one 1 MB word.
    text = " " * 1_000_000 + "the app crashes" if document == "whitespace" else "a" * 1_000_000
    items = [Item(text=text)]
    options = {"overflow_policy": policy}
    seen = _counting_tokenizer_chars(monkeypatch)

    searched = rig.adapter.extract(items, labels=_LABELS, options=options)
    with_search = sum(seen)
    seen.clear()
    monkeypatch.setattr(rig.adapter, "_visible_tokens", lambda: None)
    whole = rig.adapter.extract(items, labels=_LABELS, options=options)

    assert _outputs(searched) == _outputs(whole)
    # The search for a cut stops at 64 characters per readable token, and the
    # document is then tokenized once for every check and for metering.
    search_bound = 2 * 64 * (_MAX_LENGTH - 2 + 8)
    assert with_search <= sum(seen) + search_bound


_GRAPH_REQUESTS = [
    {"labels": _LABELS},
    {"labels": _LABELS, "instruction": _INSTRUCTION, "options": {"classification_type": "multi-label"}},
    {"options": {"label_groups": _GROUPS, "group_encoding": "joint"}},
    {"options": {"label_groups": _GROUPS}},
]
_GRAPH_REQUEST_IDS = ["labels", "labels-context", "joint", "separate"]


_LAYOUTS = {"params": [(False, "simple"), (True, "mlp")], "ids": ["text-first", "labels-first"]}


def _cuda_rig(request: pytest.FixtureRequest, mode: str) -> _Rig:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    prompt_first, scorer = request.param
    return _Rig(prompt_first=prompt_first, scorer=scorer, device="cuda:0", dtype=torch.float16, cuda_graphs=mode)


@pytest.fixture(scope="module", **_LAYOUTS)
def exact_rig(request: pytest.FixtureRequest) -> _Rig:
    return _cuda_rig(request, "exact")


@pytest.fixture(scope="module", **_LAYOUTS)
def bucketed_rig(request: pytest.FixtureRequest) -> _Rig:
    return _cuda_rig(request, "bucketed")


def _fresh_runner(rig: _Rig, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The rig's graph runner, emptied, with a full recording burst, counting replays."""
    runner = rig.adapter._graphs
    assert runner is not None
    runner.clear()
    runner._recording_credit = 16.0
    runner.replayed = []
    replay = runner._replay

    def counted(entry: Any, inputs: dict[str, torch.Tensor], length: int) -> torch.Tensor:
        runner.replayed.append(length)
        return replay(entry, inputs, length)

    monkeypatch.setattr(runner, "_replay", counted)
    return runner


_EAGER = {"cuda_graphs": "off"}


def _eager_request(request_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {**request_kwargs, "options": {**request_kwargs.get("options", {}), **_EAGER}}


@pytest.mark.gpu_hw
@pytest.mark.parametrize("request_kwargs", _GRAPH_REQUESTS, ids=_GRAPH_REQUEST_IDS)
def test_exact_cuda_graphs_score_bit_for_bit_like_eager(
    exact_rig: _Rig, request_kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Eleven texts of different lengths, sent three times: a shape is recorded
    # the second time it is seen and replayed the third.
    items = [Item(text=text) for text in _TEXTS]
    runner = _fresh_runner(exact_rig, monkeypatch)

    eager = _outputs(exact_rig.adapter.extract(items, **_eager_request(request_kwargs)))
    for _ in range(3):
        assert _outputs(exact_rig.adapter.extract(items, **request_kwargs)) == eager

    assert runner.graph_count > 0
    assert runner.replayed  # the third pass replayed
    assert not runner.disabled
    # Graphs of one length share its relative-position table.
    assert set(runner._relative_pos) == {length for _, length, _ in runner._graphs}


@pytest.mark.gpu_hw
@pytest.mark.parametrize("request_kwargs", _GRAPH_REQUESTS, ids=_GRAPH_REQUEST_IDS)
def test_bucketed_cuda_graphs_stay_close_to_eager(
    bucketed_rig: _Rig, request_kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [Item(text=text) for text in _TEXTS]
    runner = _fresh_runner(bucketed_rig, monkeypatch)

    eager = bucketed_rig.adapter.extract(items, **_eager_request(request_kwargs))
    for _ in range(2):
        graphed = bucketed_rig.adapter.extract(items, **request_kwargs)
        if eager.data:
            pairs = [
                (a["probabilities"], b["probabilities"])
                for row_a, row_b in zip(eager.data, graphed.data or [], strict=True)
                for a, b in zip(row_a.values(), row_b.values(), strict=True)
            ]
        else:
            pairs = list(zip(_scores(eager), _scores(graphed), strict=True))
        for expected, got in pairs:
            assert got == pytest.approx(expected, abs=2e-2)
        assert graphed.input_token_counts == eager.input_token_counts
    assert runner.graph_count > 0
    assert runner.replayed  # the second pass replayed
    assert not runner.disabled


@pytest.mark.gpu_hw
def test_graphs_over_their_memory_budget_are_dropped(exact_rig: _Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    items = [Item(text=text) for text in _TEXTS]
    runner = _fresh_runner(exact_rig, monkeypatch)
    monkeypatch.setattr(runner, "_memory_budget", lambda device: 0)  # every recording goes over

    eager = _outputs(exact_rig.adapter.extract(items, labels=_LABELS, options=_EAGER))
    torch.cuda.synchronize()
    reserved = torch.cuda.memory_reserved()
    for _ in range(3):
        assert _outputs(exact_rig.adapter.extract(items, labels=_LABELS)) == eager

    assert runner.graph_count == 0
    assert not runner.disabled
    assert runner._device_bytes == 0
    # The dropped graphs' pool went back to the device.
    assert torch.cuda.memory_reserved() <= reserved


class _OutOfMemoryGraph:
    """A CUDA graph whose recording runs out of memory."""

    def capture_begin(self, **_: Any) -> None:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory while recording")

    def capture_end(self) -> None:
        pass


@pytest.mark.gpu_hw
def test_running_out_of_memory_while_recording_answers_eagerly(
    exact_rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [Item(text=text) for text in _TEXTS]
    runner = _fresh_runner(exact_rig, monkeypatch)
    eager = _outputs(exact_rig.adapter.extract(items, labels=_LABELS, options=_EAGER))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _OutOfMemoryGraph)

    # The second sighting records; the recording fails, and the call still gets its answer.
    for _ in range(2):
        assert _outputs(exact_rig.adapter.extract(items, labels=_LABELS)) == eager

    assert runner.graph_count == 0
    assert not runner.disabled
    assert runner._recording_credit < 1  # paused


@pytest.mark.gpu_hw
def test_a_warm_up_that_runs_out_of_memory_is_tried_again(exact_rig: _Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    items = [Item(text=text) for text in _TEXTS[:3]]
    runner = _fresh_runner(exact_rig, monkeypatch)
    runner._stream = None  # the next recording creates and warms up its stream
    model = runner._model
    failures = []

    def warm_up_fails_once(**inputs: Any) -> Any:
        # The warm-up is the only one-row forward of a three-item request.
        if inputs["input_ids"].shape[0] == 1 and not failures:
            failures.append(True)
            raise torch.cuda.OutOfMemoryError("CUDA out of memory during warm-up")
        return model(**inputs)

    monkeypatch.setattr(runner, "_model", warm_up_fails_once)
    eager = _outputs(exact_rig.adapter.extract(items, labels=_LABELS, options=_EAGER))
    for _ in range(2):  # the second sighting records; its warm-up fails
        assert _outputs(exact_rig.adapter.extract(items, labels=_LABELS)) == eager
    assert failures
    assert runner._stream is None
    assert not runner.disabled

    runner._recording_credit = 16.0  # skip the pause
    # Dropping the graphs reset the sightings: seen, then warmed up and recorded, then replayed.
    for _ in range(3):
        assert _outputs(exact_rig.adapter.extract(items, labels=_LABELS)) == eager
    assert runner._stream is not None
    assert runner.graph_count > 0
    assert runner.replayed
    assert not runner.disabled


@pytest.mark.gpu_hw
def test_a_forward_that_cannot_be_recorded_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rig = _Rig(prompt_first=True, scorer="mlp", device="cuda:0", dtype=torch.float16, cuda_graphs="bucketed")
    items = [Item(text=text) for text in _TEXTS[:3]]
    eager = _outputs(rig.adapter.extract(items, labels=_LABELS, options=_EAGER))
    runner = rig.adapter._graphs
    assert runner is not None
    # Without the precomputed table, DeBERTa copies a CPU scalar to the GPU
    # while recording, which CUDA refuses: the runner turns itself off.
    monkeypatch.setattr(runner, "_build_relative_pos", lambda hidden: None)

    graphed = rig.adapter.extract(items, labels=_LABELS)

    assert runner.disabled
    assert _outputs(graphed) == eager
    assert _outputs(rig.adapter.extract(items, labels=_LABELS)) == eager


@pytest.mark.parametrize("mode", ["exact", "bucketed"])
def test_cuda_graphs_run_eagerly_on_cpu(mode: str) -> None:
    items = [Item(text=text) for text in _TEXTS]
    graphs_rig = _Rig(prompt_first=True, scorer="mlp", cuda_graphs=mode)
    plain_rig = _Rig(prompt_first=True, scorer="mlp")

    eager = plain_rig.adapter.extract(items, labels=_LABELS)
    for _ in range(2):
        graphed = graphs_rig.adapter.extract(items, labels=_LABELS)
        assert _outputs(graphed) == _outputs(eager)
    assert graphs_rig.adapter._graphs is None


@pytest.mark.parametrize("value", ["exact", "bucketed", "on", True, None, 1])
def test_requests_cannot_turn_cuda_graphs_on(rig: _Rig, value: object) -> None:
    with pytest.raises(InvalidInputError, match="accepts only 'off'"):
        rig.adapter.extract([Item(text=_TEXTS[0])], labels=_LABELS, options={"cuda_graphs": value})


def test_requests_may_turn_cuda_graphs_off(rig: _Rig) -> None:
    items = [Item(text=text) for text in _TEXTS]

    assert _outputs(rig.adapter.extract(items, labels=_LABELS, options=_EAGER)) == _outputs(
        rig.adapter.extract(items, labels=_LABELS)
    )
