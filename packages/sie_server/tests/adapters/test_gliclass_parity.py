"""GLiClass scores against the gliclass pipeline, on a tiny random model (no downloads).

Before scoring moved into the adapter, a labels request ran exactly
``ZeroShotClassificationPipeline(texts, labels, threshold=0.0,
return_hierarchical=True, prompt=..., examples=...)``. The adapter must still
return the scores that call returns, bit for bit. Separate label groups must
score like one labels request per group.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from gliclass import GLiClassModel, GLiClassModelConfig, ZeroShotClassificationPipeline
from sie_server.adapters.gliclass import GLiClassAdapter
from sie_server.types.inputs import Item
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import DebertaV2Config, PreTrainedTokenizerFast

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

    def __init__(self, *, prompt_first: bool, scorer: str) -> None:
        tokenizer = _tokenizer()
        model = _model(tokenizer, prompt_first=prompt_first, scorer=scorer)
        self.pipelines = {
            classification_type: ZeroShotClassificationPipeline(
                model,
                tokenizer,
                max_length=_MAX_LENGTH,
                classification_type=classification_type,
                device="cpu",
                progress_bar=False,
            )
            for classification_type in ("single-label", "multi-label")
        }
        self.adapter = GLiClassAdapter("tiny", max_seq_length=_MAX_LENGTH)
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
