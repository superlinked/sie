"""GLiClass models with a ModernBERT encoder on the packed flash-attention path, against the gliclass forward.

The CPU tests stand a small reference for ``flash_attn_varlen_func`` in for the
kernel, so they check what the runner feeds the shared ModernBERT layer loop
and what it does with the result: packing, positions and RoPE bases, sliding
windows, segment embeddings, repadding, and the scoring head's inputs. The
tiny models have the head layouts of the published ModernBERT checkpoints.
The ``gpu_hw`` test runs the real kernel in float16.
"""

from __future__ import annotations

import logging
import sys
import types
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
import torch
from gliclass import GLiClassModel, GLiClassModelConfig, ZeroShotClassificationPipeline
from sie_server.adapters.gliclass import GLiClassAdapter
from sie_server.adapters.gliclass.modernbert_flash import ModernBertFlashEncoder, token_bound, unsupported_reason
from sie_server.core import inference
from sie_server.core.loader import _build_adapter_kwargs, load_model_configs, reject_unknown_loadtime_options
from sie_server.types.inputs import Item
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from torch.nn import functional
from transformers import DebertaV2Config, ModernBertConfig, PreTrainedTokenizerFast

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
# The shipped profiles that load with the flash encoder: every GLiClass model
# with a ModernBERT or mmBERT encoder (see the server README).
_FLASH_BY_DEFAULT = {
    "knowledgator/gliclass-edge-v3.0",
    "knowledgator/gliclass-instruct-edge-v1.0",
    "knowledgator/gliclass-modern-base-v3.0",
    "knowledgator/gliclass-modern-large-v3.0",
    "knowledgator/gliclass-multilang-edge",
    "knowledgator/opir-edge-multilang-v1.0",
    "knowledgator/opir-edge-v1.0",
}

_MAX_LENGTH = 64
# Lengths from a few tokens to past the window, so rows pad against each other
# and span several sliding windows of the local layers.
_TEXTS: list[str] = [f"the app crashes on startup {'when config file is missing ' * i}".strip() for i in range(11)]
_LONG_TEXT: str = " ".join(["when config file is missing"] * 40)
_LABELS = ["billing", "bug report", "feature request"]
_GROUPS = {"topic": _LABELS, "urgency": ["low", "high"], "remote": ["yes", "no", "can exploit it over network"]}
_EXAMPLES = [{"text": "charged twice for one order", "labels": ["billing"]}]
_INSTRUCTION = "classify the ticket"
_WORDS = sorted(
    {
        word
        for text in [*_TEXTS, _LONG_TEXT, _EXAMPLES[0]["text"], _INSTRUCTION, *(x for v in _GROUPS.values() for x in v)]
        for word in text.split()
    }
)
# Head layouts of the published ModernBERT GLiClass checkpoints.
_LAYOUTS: dict[str, dict[str, Any]] = {
    # gliclass-edge-v3.0, opir-edge-v1.0
    "edge": {"prompt_first": True, "scorer_type": "mlp", "pooling_strategy": "first"},
    # gliclass-instruct-edge-v1.0: segment embeddings, averaged label tokens
    "instruct": {
        "prompt_first": True,
        "scorer_type": "mlp",
        "pooling_strategy": "first",
        "class_token_pooling": "average",
        "use_segment_embeddings": True,
    },
    # gliclass-multilang-edge, opir-edge-multilang-v1.0: text tokens through a cross-attention scorer
    "multilang": {
        "prompt_first": True,
        "scorer_type": "cross-attn",
        "pooling_strategy": "pass",
        "class_token_pooling": "average",
        "extract_text_features": True,
    },
    # gliclass-modern-{base,large}-v3.0 score with a dot product; text first exercises the other order
    "text-first": {"prompt_first": False, "scorer_type": "simple", "pooling_strategy": "first"},
}
_REQUESTS: dict[str, dict[str, Any]] = {
    "labels": {"labels": _LABELS},
    "multi-label": {"labels": _LABELS, "options": {"classification_type": "multi-label"}},
    "instruction": {"labels": _LABELS, "instruction": _INSTRUCTION},
    "examples": {"labels": _LABELS, "options": {"examples": _EXAMPLES}},
    "separate": {"options": {"label_groups": _GROUPS}},
    "joint": {"options": {"label_groups": _GROUPS, "group_encoding": "joint"}},
    "joint-examples": {
        "options": {
            "label_groups": _GROUPS,
            "group_encoding": "joint",
            "examples": [{"text": "charged twice for one order", "labels": {"topic": "billing"}}],
        }
    },
}


def _reference_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """``flash_attn_varlen_func`` for packed ``[tokens, heads, dim]`` rows, one sequence at a time."""
    assert not causal
    assert torch.equal(cu_seqlens_q, cu_seqlens_k)
    bounds = cu_seqlens_q.tolist()
    lengths = [end - start for start, end in pairwise(bounds)]
    assert max(lengths) == max_seqlen_q == max_seqlen_k
    out = torch.empty_like(query)
    for start, end in pairwise(bounds):
        rows = [t[start:end].transpose(0, 1) for t in (query, key, value)]
        mask = None
        if window_size != (-1, -1):
            positions = torch.arange(end - start, device=query.device)
            mask = (positions[:, None] - positions[None, :]).abs() <= window_size[0]
        attended = functional.scaled_dot_product_attention(*rows, attn_mask=mask, scale=softmax_scale)
        out[start:end] = attended.transpose(0, 1)
    return out


@pytest.fixture
def reference_flash(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("flash_attn")
    module.flash_attn_varlen_func = _reference_varlen  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "flash_attn", module)


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
        # As ModernBERT tokenizers: no token type ids.
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.add_tokens(["<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>"], special_tokens=True)
    tokenizer.model_max_length = _MAX_LENGTH
    return tokenizer


def _encoder_config(tokenizer: PreTrainedTokenizerFast) -> ModernBertConfig:
    config = ModernBertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=2,
        max_position_embeddings=128,
        # Layers 0 and 2 attend globally, layer 1 through a window of 4 tokens
        # each side; the two RoPE bases differ, as in ModernBERT.
        global_attn_every_n_layers=2,
        local_attention=8,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        cls_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        sep_token_id=tokenizer.sep_token_id,
        reference_compile=False,
    )
    config._attn_implementation = "sdpa"
    return config


def _model(tokenizer: PreTrainedTokenizerFast, **layout: Any) -> GLiClassModel:
    torch.manual_seed(0)
    config = GLiClassModelConfig(
        encoder_config=_encoder_config(tokenizer),
        encoder_model="tiny-modernbert-not-on-disk",
        class_token_index=tokenizer.convert_tokens_to_ids("<<LABEL>>"),
        text_token_index=tokenizer.convert_tokens_to_ids("<<SEP>>"),
        example_token_index=tokenizer.convert_tokens_to_ids("<<EXAMPLE>>"),
        vocab_size=len(tokenizer),
        scorer_mlp_hidden_size=32,
        scorer_num_heads=2,
        dropout=0.0,
        **layout,
    )
    model = GLiClassModel(config).eval()
    with torch.no_grad():
        # At initialization scale, attention is nearly uniform and segment
        # embeddings are small, so positions, RoPE bases, windows and segment
        # ids barely move scores. Larger weights make each of them visible.
        for layer in model.model.encoder_model.layers:
            layer.attn.Wqkv.weight.normal_(std=0.3)
        segments = getattr(model.model, "segment_embeddings", None)
        if segments is not None:
            segments.weight.normal_(std=1.0)
    return model


class _Rig:
    """A tiny ModernBERT GLiClass model behind two adapters: the gliclass forward and the flash encoder."""

    def __init__(self, layout: str, *, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        tokenizer = _tokenizer()
        self.model = _model(tokenizer, **_LAYOUTS[layout]).to(device, dtype=dtype)
        pipe = ZeroShotClassificationPipeline(
            self.model,
            tokenizer,
            max_length=_MAX_LENGTH,
            classification_type="single-label",
            device=device,
            progress_bar=False,
        ).pipe
        self.eager = GLiClassAdapter("tiny", max_seq_length=_MAX_LENGTH)
        self.eager._attach(pipe, tokenizer)
        self.flash = GLiClassAdapter("tiny", max_seq_length=_MAX_LENGTH, modernbert_flash=True)
        self.flash._attach(pipe, tokenizer)
        if device == "cpu":
            # Off CUDA the adapter keeps the gliclass forward; drive the runner directly.
            assert self.flash._flash is None
            self.flash._flash = ModernBertFlashEncoder(self.model, max_length=_MAX_LENGTH)
        assert self.flash._flash is not None
        self.runner: ModernBertFlashEncoder = self.flash._flash


def _answers(output: Any) -> list[dict[str, float]]:
    if output.data:
        return [
            {f"{group}.{label}": p for group, answer in row.items() for label, p in answer["probabilities"].items()}
            for row in output.data
        ]
    return [{c["label"]: c["score"] for c in row} for row in output.classifications]


def _assert_close(eager: Any, flash: Any, *, abs_tol: float) -> None:
    assert flash.input_token_counts == eager.input_token_counts
    assert flash.errors == eager.errors
    for expected, got in zip(_answers(eager), _answers(flash), strict=True):
        assert got.keys() == expected.keys()
        for label, probability in expected.items():
            assert got[label] == pytest.approx(probability, abs=abs_tol), label


@pytest.fixture(scope="module", params=list(_LAYOUTS))
def rig(request: pytest.FixtureRequest) -> _Rig:
    return _Rig(request.param)


@pytest.mark.parametrize("request_kwargs", list(_REQUESTS.values()), ids=list(_REQUESTS))
def test_flash_encoder_scores_like_the_gliclass_forward(
    rig: _Rig, reference_flash: None, request_kwargs: dict[str, Any]
) -> None:
    # Eleven texts of different lengths: padded forwards of eight and three rows.
    items = [Item(text=text) for text in _TEXTS]
    before = rig.runner.stats.flash

    flash = rig.flash.extract(items, **request_kwargs)

    assert rig.runner.stats.flash > before
    assert not rig.runner.stats.eager
    _assert_close(rig.eager.extract(items, **request_kwargs), flash, abs_tol=1e-5)


def test_truncated_long_documents_score_like_the_gliclass_forward(rig: _Rig, reference_flash: None) -> None:
    items = [Item(text=_LONG_TEXT), Item(text=_TEXTS[0]), Item(text=_LONG_TEXT + " the app crashes")]
    for kwargs in (_REQUESTS["labels"], _REQUESTS["separate"], _REQUESTS["joint"]):
        request: dict[str, Any] = {
            **kwargs,
            "options": {**kwargs.get("options", {}), "overflow_policy": "truncate_text"},
        }
        _assert_close(rig.eager.extract(items, **request), rig.flash.extract(items, **request), abs_tol=1e-5)
    assert not rig.runner.stats.eager


def test_the_scoring_head_reads_zeros_at_padded_positions(reference_flash: None) -> None:
    rig = _Rig("edge")
    head = rig.model.model.process_encoder_output
    seen: list[torch.Tensor] = []

    def spy(input_ids: Any, attention_mask: torch.Tensor, encoder_layer: torch.Tensor, *args: Any) -> Any:
        seen.append(encoder_layer[attention_mask == 0])
        return head(input_ids, attention_mask, encoder_layer, *args)

    rig.model.model.process_encoder_output = spy
    try:
        rig.flash.extract([Item(text=_TEXTS[0]), Item(text=_TEXTS[6])], labels=_LABELS)
    finally:
        rig.model.model.process_encoder_output = head
    assert seen[0].numel() > 0
    assert torch.count_nonzero(seen[0]) == 0


def _padded(rig: _Rig, texts: list[str]) -> dict[str, torch.Tensor]:
    pipe = rig.flash._pipe
    assert pipe is not None
    return dict(pipe.prepare_inputs(texts, _LABELS, same_labels=True))


def test_rows_that_are_not_right_padded_run_the_gliclass_forward(reference_flash: None) -> None:
    rig = _Rig("edge")
    inputs = _padded(rig, [_TEXTS[0], _TEXTS[5]])
    inputs = {name: tensor.flip(-1) for name, tensor in inputs.items()}  # padding first

    assert rig.runner.run(inputs, len(_LABELS)) is None
    assert rig.runner.stats.eager == {"not_right_padded": 1}


def test_inputs_other_than_ids_and_mask_run_the_gliclass_forward(reference_flash: None) -> None:
    rig = _Rig("edge")
    inputs = _padded(rig, [_TEXTS[0]])
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])

    assert rig.runner.run(inputs, len(_LABELS)) is None
    assert rig.runner.stats.eager == {"unsupported_inputs": 1}


def test_rows_past_the_rope_tables_run_the_gliclass_forward(reference_flash: None) -> None:
    rig = _Rig("edge")
    short = ModernBertFlashEncoder(rig.model, max_length=8)

    assert short.run(_padded(rig, [_TEXTS[5]]), len(_LABELS)) is None
    assert short.stats.eager == {"too_long": 1}


def test_forwards_past_the_token_bound_run_the_gliclass_forward(reference_flash: None) -> None:
    rig = _Rig("edge")
    items = [Item(text=text) for text in _TEXTS[:4]]
    inputs = _padded(rig, _TEXTS[:4])
    rig.runner._max_tokens = int(inputs["attention_mask"].sum()) - 1

    assert rig.runner.run(inputs, len(_LABELS)) is None
    assert rig.runner.stats.eager == {"too_large": 1}
    # The gliclass forward answers such a request, so it scores exactly as that path does.
    assert _answers(rig.flash.extract(items, labels=_LABELS)) == _answers(rig.eager.extract(items, labels=_LABELS))
    assert rig.runner.stats.flash == 0


@pytest.mark.parametrize(("hidden_size", "tokens"), [(256, 4096), (384, 4096), (768, 2048), (1024, 1024)])
def test_wider_encoders_hand_smaller_forwards_to_the_gliclass_forward(hidden_size: int, tokens: int) -> None:
    assert token_bound(hidden_size) == tokens


def _supported_model() -> GLiClassModel:
    return _model(_tokenizer(), **_LAYOUTS["edge"]).half()


@pytest.fixture
def flash_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference, "is_flash_attention_available", lambda device=None: True)


def test_a_float16_modernbert_model_on_cuda_is_supported(flash_available: None) -> None:
    assert unsupported_reason(_supported_model(), "cuda:0") is None


def test_off_cuda_the_model_runs_the_gliclass_forward(flash_available: None) -> None:
    assert unsupported_reason(_supported_model(), "cpu") == "flash attention needs a CUDA device"
    assert unsupported_reason(_supported_model(), "mps") == "flash attention needs a CUDA device"


def test_without_flash_attn_the_model_runs_the_gliclass_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference, "is_flash_attention_available", lambda device=None: False)
    reason = unsupported_reason(_supported_model(), "cuda:0")
    assert reason is not None
    assert "flash-attn" in reason


def test_a_deberta_encoder_runs_the_gliclass_forward(flash_available: None) -> None:
    tokenizer = _tokenizer()
    config = GLiClassModelConfig(
        encoder_config=DebertaV2Config(
            vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64
        ),
        encoder_model="tiny-deberta-not-on-disk",
        vocab_size=len(tokenizer),
    )
    reason = unsupported_reason(GLiClassModel(config).half(), "cuda:0")
    assert reason == "the deberta-v2 encoder is not a ModernBERT"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"squeeze_layers": True}, "the model scores intermediate encoder layers"),
        ({"encoder_layer_id": 2}, "the model scores intermediate encoder layers"),
        ({"architecture_type": "bi-encoder"}, "only uni-encoder GLiClass models are supported"),
    ],
)
def test_models_the_runner_does_not_cover_run_the_gliclass_forward(
    flash_available: None, change: dict[str, Any], reason: str
) -> None:
    model = _supported_model()
    for name, value in change.items():
        setattr(model.config, name, value)
    assert unsupported_reason(model, "cuda:0") == reason


def test_float32_weights_run_the_gliclass_forward(flash_available: None) -> None:
    reason = unsupported_reason(_supported_model().float(), "cuda:0")
    assert reason == "flash attention needs float16 or bfloat16 weights, not torch.float32"


def test_the_option_is_off_by_default() -> None:
    rig_model = _model(_tokenizer(), **_LAYOUTS["edge"])
    adapter = GLiClassAdapter("tiny", max_seq_length=_MAX_LENGTH)
    pipe = ZeroShotClassificationPipeline(
        rig_model, _tokenizer(), max_length=_MAX_LENGTH, device="cpu", progress_bar=False
    ).pipe
    adapter._attach(pipe, _tokenizer())
    assert adapter._flash is None


def test_off_cuda_the_option_logs_why_it_does_not_apply(caplog: pytest.LogCaptureFixture) -> None:
    tokenizer = _tokenizer()
    pipe = ZeroShotClassificationPipeline(
        _model(tokenizer, **_LAYOUTS["edge"]), tokenizer, max_length=_MAX_LENGTH, device="cpu", progress_bar=False
    ).pipe
    adapter = GLiClassAdapter("tiny-edge", max_seq_length=_MAX_LENGTH, modernbert_flash=True)
    with caplog.at_level(logging.INFO, logger="sie_server.adapters.gliclass"):
        adapter._attach(pipe, tokenizer)
    assert adapter._flash is None
    assert "tiny-edge" in caplog.text
    assert "CUDA" in caplog.text


@pytest.mark.parametrize("value", ["true", 1, None])
def test_modernbert_flash_must_be_a_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="modernbert_flash"):
        GLiClassAdapter("tiny", modernbert_flash=value)  # ty: ignore[invalid-argument-type]


def test_unloading_drops_the_runner(reference_flash: None) -> None:
    rig = _Rig("edge")
    rig.flash.unload()
    assert rig.flash._flash is None


def test_the_modernbert_gliclass_models_ship_with_the_flash_encoder() -> None:
    configs = load_model_configs(_MODELS_DIR)
    # Named profiles (``model:profile``) inherit the default profile's load-time options.
    enabled = {
        name.split(":")[0]
        for name, config in configs.items()
        if config.resolve_profile("default").adapter_path.endswith(":GLiClassAdapter")
        and config.resolve_profile("default").loadtime.get("modernbert_flash", False)
    }

    assert enabled == _FLASH_BY_DEFAULT
    for name in _FLASH_BY_DEFAULT:
        loadtime = configs[name].resolve_profile("default").loadtime
        reject_unknown_loadtime_options(GLiClassAdapter, loadtime, model_name=name)
        adapter = GLiClassAdapter(**_build_adapter_kwargs(configs[name], "float16"))
        assert adapter._modernbert_flash is True


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # ty: ignore[unresolved-import]
    except ImportError:
        return False
    return True


@pytest.mark.gpu_hw
@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_the_flash_kernel_scores_like_the_gliclass_forward_in_float16(layout: str) -> None:
    if not torch.cuda.is_available() or not _has_flash_attn():
        pytest.skip("requires CUDA and flash-attn")
    rig = _Rig(layout, device="cuda:0", dtype=torch.float16)
    items = [Item(text=text) for text in _TEXTS]
    for request_kwargs in _REQUESTS.values():
        _assert_close(
            rig.eager.extract(items, **request_kwargs), rig.flash.extract(items, **request_kwargs), abs_tol=2e-2
        )
    assert rig.runner.stats.flash > 0
    assert not rig.runner.stats.eager
