"""Native Qwen3.5 decision backbones with block-bidirectional full attention."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, ClassVar

import torch

from sie_server.adapters._checkpoint import resolve_checkpoint
from sie_server.adapters._qwen3_5_packed import resolve_kernels
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._transformers_compat import install_transformers_security_fixes
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.adapters.decision import d3, qev
from sie_server.adapters.decision.base import MAX_ITEM_TOKENS, DecisionBatchAdapter, Row
from sie_server.adapters.decision.packed import PackedDecisionReadout
from sie_server.adapters.decision.questions import Question
from sie_server.adapters.errors import InputTooLongError
from sie_server.adapters.pplx_decider.model import validate_tokenizer
from sie_server.types.inputs import InvalidInputError, Item

_ATTENTION = "sie_decision_prefix_bidir"
logger = logging.getLogger(__name__)
_FILES = (
    "config.json",
    "decision_config.json",
    "readout.json",
    "readout.safetensors",
    "model*.safetensors",
    "model*.safetensors.index.json",
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template*",
)


def prefix_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: tuple[tuple[int, int], ...],
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Prefix queries see the prefix; suffix queries see both blocks, without a quadratic mask."""
    _ = module, kwargs
    if dropout or not isinstance(attention_mask, tuple) or len(attention_mask) != query.shape[0]:
        raise ValueError("Decision attention requires one block boundary per row and no dropout")
    groups = query.shape[1] // key.shape[1]
    result = torch.zeros_like(query)
    for row, (first, end) in enumerate(attention_mask):
        if not 0 <= first < end < query.shape[2] or key.shape[2] != query.shape[2]:
            raise ValueError("Decision attention requires complete uncached prefix and suffix blocks")
        q = query[row : row + 1, :, first:]
        k = key[row : row + 1, :, first:].repeat_interleave(groups, dim=1)
        v = value[row : row + 1, :, first:].repeat_interleave(groups, dim=1)
        size = end - first
        result[row : row + 1, :, first:end] = torch.nn.functional.scaled_dot_product_attention(
            q[:, :, :size], k[:, :, :size], v[:, :, :size], scale=scaling
        )
        result[row : row + 1, :, end:] = torch.nn.functional.scaled_dot_product_attention(
            q[:, :, size:], k, v, scale=scaling
        )
    return result.transpose(1, 2).contiguous(), None


class QwenReadout(torch.nn.Module):
    def __init__(self, backbone: Any, weight: torch.Tensor) -> None:
        super().__init__()
        from transformers import AttentionInterface

        AttentionInterface.register(_ATTENTION, prefix_attention)
        backbone.config._attn_implementation = _ATTENTION
        self.backbone = backbone
        self.readout = torch.nn.Linear(
            weight.shape[1], weight.shape[0], bias=False, device=weight.device, dtype=weight.dtype
        )
        self.readout.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.packed: PackedDecisionReadout | None = None
        self.pack_single = False
        self.requires_grad_(False).eval()

    def forward(self, ids: torch.Tensor, present: torch.Tensor, bounds: tuple[tuple[int, int], ...]) -> torch.Tensor:
        from transformers.masking_utils import create_recurrent_attention_mask  # ty: ignore[unresolved-import]

        if self.packed is not None and (len(bounds) > 1 or self.pack_single):
            return self.packed(ids, present, bounds)
        embeddings = self.backbone.embed_tokens(ids)
        scan = create_recurrent_attention_mask(
            config=self.backbone.config, inputs_embeds=embeddings, attention_mask=present
        )
        hidden = self.backbone(
            inputs_embeds=embeddings,
            position_ids=(present.long().cumsum(-1) - 1).clamp_min(0),
            attention_mask={"full_attention": bounds, "linear_attention": scan},
            use_cache=False,
        ).last_hidden_state[:, -1]
        logits = self.readout(hidden.to(self.readout.weight.dtype)).float()
        if not torch.isfinite(logits).all():
            raise ValueError("Decision head returned nonfinite logits")
        return logits


def _strict_load(model_class: Any, directory: Path, device: str, dtype: torch.dtype) -> Any:
    model, loading = model_class.from_pretrained(
        str(directory),
        dtype=dtype,
        device_map={"": device},
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Decision checkpoint does not exactly match its native Qwen3.5 architecture")
    return model


class QwenDecisionAdapter(DecisionBatchAdapter):
    """d3 and qev recipes over the same native Transformers backbone and bounded batch path."""

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=("_model", "_tokenizer"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        decision_format: str,
        max_seq_length: int = 8192,
        max_forward_tokens: int = 16384,
        compute_precision: ComputePrecision = "bfloat16",
        packed: bool | None = None,
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        if decision_format not in {"d3", "qev"}:
            raise ValueError("decision_format must be d3 or qev")
        if compute_precision not in {"bfloat16", "float32"}:
            raise ValueError("Qwen decisions require bfloat16 or float32")
        if packed is not None and type(packed) is not bool:
            raise ValueError("packed must be a boolean or None")
        if type(max_seq_length) is not int or not 1 <= max_seq_length <= 16384:
            raise ValueError("Qwen decision max_seq_length must be between 1 and 16384")
        if type(max_forward_tokens) is not int or not max_seq_length <= max_forward_tokens <= MAX_ITEM_TOKENS:
            raise ValueError("max_forward_tokens must cover one full row and be at most 32768")
        self._model_name_or_path = str(model_name_or_path)
        self._revision, self._compute_precision = revision, compute_precision
        self._decision_format = decision_format
        self._max_seq_length, self._max_forward_tokens = max_seq_length, max_forward_tokens
        self._model: QwenReadout | None = None
        self._packed = packed
        self._tokenizer: Any = None
        self._device: str | None = None
        self._codes: tuple[str, ...] = ()
        self._temperatures: tuple[float, ...] = (1.0,)
        self._temperature_bounds: tuple[int, ...] = ()

    def load(self, device: str) -> None:
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5.modeling_qwen3_5 import (  # ty: ignore[unresolved-import]
            Qwen3_5ForConditionalGeneration,
            Qwen3_5Model,
        )

        directory = resolve_checkpoint(self._model_name_or_path, revision=self._revision, allow_patterns=_FILES)
        install_transformers_security_fixes()
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        dtype = self._resolve_dtype() if device.startswith("cuda") else torch.float32
        if self._decision_format == "d3":
            config = json.loads((directory / "decision_config.json").read_text())
            expected = {
                "format_version": 1,
                "format_id": "d3-code-readout-v1",
                "prompt": "d3",
                "pooling": "last",
                "attention_mode": "prefix_bidir",
                "prompt_layout": "header-v1",
                "readout_dtype": "float32",
            }
            if any(config.get(key) != value for key, value in expected.items()):
                raise ValueError("Unsupported d3 checkpoint contract")
            validate_tokenizer(tokenizer, config)
            codes = tuple(config["codes"])
            calibration = config.get("temperature_by_questions")
            bounds = tuple(calibration["max_questions"]) if calibration else ()
            temperatures = tuple(calibration["temperature"]) if calibration else (config["temperature"],)
            if (
                len(temperatures) != len(bounds) + 1
                or any(type(value) is not int or value < 1 for value in bounds)
                or list(bounds) != sorted(set(bounds))
                or any(
                    type(value) not in (float, int) or not math.isfinite(value) or value <= 0 for value in temperatures
                )
            ):
                raise ValueError("Invalid d3 question-count calibration")
            weights = load_file(str(directory / "readout.safetensors"), device=device)
            if set(weights) != {"weight"} or weights["weight"].ndim != 2 or weights["weight"].shape[0] != 255:
                raise ValueError("d3 needs exactly one 255-way readout weight")
            model = _strict_load(Qwen3_5Model, directory, device, dtype)
            backbone, weight = model.language_model, weights["weight"].float()
        else:
            config = json.loads((directory / "config.json").read_text())
            readout = json.loads((directory / "readout.json").read_text())
            overrides = config.get("modeling_overrides", {})
            if (
                overrides.get("attention") != {"full_attention": "block_bidirectional", "linear_attention": "causal"}
                or overrides.get("readout") != "slot_lm"
                or readout.get("readout") != "slot_lm"
            ):
                raise ValueError("qev requires block-bidirectional full attention and a slot_lm readout")
            codes = qev.SLOTS
            encoded = [tokenizer.encode(code, add_special_tokens=False) for code in codes]
            if any(len(ids) != 1 or tokenizer.decode(ids).strip() != code for code, ids in zip(codes, encoded)):
                raise ValueError("qev tokenizer does not preserve all 256 slot labels")
            token_ids = [ids[0] for ids in encoded]
            if len(set(token_ids)) != len(token_ids):
                raise ValueError("qev slot labels must have distinct token ids")
            model = _strict_load(Qwen3_5ForConditionalGeneration, directory, device, dtype)
            backbone = model.model.language_model
            weight = model.lm_head.weight[token_ids].detach().float()
            bounds, temperatures = (), (1.0,)
        if weight.shape[1] != backbone.config.hidden_size or not torch.isfinite(weight).all():
            raise ValueError("Invalid decision readout shape or values")
        native = QwenReadout(backbone, weight)
        self._model = native
        kernels = (
            resolve_kernels(device, mode=False if self._packed is False else None)
            if device.startswith("cuda") and dtype == torch.bfloat16
            else None
        )
        if kernels is not None:
            try:
                from flash_attn import flash_attn_varlen_func  # ty: ignore[unresolved-import]
            except ImportError:
                if self._packed:
                    raise RuntimeError("Packed Qwen decisions require FlashAttention") from None
                logger.info("FlashAttention unavailable; keeping native decision execution")
            else:
                native.packed = PackedDecisionReadout(backbone, native.readout, kernels, flash_attn_varlen_func)
                native.pack_single = self._packed is True
        elif self._packed:
            raise ValueError("Packed Qwen decisions require CUDA bfloat16 and fused Qwen kernels")
        self._tokenizer, self._device, self._codes = tokenizer, device, codes
        self._temperature_bounds, self._temperatures = bounds, temperatures

    def _check_decision_loaded(self) -> None:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

    def _parse_questions(self, schema: dict[str, Any] | None) -> tuple[Question, ...]:
        questions = super()._parse_questions(schema)
        for question in questions:
            if question.kind != "score":
                continue
            if len(question.keys) > 10:
                raise InvalidInputError("Qwen decision score questions support at most 10 levels")
            if self._decision_format == "d3" and (
                not all(question.descriptions) or len(set(question.descriptions)) != len(question.descriptions)
            ):
                raise InvalidInputError("d3 score levels must be distinct and nonempty")
        return questions

    def _item_rows(self, index: int, item: Item, questions: tuple[Question, ...]) -> list[Row]:
        state = self._raw_state(item)
        with self._tokenizer_guard():
            rendered = (
                d3.prompts(self._tokenizer, state, questions, self._codes, max_seq_length=self._max_seq_length)
                if self._decision_format == "d3"
                else qev.prompts(self._tokenizer, state, questions, max_seq_length=self._max_seq_length)
            )
        if (
            any(len(ids) > self._max_seq_length for ids, _ in rendered)
            or sum(len(ids) for ids, _ in rendered) > MAX_ITEM_TOKENS
        ):
            raise InputTooLongError("Decision rows exceed the token budget; no input was truncated")
        bucket = next(
            (i for i, bound in enumerate(self._temperature_bounds) if len(questions) <= bound),
            len(self._temperature_bounds),
        )
        return [
            Row(
                index,
                (question,),
                {"input_ids": torch.tensor(ids, dtype=torch.long)},
                prefix_length=prefix,
                temperature=self._temperatures[bucket],
            )
            for question, (ids, prefix) in zip(questions, rendered, strict=True)
        ]

    def _forward(self, rows: list[Row]) -> list[list[float]]:
        self._check_decision_loaded()
        model = self._model
        if model is None:
            raise RuntimeError(ERR_NOT_LOADED)
        width = max(row.length for row in rows)
        pad = self._tokenizer.pad_token_id
        if pad is None:
            pad = self._tokenizer.eos_token_id
        ids = torch.stack(
            [torch.nn.functional.pad(row.inputs["input_ids"], (width - row.length, 0), value=pad) for row in rows]
        ).to(self._device)
        present = torch.stack([torch.arange(width) >= width - row.length for row in rows]).to(self._device)
        if any(row.prefix_length is None for row in rows):
            raise ValueError("Qwen decision rows require a prefix boundary")
        bounds = tuple((width - row.length, width - row.length + (row.prefix_length or 0)) for row in rows)
        with torch.inference_mode():
            logits = model(ids, present, bounds)
            result = [
                (values[: len(row.question.keys)] / row.temperature).softmax(-1).cpu().tolist()
                for row, values in zip(rows, logits, strict=True)
            ]
        for row, values in zip(rows, result, strict=True):
            if self._decision_format == "qev" and row.question.kind == "noul":
                values.reverse()
        return result

    def _answer(self, question: Question, probabilities: list[float]) -> dict[str, Any]:
        result = super()._answer(question, probabilities)
        if self._decision_format == "d3" and question.kind != "noul":
            entropy = -math.fsum(value * math.log(value) for value in probabilities if value > 0)
            result["confidence"] = (
                min(1.0, max(0.0, 1.0 - entropy / math.log(len(probabilities)))) if len(probabilities) > 1 else 1.0
            )
            if question.kind == "score":
                result["legend"] = {
                    key: value
                    if isinstance(value, str)
                    else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    for key, value in zip(question.keys, question.legend, strict=True)
                }
        return result
