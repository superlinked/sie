"""Native Qwen3.5 backbone and the checkpoint's separate decision readout.

Adapted from the Apache-2.0 reference distributed with
perplexity-ai/pplx-decider-v1.1-27b at 14c9d90137962d300f3de95633f7b6535e80c5ce.
The upstream model derives from Qwen/Qwen3.8-27B, revision
1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, under Apache-2.0. Its modifications
are a trained 255-option readout, fine-tuned backbone/vision weights,
noncausal full attention, last-token pooling and fitted temperature.
SIE adds strict local checkpoint validation and bounded text-only serving.
"""

from __future__ import annotations

import itertools
import json
import math
import string
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from sie_server.adapters.pplx_decider.questions import MAX_OPTIONS


def checkpoint_config(directory: Path) -> dict[str, Any]:
    """Validate saved semantics before acquiring the backbone's weights."""
    config = json.loads((directory / "decision_config.json").read_text())
    if not isinstance(config, dict) or type(config.get("format_version")) is not int or config["format_version"] != 1:
        raise ValueError("Unsupported Perplexity decision checkpoint format")
    revision = config.get("revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in string.hexdigits for c in revision):
        raise ValueError("The checkpoint must declare an immutable base revision")
    temperature = config.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Checkpoint temperature must be positive and finite")
    if config.get("attention_mode", "causal") not in {"causal", "noncausal_full_attention"}:
        raise ValueError("Unsupported checkpoint attention mode")
    if config.get("pooling", "last") not in {"last", "mean"}:
        raise ValueError("Unsupported checkpoint pooling mode")
    codes, token_ids = config.get("codes"), config.get("token_ids")
    if (
        not isinstance(codes, list)
        or len(codes) != MAX_OPTIONS
        or any(not isinstance(code, str) for code in codes)
        or len(set(codes)) != MAX_OPTIONS
        or not isinstance(token_ids, list)
        or len(token_ids) != MAX_OPTIONS
        or any(type(token) is not int or token < 0 for token in token_ids)
        or len(set(token_ids)) != MAX_OPTIONS
    ):
        raise ValueError("The checkpoint must declare 255 distinct answer codes and token IDs")
    return config


def validate_tokenizer(tokenizer: Any, config: Mapping[str, Any]) -> None:
    """Verify code ordering and single-token behavior against the saved readout."""
    candidates = list(string.ascii_uppercase) + [
        "".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)
    ]
    codes = [code for code in candidates if len(tokenizer.encode(code, add_special_tokens=False)) == 1][:MAX_OPTIONS]
    ids = [tokenizer.encode(code, add_special_tokens=False)[0] for code in codes]
    if codes != config["codes"] or ids != config["token_ids"] or len(set(ids)) != MAX_OPTIONS:
        raise ValueError("Checkpoint answer vocabulary differs from its tokenizer")
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Choose an option."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if any(
        tokenizer.encode(prefix + code, add_special_tokens=False) != [*prefix_ids, token]
        for code, token in zip(codes, ids, strict=True)
    ):
        raise ValueError("Answer codes must remain single tokens after the chat prefix")


def enable_noncausal_full_attention(text_model: Any) -> None:
    """Keep padding and native recurrence; only full-attention layers become noncausal."""
    from transformers.masking_utils import (
        create_recurrent_attention_mask,  # ty: ignore[unresolved-import] -- transformers5 bundle
    )

    if text_model.config._attn_implementation != "sdpa":
        raise ValueError("Perplexity noncausal full attention requires SDPA")

    def mask_inputs(
        module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if args or kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("Perplexity decisions require keyword inputs and no KV cache")
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            embeddings = module.embed_tokens(kwargs["input_ids"])
        padding = kwargs.get("attention_mask")
        if padding is None:
            padding = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        if (
            not isinstance(padding, torch.Tensor)
            or padding.ndim != 2
            or padding.shape != embeddings.shape[:2]
            or not padding.bool().any(dim=-1).all()
        ):
            raise ValueError("Expected the complete nonempty processor padding mask")
        kwargs["attention_mask"] = {
            "full_attention": padding[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(
                config=module.config, inputs_embeds=embeddings, attention_mask=padding
            ),
        }
        return args, kwargs

    text_model.register_forward_pre_hook(mask_inputs, with_kwargs=True)


class DecisionModel(torch.nn.Module):
    """One forward produces 255 calibrated option logits, never generated tokens."""

    def __init__(self, backbone: Any, readout: torch.nn.Linear, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.backbone = backbone
        self.readout = readout
        self.temperature = float(config["temperature"])
        self.pooling = config.get("pooling", "last")

    def forward(self, inputs: dict[str, torch.Tensor], counts: list[int]) -> torch.Tensor:
        hidden = self.backbone(**inputs, use_cache=False).last_hidden_state
        if self.pooling == "mean":
            valid = inputs["attention_mask"].bool()
            lengths = valid.sum(dim=1, keepdim=True)
            if not lengths.gt(0).all():
                raise ValueError("Decision pooling requires nonempty rows")
            hidden = (hidden.float().masked_fill(~valid.unsqueeze(-1), 0).sum(dim=1) / lengths).to(hidden.dtype)
        else:
            hidden = hidden[:, -1]
        logits = self.readout(hidden).float()
        if logits.shape != (len(counts), MAX_OPTIONS) or not torch.isfinite(logits).all():
            raise ValueError("Decision model returned invalid logits")
        mask = (
            torch.arange(MAX_OPTIONS, device=logits.device)[None] >= torch.tensor(counts, device=logits.device)[:, None]
        )
        return (logits.masked_fill(mask, -1e9) / self.temperature).softmax(-1)


def load_model(directory: Path, config: Mapping[str, Any], device: str, dtype: torch.dtype) -> DecisionModel:
    """Load safetensors locally through the native class, without remote Python code."""
    # This native architecture is provided by the optional transformers5 bundle.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model  # ty: ignore[unresolved-import]

    backbone, info = Qwen3_5Model.from_pretrained(
        str(directory),
        dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Perplexity backbone weights do not match the native checkpoint architecture")
    weights = load_file(str(directory / "readout.safetensors"))
    size = backbone.config.text_config.hidden_size
    if (
        set(weights) != {"weight"}
        or weights["weight"].shape != (MAX_OPTIONS, size)
        or not weights["weight"].is_floating_point()
        or not torch.isfinite(weights["weight"]).all()
    ):
        raise ValueError("Invalid Perplexity decision readout")
    readout = torch.nn.Linear(size, MAX_OPTIONS, bias=False, dtype=dtype, device=device)
    readout.load_state_dict(weights, strict=True)
    if not torch.isfinite(readout.weight).all():
        raise ValueError("Perplexity decision readout overflows the requested compute precision")
    if config.get("attention_mode", "causal") == "noncausal_full_attention":
        enable_noncausal_full_attention(backbone.language_model)
    model = DecisionModel(backbone, readout, config)
    model.requires_grad_(False).eval()
    return model
