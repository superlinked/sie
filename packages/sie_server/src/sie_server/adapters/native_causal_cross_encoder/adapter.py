"""Publisher-native causal CrossEncoder scoring with complete input admission."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.adapters.errors import InputTooLongError
from sie_server.core.inference_output import ScoreOutput
from sie_server.types.inputs import InvalidInputError, Item

_CONTEXT_TOKENS = 32768
_FORWARD_PADDED_TOKENS = 32768
_TRUE_TOKEN_ID = 9454
_MODULE_TYPES = (
    "sentence_transformers.base.modules.transformer.Transformer",
    "sentence_transformers.cross_encoder.modules.logit_score.LogitScore",
)
_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
_ERR_NATIVE_FEATURES = "Native causal CrossEncoder returned invalid complete-row features"
_ERR_NATIVE_SCORES = "Native causal CrossEncoder returned invalid scores"


class _NativeInputTooLongError(InputTooLongError, InvalidInputError):
    """Preserve INPUT_TOO_LONG while permitting fused-request isolation."""


@dataclass(frozen=True, slots=True)
class _NativeRows:
    """One immutable admitted encoded row for every original text pair."""

    pairs: tuple[tuple[str, str], ...]
    ids: tuple[tuple[int, ...], ...]
    counts: tuple[int, ...]
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        size = len(self.pairs)
        if (
            not len(self.ids) == len(self.counts) == len(self.indices) == size
            or any(type(index) is not int for index in self.indices)
            or self.indices != tuple(range(size))
            or any(
                type(count) is not int or not row or len(row) != count
                for row, count in zip(self.ids, self.counts, strict=True)
            )
        ):
            raise RuntimeError("Native causal CrossEncoder admission rows are misaligned")


class NativeCausalCrossEncoderAdapter(BaseAdapter):
    """Score ZeRank 2 pairs using its saved Transformer and LogitScore modules.

    The publisher's query/system and document/user chat template is preserved.
    Scores are raw positive-token logits, with the saved Identity activation.
    Complete native rows, including template tokens, must fit the configured
    context. Runtime ``max_seq_length`` can reduce that context. Token usage
    counts those exact rows without left padding. Forwards are bounded to
    32768 padded tokens independently of the outer scheduler's cost estimate.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",), outputs=("score",), unload_fields=("_model", "_tokenizer", "_move_features")
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        revision: str | None = None,
        max_seq_length: int | None = None,
        compute_precision: ComputePrecision | None = None,
    ) -> None:
        self._model_name_or_path = str(model_name_or_path)
        self._revision = revision
        self._max_seq_length = self._validate_limit(
            _CONTEXT_TOKENS if max_seq_length is None else max_seq_length, _CONTEXT_TOKENS
        )
        if compute_precision is not None and compute_precision not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Native causal CrossEncoder compute_precision must be float16, bfloat16 or float32")
        self._compute_precision = compute_precision
        self._forward_token_budget = _FORWARD_PADDED_TOKENS
        self._model: Any = None
        self._tokenizer: Any = None
        self._move_features: Any = None
        self._device: str | None = None

    def _require_saved_modules(self) -> None:
        root = Path(self._model_name_or_path)
        if root.is_dir():
            path = root / "modules.json"
        else:
            path = Path(hf_hub_download(self._model_name_or_path, "modules.json", revision=self._revision))
        try:
            modules = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("Native causal CrossEncoder requires saved modules.json") from exc
        if (
            not isinstance(modules, list)
            or len(modules) != 2
            or any(
                not isinstance(module, dict)
                or cast("dict[str, Any]", module).get("idx") != index
                or cast("dict[str, Any]", module).get("type") != expected
                for index, (module, expected) in enumerate(zip(modules, _MODULE_TYPES, strict=True))
            )
        ):
            raise RuntimeError("Native causal CrossEncoder requires saved Transformer and LogitScore modules")

    def load(self, device: str) -> None:
        """Load and verify the native checkpoint without a fallback scorer."""
        from sentence_transformers import CrossEncoder
        from sentence_transformers.base.modules import Transformer
        from sentence_transformers.cross_encoder.modules import LogitScore
        from sentence_transformers.util import batch_to_device

        self._require_saved_modules()
        precision = self._compute_precision or ("bfloat16" if device.startswith("cuda") else "float32")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[precision]
        model = CrossEncoder(
            self._model_name_or_path,
            revision=self._revision,
            device=device,
            max_length=self._max_seq_length,
            trust_remote_code=False,
            model_kwargs={"dtype": dtype, "attn_implementation": "sdpa"},
        )
        modules = list(model.children())
        if len(modules) != 2 or type(modules[0]) is not Transformer or type(modules[1]) is not LogitScore:
            raise RuntimeError("Native causal CrossEncoder loaded unexpected module classes")
        transformer, scorer = modules
        if (
            transformer.transformer_task != "text-generation"
            or transformer.module_output_name != "causal_logits"
            or transformer.can_flatten_inputs is not False
            or transformer.do_lower_case is not False
            or scorer.module_input_name != "causal_logits"
            or type(scorer.true_token_id) is not int
            or scorer.true_token_id != _TRUE_TOKEN_ID
            or scorer.false_token_id is not None
            or type(model.activation_fn) is not torch.nn.Identity
        ):
            raise RuntimeError("Native causal CrossEncoder checkpoint does not match the raw positive-token scorer")
        expected_modalities = {
            "text": {"method": "forward", "method_output_name": "logits"},
            "message": {"method": "forward", "method_output_name": "logits", "format": "flat"},
        }
        if transformer.modality_config != expected_modalities:
            raise RuntimeError("Native causal CrossEncoder checkpoint has an unexpected input formatter")
        tokenizer = transformer.tokenizer
        processor = transformer.processor
        if tokenizer is None or tokenizer.padding_side != "left" or processor.padding_side != "left":
            raise RuntimeError("Native causal CrossEncoder requires native left padding")
        self._verify_template(processor)
        processing = copy.deepcopy(transformer.processing_kwargs)
        if (
            not isinstance(processing, dict)
            or processing.get("chat_template") != {"add_generation_prompt": True}
            or not isinstance(processing.get("text", {}), dict)
        ):
            raise RuntimeError("Native causal CrossEncoder requires the saved generation-prefix processing")
        processing["text"] = {**processing.get("text", {}), "truncation": False}
        transformer.processing_kwargs = processing
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._move_features = batch_to_device
        self._device = device

    @staticmethod
    def _verify_template(tokenizer: Any) -> None:
        if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
            raise RuntimeError("Native causal CrossEncoder requires the publisher chat template")
        query, document = "Query\n東京", "Document\nsource-tail-✓"
        expected = (
            f"<|im_start|>system\n{query}<|im_end|>\n<|im_start|>user\n{document}<|im_end|>\n<|im_start|>assistant\n"
        )
        rendered = tokenizer.apply_chat_template(
            [{"role": "query", "content": query}, {"role": "document", "content": document}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if rendered != expected:
            raise RuntimeError("Native causal CrossEncoder chat template differs from the publisher format")

    def score(
        self, query: Item, items: list[Item], *, instruction: str | None = None, options: dict[str, Any] | None = None
    ) -> list[float]:
        """Score a query against documents through the same complete-row path."""
        return self.score_pairs([query] * len(items), items, instruction=instruction, options=options).scores.tolist()

    def score_pairs(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ScoreOutput:
        """Admit every full row, score bounded native chunks, and restore pair order."""
        self._check_loaded()
        limit = self._runtime_limit(instruction, options)
        pairs = self._parse_pairs(queries, docs)
        if not pairs:
            return ScoreOutput(scores=np.empty(0, dtype=np.float32), input_token_counts=[])
        with self._tokenizer_guard():
            rows = self._admit(pairs, limit)
            scores = np.empty(len(pairs), dtype=np.float32)
            with torch.inference_mode():
                for indices in self._chunks(rows):
                    features = self._preprocess([rows.pairs[index] for index in indices])
                    ids = self._read_rows(features, len(indices), limit)
                    if ids != tuple(rows.ids[index] for index in indices):
                        raise RuntimeError("Native causal CrossEncoder encoded rows changed after admission")
                    if features["input_ids"].numel() > self._forward_token_budget:
                        raise RuntimeError(
                            "Native causal CrossEncoder preprocessing exceeded the padded forward budget"
                        )
                    output = self._model.forward(self._move_features(features, self._device))
                    if not isinstance(output, Mapping) or "scores" not in output:
                        raise RuntimeError(_ERR_NATIVE_SCORES)
                    native_scores = self._model.activation_fn(output["scores"])
                    values = self._score_vector(native_scores, len(indices))
                    for index, value in zip(indices, values, strict=True):
                        scores[rows.indices[index]] = value
        return ScoreOutput(scores=scores, input_token_counts=list(rows.counts))

    def _runtime_limit(self, instruction: str | None, options: dict[str, Any] | None) -> int:
        if instruction is not None:
            raise InvalidInputError("Native causal CrossEncoder conditions belong in the query, not instruction")
        opts = options or {}
        if set(opts) - {"max_seq_length"}:
            raise InvalidInputError("Native causal CrossEncoder supports only the max_seq_length runtime option")
        return self._validate_limit(opts.get("max_seq_length", self._max_seq_length), self._max_seq_length)

    @staticmethod
    def _validate_limit(value: object, ceiling: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
            raise InvalidInputError(f"Native causal CrossEncoder max_seq_length must be an integer from 1 to {ceiling}")
        return value

    @staticmethod
    def _parse_pairs(queries: list[Item], docs: list[Item]) -> tuple[tuple[str, str], ...]:
        if len(queries) != len(docs):
            raise InvalidInputError("Native causal CrossEncoder requires equal numbers of queries and documents")
        pairs: list[tuple[str, str]] = []
        for query, document in zip(queries, docs, strict=True):
            pairs.append(
                (NativeCausalCrossEncoderAdapter._text(query), NativeCausalCrossEncoderAdapter._text(document))
            )
        return tuple(pairs)

    @staticmethod
    def _text(item: Item) -> str:
        if (
            not isinstance(item.text, str)
            or item.images is not None
            or item.audio is not None
            or item.video is not None
            or item.document is not None
        ):
            raise InvalidInputError("Native causal CrossEncoder accepts text-only query and document items")
        return item.text

    def _preprocess(self, pairs: list[tuple[str, str]]) -> dict[str, Any]:
        try:
            features = self._model.preprocess(pairs)
        except (ValueError, TypeError, KeyError):
            raise RuntimeError("Native causal CrossEncoder preprocessing failed") from None
        if not isinstance(features, Mapping):
            raise RuntimeError(_ERR_NATIVE_FEATURES)
        return dict(features)

    @staticmethod
    def _read_rows(features: dict[str, Any], size: int, limit: int) -> tuple[tuple[int, ...], ...]:
        ids, mask = features.get("input_ids"), features.get("attention_mask")
        if (
            not isinstance(ids, torch.Tensor)
            or not isinstance(mask, torch.Tensor)
            or ids.ndim != 2
            or ids.shape[0] != size
            or ids.shape[1] == 0
            or ids.dtype not in _INTEGER_DTYPES
            or mask.shape != ids.shape
            or mask.dtype not in _INTEGER_DTYPES | {torch.bool}
            or type(features.get("logits_to_keep")) is not int
            or features.get("logits_to_keep") != 1
            or not torch.all((mask == 0) | (mask == 1)).item()
            or not torch.all(mask[:, -1] == 1).item()
            or not torch.all(mask[:, 1:] >= mask[:, :-1]).item()
            or not torch.all(ids >= 0).item()
        ):
            raise RuntimeError(_ERR_NATIVE_FEATURES)
        counts = mask.sum(dim=1).tolist()
        if any(count > limit for count in counts):
            raise _NativeInputTooLongError(
                f"Native causal CrossEncoder complete input exceeds the {limit}-token context, including template tokens"
            )
        return tuple(tuple(row[attended.bool()].tolist()) for row, attended in zip(ids, mask, strict=True))

    def _admit(self, pairs: tuple[tuple[str, str], ...], limit: int) -> _NativeRows:
        ids = tuple(self._read_rows(self._preprocess([pair]), 1, limit)[0] for pair in pairs)
        return _NativeRows(pairs=pairs, ids=ids, counts=tuple(map(len, ids)), indices=tuple(range(len(pairs))))

    def _chunks(self, rows: _NativeRows) -> list[list[int]]:
        chunks: list[list[int]] = []
        current: list[int] = []
        for index in sorted(range(len(rows.pairs)), key=lambda index: rows.counts[index]):
            if rows.counts[index] > self._forward_token_budget:
                raise RuntimeError("Native causal CrossEncoder row exceeds the private forward budget")
            if current and (len(current) + 1) * rows.counts[index] > self._forward_token_budget:
                chunks.append(current)
                current = []
            current.append(index)
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _score_vector(scores: Any, size: int) -> np.ndarray:
        if not isinstance(scores, torch.Tensor) or not scores.is_floating_point():
            raise RuntimeError(_ERR_NATIVE_SCORES)
        if scores.ndim == 0 and size == 1:
            scores = scores.reshape(1)
        elif scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores.squeeze(1)
        if scores.ndim != 1 or scores.shape[0] != size or not torch.isfinite(scores).all().item():
            raise RuntimeError(_ERR_NATIVE_SCORES)
        values = scores.detach().to(dtype=torch.float32).cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError(_ERR_NATIVE_SCORES)
        return values

    def count_pair_input_tokens(
        self, query: Item, docs: list[Item], *, instruction: str | None = None
    ) -> list[int] | None:
        """Count through the same complete native admission path, without a forward."""
        try:
            self._check_loaded()
            limit = self._runtime_limit(instruction, None)
            pairs = self._parse_pairs([query] * len(docs), docs)
            with self._tokenizer_guard():
                return list(self._admit(pairs, limit).counts)
        except (InvalidInputError, RuntimeError):
            return None

    def count_input_tokens(self, items: list[Item]) -> None:
        """Single text tokenization is not the native causal pair's accounting basis."""
        _ = items
