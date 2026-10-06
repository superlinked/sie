"""Privacy Filter extraction through the pinned official OPF runtime.

The checkpoint's calibrated Viterbi decoder supplies source character spans.
The native runtime has no span confidence scores. The HTTP API's existing
``score=1.0`` default represents an unscored detection; ``data`` explicitly
reports that confidence scores are unavailable.
"""

from __future__ import annotations

import json
import math
from itertools import chain
from pathlib import Path
from typing import Any, ClassVar

import torch
from huggingface_hub import snapshot_download

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.responses import Entity

PRIVACY_LABELS = (
    "account_number",
    "private_address",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "private_date",
    "secret",
)
_BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)
_ENTRY_KEY = "transition_bias_background_to_start"
_NATIVE_FILES = ("config.json", "model.safetensors", "dtypes.json", "viterbi_calibration.json")
_DEFAULT_CONTEXT = 128000
_MAX_BIAS_DELTA = 10.0
_MAX_NATIVE_TOKEN_BYTES = 128


class _StrictDecoder:
    """Prevent the official predictor's argmax fallback on an invalid decode."""

    def __init__(self, decoder: Any) -> None:
        self._decoder = decoder

    def decode(self, scores: torch.Tensor) -> list[int]:
        if scores.ndim != 2 or scores.shape[1] != 33 or not bool(torch.isfinite(scores).all()):
            raise ValueError("Privacy Filter returned invalid token scores")
        # The native float32 dynamic program can overflow and internally fall
        # back to argmax even for finite scores. Bound every possible path sum.
        limit = torch.finfo(torch.float32).max / (4 * (scores.shape[0] + 1))
        if (scores.numel() and bool(scores.abs().max() > limit)) or any(
            abs(getattr(self._decoder, key, 0.0)) > limit for key in _BIAS_KEYS
        ):
            raise ValueError("Privacy Filter token scores exceed the safe Viterbi range")
        labels = self._decoder.decode(scores)
        if len(labels) != scores.shape[0] or any(
            not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < 33 for label in labels
        ):
            raise ValueError("Privacy Filter Viterbi decoder returned invalid token labels")
        self._check_boundaries(labels)
        return labels

    def _check_boundaries(self, labels: list[int]) -> None:
        info = self._decoder.label_info
        active: int | None = None
        for label in labels:
            if label not in info.token_boundary_tags or label not in info.token_to_span_label:
                raise ValueError("Privacy Filter Viterbi decoder returned unknown BIOES labels")
            tag = info.token_boundary_tags[label]
            category = info.token_to_span_label[label]
            if active is None:
                if tag == "B":
                    active = category
                elif tag not in {None, "S"}:
                    raise ValueError("Privacy Filter Viterbi decoder returned an invalid BIOES path")
            elif tag not in {"I", "E"} or category != active:
                raise ValueError("Privacy Filter Viterbi decoder returned an invalid BIOES path")
            elif tag == "E":
                active = None
        if active is not None:
            raise ValueError("Privacy Filter Viterbi decoder returned an incomplete BIOES path")


class PrivacyFilterAdapter(BaseAdapter):
    """Extract the eight trained privacy categories with checkpoint calibration.

    ``labels`` may select a subset of the trained categories; it does not change
    the model's policy. ``options.span_entry_bias`` adds a finite delta in
    [-10, 10] to the checkpoint's background-to-span transition bias. Every
    other calibrated transition is preserved. One complete document must fit
    ``max_seq_length``; longer documents return ``INPUT_TOO_LONG`` without a
    forward pass. Usage counts the exact native input tokens, excluding failed
    items, and never counts a truncated prefix as a complete document.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=("_model", "_runtime", "_biases"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        max_seq_length: int | None = None,
        checkpoint_subdir: str = "original",
        trim_span_whitespace: bool = True,
        compute_precision: ComputePrecision = "bfloat16",
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        window = _DEFAULT_CONTEXT if max_seq_length is None else max_seq_length
        if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
            raise ValueError("Privacy Filter max_seq_length must be a positive integer")
        if checkpoint_subdir not in {"", "original"}:
            raise ValueError("Privacy Filter checkpoint_subdir must be 'original' or empty for native local weights")
        if not isinstance(trim_span_whitespace, bool):
            raise ValueError("Privacy Filter trim_span_whitespace must be boolean")
        if compute_precision not in {"bfloat16", "float32"}:
            raise ValueError("Privacy Filter supports bfloat16 or float32 compute precision")
        self._model_name_or_path = str(model_name_or_path)
        self._revision = revision
        self._window = window
        self._chars_per_token_limit = _MAX_NATIVE_TOKEN_BYTES
        self._checkpoint_subdir = checkpoint_subdir
        self._trim_span_whitespace = trim_span_whitespace
        self._compute_precision = compute_precision
        self._model: Any = None
        self._runtime: Any = None
        self._biases: dict[str, float] | None = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        try:
            from opf._core.decoding import resolve_viterbi_biases_from_calibration_path  # ty:ignore[unresolved-import]
            from opf._core.runtime import load_inference_runtime  # ty:ignore[unresolved-import]
        except ImportError as exc:
            raise RuntimeError(
                "Privacy Filter requires the official pinned opf runtime in the transformers5 bundle"
            ) from exc

        root = Path(self._model_name_or_path)
        if not root.is_dir():
            prefix = f"{self._checkpoint_subdir}/" if self._checkpoint_subdir else ""
            root = Path(
                snapshot_download(
                    repo_id=self._model_name_or_path,
                    revision=self._revision,
                    allow_patterns=[f"{prefix}{name}" for name in _NATIVE_FILES],
                )
            )
        checkpoint = root / self._checkpoint_subdir
        for name in _NATIVE_FILES:
            if not (checkpoint / name).is_file():
                raise ValueError(f"Privacy Filter native checkpoint is missing {name}")
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("model_type") != "privacy_filter":
            raise ValueError("Privacy Filter requires the original native checkpoint configuration")
        if config.get("encoding") != "o200k_base":
            raise ValueError("Privacy Filter requires the native o200k_base encoding")
        if self._compute_precision == "bfloat16" and config.get("param_dtype") != "bfloat16":
            raise ValueError("Privacy Filter bfloat16 profile requires the native bfloat16 checkpoint")
        capacity = config.get("max_position_embeddings")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or not 0 < self._window <= capacity:
            raise ValueError("Privacy Filter max_seq_length exceeds the native checkpoint capacity")
        biases = resolve_viterbi_biases_from_calibration_path(str(checkpoint / "viterbi_calibration.json"))
        if set(biases) != set(_BIAS_KEYS) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in biases.values()
        ):
            raise ValueError("Privacy Filter calibration must contain six finite transition biases")
        runtime = load_inference_runtime(
            checkpoint=str(checkpoint),
            device_name=device,
            n_ctx_override=self._window,
            trim_span_whitespace=self._trim_span_whitespace,
            discard_overlapping_predicted_spans=False,
            output_mode="typed",
        )
        if set(runtime.label_info.span_class_names) != {"O", *PRIVACY_LABELS}:
            raise ValueError("Privacy Filter checkpoint has an unsupported privacy label taxonomy")
        if runtime.n_ctx != self._window:
            raise ValueError("Privacy Filter runtime did not apply the configured context length")
        token_bytes = self._maximum_token_bytes(runtime.encoding)
        # The native BF16 model intentionally keeps sinks, norms and rotary
        # tables in FP32. A blanket BF16 cast changes those checkpoint values.
        if self._compute_precision == "float32":
            runtime.model.to(dtype=torch.float32)
        self._runtime = runtime
        self._model = runtime.model
        self._biases = {key: float(value) for key, value in biases.items()}
        self._chars_per_token_limit = token_bytes
        self._device = device

    def extract(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        prepared_items: list[Any] | None = None,
    ) -> ExtractOutput:
        self._check_loaded()
        if output_schema is not None or instruction is not None:
            raise InvalidInputError(
                "Privacy Filter uses its trained privacy policy, not a request schema or instruction"
            )
        selected = self._selected_labels(labels)
        opts = options or {}
        if set(opts) - {"span_entry_bias"}:
            raise InvalidInputError("Privacy Filter only supports the span_entry_bias runtime option")
        delta = self._bias_delta(opts.get("span_entry_bias", 0.0))
        decoder = self._decoder(delta)
        from opf._core.runtime import predict_text  # ty:ignore[unresolved-import]

        entities: list[list[Entity]] = []
        errors: list[ExtractItemError | None] = []
        counts: list[int] = []
        for item in items:
            text = item.text
            if text is None:
                raise InvalidInputError("Privacy Filter requires text input")
            if len(text) > self._window * self._chars_per_token_limit:
                entities.append([])
                errors.append(
                    ExtractItemError(code="INPUT_TOO_LONG", message="Privacy Filter input exceeds the text limit")
                )
                counts.append(0)
                continue
            tokens = self._runtime.encoding.encode(text, allowed_special="all")
            if len(tokens) > self._window:
                entities.append([])
                errors.append(
                    ExtractItemError(
                        code="INPUT_TOO_LONG",
                        message="Privacy Filter requires the entire document to fit the configured token window",
                    )
                )
                counts.append(0)
                continue
            if self._runtime.encoding.decode(tokens) != text:
                entities.append([])
                errors.append(
                    ExtractItemError(
                        code="INVALID_INPUT", message="Privacy Filter tokenizer cannot preserve the source text"
                    )
                )
                counts.append(0)
                continue
            try:
                prediction = predict_text(self._runtime, text, decoder=decoder)
                if prediction.decoded_mismatch or prediction.text != text:
                    raise ValueError("Privacy Filter returned spans in a different source text")
                found = self._entities(prediction.spans, text, selected)
            except ValueError:
                entities.append([])
                errors.append(
                    ExtractItemError(
                        code="INFERENCE_ERROR", message="Privacy Filter returned invalid source spans or token scores"
                    )
                )
                counts.append(0)
                continue
            entities.append(found)
            errors.append(None)
            counts.append(len(tokens))
        return ExtractOutput(
            entities=entities,
            data=[{"confidence_scores_available": False, "decoder": "viterbi"} for _ in items],
            errors=errors if any(errors) else None,
            input_token_counts=counts,
        )

    def _decoder(self, delta: float) -> _StrictDecoder:
        from opf._core.decoding import ViterbiCRFDecoder  # ty:ignore[unresolved-import]

        if self._biases is None:
            raise RuntimeError("Privacy Filter calibration is not loaded")
        biases = dict(self._biases)
        biases[_ENTRY_KEY] += delta
        return _StrictDecoder(ViterbiCRFDecoder(label_info=self._runtime.label_info, **biases))

    @staticmethod
    def _maximum_token_bytes(encoding: Any) -> int:
        """A safe character bound for any complete input fitting this tokenizer.

        UTF-8 characters take at least one byte. Every decoded native token has
        at most this many bytes, including special tokens accepted by OPF. A
        character bound based on a typical token length would reject valid
        compressed inputs, such as a single token of 128 spaces.
        """
        ranks = getattr(encoding, "_mergeable_ranks", None)
        specials = getattr(encoding, "_special_tokens", None)
        if not isinstance(ranks, dict) or not ranks or not isinstance(specials, dict):
            raise ValueError("Privacy Filter cannot verify the native tokenizer vocabulary")
        if any(not isinstance(token, bytes) for token in ranks) or any(
            not isinstance(token, str) for token in specials
        ):
            raise ValueError("Privacy Filter returned an invalid tokenizer vocabulary")
        return max(chain((len(token) for token in ranks), (len(token.encode("utf-8")) for token in specials)))

    @staticmethod
    def _selected_labels(labels: list[str] | None) -> set[str]:
        if labels is None:
            return set(PRIVACY_LABELS)
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or label not in PRIVACY_LABELS for label in labels)
        ):
            raise InvalidInputError("Privacy Filter labels must be a non-empty subset of its eight trained categories")
        if len(labels) != len(set(labels)):
            raise InvalidInputError("Privacy Filter labels must be unique")
        return set(labels)

    @staticmethod
    def _bias_delta(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidInputError("Privacy Filter span_entry_bias must be a finite number in [-10, 10]")
        try:
            delta = float(value)
        except OverflowError as exc:
            raise InvalidInputError("Privacy Filter span_entry_bias must be a finite number in [-10, 10]") from exc
        if not math.isfinite(delta) or abs(delta) > _MAX_BIAS_DELTA:
            raise InvalidInputError("Privacy Filter span_entry_bias must be a finite number in [-10, 10]")
        return delta

    @staticmethod
    def _entities(spans: Any, text: str, selected: set[str]) -> list[Entity]:
        result: list[Entity] = []
        for span in spans:
            start, end = span.start, span.end
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or not 0 <= start < end <= len(text)
                or span.text != text[start:end]
                or span.label not in PRIVACY_LABELS
            ):
                raise ValueError("Privacy Filter returned an invalid source span")
            if span.label in selected:
                result.append(Entity(text=span.text, label=span.label, start=start, end=end))
        result.sort(key=lambda entity: (entity["start"], entity["end"], entity["label"]))
        return result

    def extract_item_costs(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[int] | None:
        return [min(len(item.text or ""), self._window * self._chars_per_token_limit) for item in items]
