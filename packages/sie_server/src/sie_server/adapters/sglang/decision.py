"""Selected-label decision scoring through SIE's existing CUDA 13 SGLang engine."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar, cast
from uuid import uuid4

import requests
import torch

from sie_server.adapters._checkpoint import resolve_checkpoint
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._transformers_compat import install_transformers_security_fixes
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.adapters.decision import knowline
from sie_server.adapters.decision.base import MAX_ITEM_TOKENS, DecisionBatchAdapter, Row
from sie_server.adapters.decision.questions import Question
from sie_server.adapters.errors import InputTooLongError
from sie_server.adapters.sglang.cuda13 import SGLangCuda13Adapter
from sie_server.types.inputs import InvalidInputError, Item


class SGLangDecisionAdapter(DecisionBatchAdapter):
    """Reuse the generation engine lifecycle while exposing only typed extraction."""

    spec: ClassVar[AdapterSpec] = AdapterSpec(inputs=("text",), outputs=("json",), unload_fields=("_tokenizer",))
    requires_main_thread = False
    manages_own_load_timeout = True

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        max_seq_length: int = 8192,
        max_forward_tokens: int = 16384,
        structured_noul: bool = True,
        compute_precision: ComputePrecision = "bfloat16",
        revision: str | None = None,
        request_timeout_s: float = 120.0,
        **kwargs: Any,
    ) -> None:
        if type(max_seq_length) is not int or not 1 <= max_seq_length <= 16384:
            raise ValueError("Decision max_seq_length must be between 1 and 16384")
        if type(max_forward_tokens) is not int or not max_seq_length <= max_forward_tokens <= MAX_ITEM_TOKENS:
            raise ValueError("max_forward_tokens must cover one full row and be at most 32768")
        if type(structured_noul) is not bool:
            raise ValueError("structured_noul must be a boolean")
        if (
            type(request_timeout_s) not in (int, float)
            or not math.isfinite(request_timeout_s)
            or request_timeout_s <= 0
        ):
            raise ValueError("request_timeout_s must be positive and finite")
        forbidden = {"trust_remote_code", "grammar_backend", "lora_paths", "lora_revisions", "speculative"}
        if forbidden.intersection(kwargs):
            raise ValueError(f"Decision scoring does not accept {sorted(forbidden.intersection(kwargs))}")
        self._model_name_or_path, self._revision = str(model_name_or_path), revision
        self._compute_precision = compute_precision
        self._max_seq_length, self._max_forward_tokens = max_seq_length, max_forward_tokens
        self._structured_noul, self._request_timeout_s = structured_noul, request_timeout_s
        self._engine = SGLangCuda13Adapter(
            str(model_name_or_path),
            max_seq_length=max_seq_length + 1,
            compute_precision=compute_precision,
            revision=revision,
            trust_remote_code=False,
            grammar_backend=None,
            **kwargs,
        )
        self._tokenizer: Any = None
        self._codes: tuple[str, ...] = ()
        self._code_ids: tuple[int, ...] = ()
        self._session: requests.Session | None = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        from transformers import AutoTokenizer

        if not device.startswith("cuda"):
            raise ValueError("SGLang decision scoring requires CUDA")
        directory = resolve_checkpoint(
            self._model_name_or_path,
            revision=self._revision,
            allow_patterns=(
                "config.json",
                "tokenizer*",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "merges.txt",
                "chat_template*",
            ),
        )
        install_transformers_security_fixes()
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        codes, ids = knowline.labels(tokenizer)
        try:
            self._engine.load(device)
            self._session = requests.Session()
            self._session.trust_env = False
        except BaseException:
            self.unload()
            raise
        self._tokenizer, self._codes, self._code_ids, self._device = tokenizer, codes, ids, device

    def unload(self) -> None:
        try:
            self._engine.unload()
        finally:
            if self._session is not None:
                self._session.close()
            self._session, self._tokenizer, self._device = None, None, None
            self._codes, self._code_ids = (), ()

    def memory_footprint(self) -> int:
        return self._engine.memory_footprint()

    def load_required_memory_bytes(self, *, device_type: str, device_total_bytes: int) -> int | None:
        return self._engine.load_required_memory_bytes(device_type=device_type, device_total_bytes=device_total_bytes)

    def engine_exit_code(self) -> int | None:
        return self._engine.engine_exit_code()

    def _check_decision_loaded(self) -> None:
        if self._tokenizer is None or self._session is None or self._engine.server_url is None:
            raise RuntimeError(ERR_NOT_LOADED)
        self._engine._check_loaded()

    def _parse_questions(self, schema: dict[str, Any] | None) -> tuple[Question, ...]:
        questions = super()._parse_questions(schema)
        if any(len(question.keys) < 2 for question in questions):
            raise InvalidInputError("KnowLine questions require at least two options")
        return questions

    def _item_rows(self, index: int, item: Item, questions: tuple[Question, ...]) -> list[Row]:
        state = self._raw_state(item)
        with self._tokenizer_guard():
            inputs = knowline.prompts(
                self._tokenizer,
                state,
                questions,
                self._codes,
                structured_noul=self._structured_noul,
                max_seq_length=self._max_seq_length,
            )
        if any(len(ids) > self._max_seq_length for ids in inputs) or sum(map(len, inputs)) > MAX_ITEM_TOKENS:
            raise InputTooLongError("Decision rows exceed the token budget; no input was truncated")
        return [
            Row(index, (question,), {"input_ids": torch.tensor(ids, dtype=torch.long)})
            for question, ids in zip(questions, inputs, strict=True)
        ]

    def _forward(self, rows: list[Row]) -> list[list[float]]:
        self._check_decision_loaded()
        session = self._session
        if session is None:
            raise RuntimeError(ERR_NOT_LOADED)
        ids = [list(self._code_ids[: len(row.question.keys)]) for row in rows]
        payload = {
            "rid": [uuid4().hex for _ in rows],
            "input_ids": [row.inputs["input_ids"].tolist() for row in rows],
            "sampling_params": {
                "max_new_tokens": 1,
                "min_new_tokens": 0,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
            "token_ids_logprob": ids,
        }
        try:
            response = session.post(
                f"{self._engine.server_url}/generate", json=payload, timeout=self._request_timeout_s
            )
            response.raise_for_status()
        except requests.RequestException:
            try:
                session.post(
                    f"{self._engine.server_url}/abort_request",
                    json={"rid": payload["rid"]},
                    timeout=min(10.0, self._request_timeout_s),
                ).raise_for_status()
            except requests.RequestException:
                pass
            raise
        results = response.json()
        if not isinstance(results, list) or len(results) != len(rows):
            raise ValueError("SGLang returned the wrong decision batch size")
        distributions: list[list[float]] = []
        for row, selected, result in zip(rows, ids, results, strict=True):
            if not isinstance(result, dict):
                raise ValueError("SGLang returned an invalid decision result")
            meta = cast("dict[str, Any]", result)["meta_info"]
            if meta.get("prompt_tokens") != row.length:
                raise ValueError("SGLang changed a decision row's input token count")
            steps = meta.get("output_token_ids_logprobs")
            if not isinstance(steps, list) or len(steps) != 1:
                raise ValueError("SGLang did not return exactly one decision readout")
            entries = steps[0]
            got = {int(entry[1]): entry[0] for entry in entries}
            if set(got) != set(selected) or len(entries) != len(selected):
                raise ValueError("SGLang did not return every selected label exactly once")
            values = [got[token] for token in selected]
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
                raise ValueError("SGLang returned nonfinite decision label probabilities")
            peak = max(values)
            weights = [math.exp(value - peak) for value in values]
            total = math.fsum(weights)
            probabilities = [weight / total for weight in weights]
            if row.question.kind == "noul":
                probabilities.reverse()
            distributions.append(probabilities)
        return distributions

    def _answer(self, question: Question, probabilities: list[float]) -> dict[str, Any]:
        result = super()._answer(question, probabilities)
        if question.kind != "noul":
            entropy = -math.fsum(value * math.log(value) for value in probabilities if value > 0)
            result["confidence"] = min(1.0, max(0.0, 1.0 - entropy / math.log(len(probabilities))))
        return result
