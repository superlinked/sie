"""CPU unit tests for Qwen2FlashAdapter query/document text formatting.

The flash kernels need CUDA, so these tests stub the transformer stack and
capture the exact text the adapter hands to its tokenizer. The options are
built from the shipped model YAMLs through the same merge the HTTP and queue
paths use, so the tests pin what a real request to these models embeds.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from sie_server.adapters.qwen2_flash import Qwen2FlashAdapter
from sie_server.config.model import ModelConfig
from sie_server.core.loader import load_model_config
from sie_server.core.runtime_options import merge_runtime_options
from sie_server.types.inputs import Item

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_ADAPTER_PATH = "sie_server.adapters.qwen2_flash:Qwen2FlashAdapter"
_QWEN3_8B = "Qwen__Qwen3-Embedding-8B.yaml"
_QWEN3_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def _qwen2_flash_model_files() -> list[str]:
    return sorted(path.name for path in _MODELS_DIR.glob("*.yaml") if _ADAPTER_PATH in path.read_text())


class _RecordingTokenizer:
    """Character-level stand-in that records every text it tokenizes."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, texts: list[str], **_: Any) -> dict[str, list[list[int]]]:
        self.texts.extend(texts)
        return {"input_ids": [[0] * len(text) for text in texts]}


def _config(model_file: str) -> ModelConfig:
    return load_model_config(_MODELS_DIR / model_file)


def _adapter(monkeypatch: pytest.MonkeyPatch) -> tuple[Qwen2FlashAdapter, _RecordingTokenizer]:
    adapter = Qwen2FlashAdapter("unused", pooling="last", causal=True)
    tokenizer = _RecordingTokenizer()
    monkeypatch.setattr(adapter, "_tokenizer", tokenizer)
    monkeypatch.setattr(
        adapter,
        "_model",
        SimpleNamespace(embed_tokens=torch.nn.Embedding(1, 4), norm=torch.nn.Identity()),
    )
    monkeypatch.setattr(adapter, "_device", "cpu")
    monkeypatch.setattr(adapter, "_dense_dim", 4)
    monkeypatch.setattr(adapter, "_run_transformer_flash", lambda hidden, *_args: hidden)
    return adapter, tokenizer


def _encode(
    monkeypatch: pytest.MonkeyPatch,
    model_file: str,
    texts: list[str],
    *,
    is_query: bool,
    instruction: str | None = None,
    request_options: dict[str, Any] | None = None,
) -> list[str]:
    adapter, tokenizer = _adapter(monkeypatch)
    options = merge_runtime_options(_config(model_file), {"is_query": is_query, **(request_options or {})})

    output = adapter.encode(
        [Item(text=text) for text in texts],
        ["dense"],
        instruction=instruction,
        is_query=is_query,
        options=options,
    )

    # Metered token counts must include the template and instruction tokens.
    assert output.extra["input_token_counts"] == [len(text) for text in tokenizer.texts]
    return tokenizer.texts


def test_shipped_qwen2_flash_models_configure_a_default_instruction() -> None:
    model_files = _qwen2_flash_model_files()

    assert _QWEN3_8B in model_files
    for model_file in model_files:
        runtime = _config(model_file).resolve_profile("default").runtime
        assert "{instruction}" in runtime["query_template"], model_file
        assert runtime["default_instruction"], model_file


def test_qwen3_8b_query_uses_profile_default_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    texts = _encode(monkeypatch, _QWEN3_8B, ["what is sie?", "flash attention"], is_query=True)

    assert texts == [
        f"Instruct: {_QWEN3_DEFAULT_INSTRUCTION}\nQuery:what is sie?",
        f"Instruct: {_QWEN3_DEFAULT_INSTRUCTION}\nQuery:flash attention",
    ]


@pytest.mark.parametrize("model_file", _qwen2_flash_model_files())
def test_every_qwen2_flash_model_applies_its_default_instruction(
    monkeypatch: pytest.MonkeyPatch,
    model_file: str,
) -> None:
    runtime = _config(model_file).resolve_profile("default").runtime

    texts = _encode(monkeypatch, model_file, ["query text"], is_query=True)

    assert texts == [runtime["query_template"].format(instruction=runtime["default_instruction"], text="query text")]


def test_explicit_instruction_wins_over_default(monkeypatch: pytest.MonkeyPatch) -> None:
    texts = _encode(monkeypatch, _QWEN3_8B, ["def add(a, b)"], is_query=True, instruction="Retrieve matching code")

    assert texts == ["Instruct: Retrieve matching code\nQuery:def add(a, b)"]


def test_explicit_empty_instruction_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    texts = _encode(monkeypatch, _QWEN3_8B, ["what is sie?"], is_query=True, instruction="")

    assert texts == ["Instruct: \nQuery:what is sie?"]


def test_request_default_instruction_overrides_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    texts = _encode(
        monkeypatch,
        _QWEN3_8B,
        ["what is sie?"],
        is_query=True,
        request_options={"default_instruction": "Find related questions"},
    )

    assert texts == ["Instruct: Find related questions\nQuery:what is sie?"]


def test_documents_get_no_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    texts = _encode(monkeypatch, _QWEN3_8B, ["SIE serves embedding models."], is_query=False)

    assert texts == ["SIE serves embedding models."]


def test_query_without_any_instruction_keeps_empty_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, tokenizer = _adapter(monkeypatch)

    adapter.encode(
        [Item(text="what is sie?")],
        ["dense"],
        is_query=True,
        options={"query_template": "Instruct: {instruction}\nQuery:{text}"},
    )

    assert tokenizer.texts == ["Instruct: \nQuery:what is sie?"]
