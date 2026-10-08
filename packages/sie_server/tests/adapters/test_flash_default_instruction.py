"""CPU unit tests for query instructions in the extract_texts-based flash adapters.

The flash kernels need CUDA, so these tests stop each adapter right after it
formats its texts and check the exact strings it would tokenize.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
from sie_server.adapters import _utils
from sie_server.adapters._utils import resolve_query_instruction
from sie_server.core.loader import load_model_config
from sie_server.core.runtime_options import merge_runtime_options
from sie_server.types.inputs import Item

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_TEMPLATE = "Instruct: {instruction}\nQuery: {text}"
_DEFAULT = "Given a web search query, retrieve relevant passages that answer the query."

# (module, class, output type) for every flash adapter that formats texts with
# the shared extract_texts helper and resolves its instruction through
# resolve_query_instruction.
_ADAPTERS = [
    ("bert_flash", "BertFlashAdapter", "dense"),
    ("modernbert_flash", "ModernBERTFlashAdapter", "dense"),
    ("nomic_flash", "NomicFlashAdapter", "dense"),
    ("qwen2_flash", "Qwen2FlashAdapter", "dense"),
    ("rope_flash", "RoPEFlashAdapter", "dense"),
    ("gte_sparse_flash", "GTESparseFlashAdapter", "sparse"),
    ("splade_flash.adapter", "SPLADEFlashAdapter", "sparse"),
]


class _FormattedTexts(Exception):  # noqa: N818 - control-flow signal, not an error
    def __init__(self, texts: list[str]) -> None:
        super().__init__(texts)
        self.texts = texts


def _formatted_texts(
    monkeypatch: pytest.MonkeyPatch,
    adapter_spec: tuple[str, str, str],
    texts: list[str],
    *,
    is_query: bool,
    instruction: str | None = None,
    options: dict[str, Any] | None = None,
) -> list[str]:
    module_name, class_name, output_type = adapter_spec
    module = importlib.import_module(f"sie_server.adapters.{module_name}")

    def stop_after_formatting(*args: Any, **kwargs: Any) -> list[str]:
        raise _FormattedTexts(_utils.extract_texts(*args, **kwargs))

    monkeypatch.setattr(module, "extract_texts", stop_after_formatting)
    adapter = getattr(module, class_name)("unused")
    # Satisfy each adapter's loaded check; NomicFlashAdapter keeps _layers, not _model.
    for loaded_attr in ("_model", "_layers", "_tokenizer"):
        if hasattr(adapter, loaded_attr):
            monkeypatch.setattr(adapter, loaded_attr, object())

    with pytest.raises(_FormattedTexts) as formatted:
        adapter.encode(
            [Item(text=text) for text in texts],
            [output_type],
            instruction=instruction,
            is_query=is_query,
            options=options,
        )
    return formatted.value.texts


_ADAPTER_IDS = [class_name for _, class_name, _ in _ADAPTERS]


@pytest.mark.parametrize("adapter_spec", _ADAPTERS, ids=_ADAPTER_IDS)
def test_query_uses_default_instruction(monkeypatch: pytest.MonkeyPatch, adapter_spec: tuple[str, str, str]) -> None:
    texts = _formatted_texts(
        monkeypatch,
        adapter_spec,
        ["what is sie?", "flash attention"],
        is_query=True,
        options={"query_template": _TEMPLATE, "default_instruction": _DEFAULT},
    )

    assert texts == [
        f"Instruct: {_DEFAULT}\nQuery: what is sie?",
        f"Instruct: {_DEFAULT}\nQuery: flash attention",
    ]


@pytest.mark.parametrize("adapter_spec", _ADAPTERS, ids=_ADAPTER_IDS)
def test_explicit_instruction_wins(monkeypatch: pytest.MonkeyPatch, adapter_spec: tuple[str, str, str]) -> None:
    texts = _formatted_texts(
        monkeypatch,
        adapter_spec,
        ["what is sie?"],
        is_query=True,
        instruction="Find related questions",
        options={"query_template": _TEMPLATE, "default_instruction": _DEFAULT},
    )

    assert texts == ["Instruct: Find related questions\nQuery: what is sie?"]


@pytest.mark.parametrize("adapter_spec", _ADAPTERS, ids=_ADAPTER_IDS)
def test_explicit_empty_instruction_is_kept(
    monkeypatch: pytest.MonkeyPatch, adapter_spec: tuple[str, str, str]
) -> None:
    texts = _formatted_texts(
        monkeypatch,
        adapter_spec,
        ["what is sie?"],
        is_query=True,
        instruction="",
        options={"query_template": _TEMPLATE, "default_instruction": _DEFAULT},
    )

    assert texts == ["Instruct: \nQuery: what is sie?"]


@pytest.mark.parametrize("adapter_spec", _ADAPTERS, ids=_ADAPTER_IDS)
def test_documents_get_no_default_instruction(
    monkeypatch: pytest.MonkeyPatch,
    adapter_spec: tuple[str, str, str],
) -> None:
    texts = _formatted_texts(
        monkeypatch,
        adapter_spec,
        ["SIE serves embedding models."],
        is_query=False,
        options={
            "query_template": _TEMPLATE,
            "doc_template": "Instruct: {instruction}\nDocument: {text}",
            "default_instruction": _DEFAULT,
        },
    )

    assert texts == ["Instruct: \nDocument: SIE serves embedding models."]


@pytest.mark.parametrize("adapter_spec", _ADAPTERS, ids=_ADAPTER_IDS)
def test_query_without_default_instruction_keeps_empty_slot(
    monkeypatch: pytest.MonkeyPatch,
    adapter_spec: tuple[str, str, str],
) -> None:
    texts = _formatted_texts(
        monkeypatch,
        adapter_spec,
        ["what is sie?"],
        is_query=True,
        options={"query_template": _TEMPLATE},
    )

    assert texts == ["Instruct: \nQuery: what is sie?"]


def test_stella_400m_query_uses_profile_default_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_model_config(_MODELS_DIR / "NovaSearch__stella_en_400M_v5.yaml")
    assert config.resolve_profile("default").adapter_path == "sie_server.adapters.rope_flash:RoPEFlashAdapter"

    texts = _formatted_texts(
        monkeypatch,
        ("rope_flash", "RoPEFlashAdapter", "dense"),
        ["what is sie?"],
        is_query=True,
        options=merge_runtime_options(config, {"is_query": True}),
    )

    assert texts == [f"Instruct: {_DEFAULT}\nQuery: what is sie?"]


@pytest.mark.parametrize(
    ("instruction", "options", "is_query", "expected"),
    [
        (None, {"default_instruction": "default"}, True, "default"),
        ("explicit", {"default_instruction": "default"}, True, "explicit"),
        ("", {"default_instruction": "default"}, True, ""),
        (None, {"default_instruction": "default"}, False, None),
        ("explicit", {"default_instruction": "default"}, False, "explicit"),
        (None, {}, True, None),
        (None, None, True, None),
    ],
)
def test_resolve_query_instruction(
    instruction: str | None,
    options: dict[str, Any] | None,
    is_query: bool,
    expected: str | None,
) -> None:
    assert resolve_query_instruction(instruction, options, is_query=is_query) == expected
