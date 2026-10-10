from __future__ import annotations

import pytest

from document_to_markdown.config import load_config, select_documents


def test_source_set_covers_four_document_shapes() -> None:
    config = load_config()

    assert [document.slug for document in config.documents] == [
        "nvidia-cfo-commentary",
        "siriuspoint-investor-deck",
        "docling-paper",
        "fema-proof-of-loss",
    ]
    assert config.cluster.model == "docling"
    assert config.cluster.request_timeout_s == 900


def test_select_documents_keeps_requested_order() -> None:
    config = load_config()

    selected = select_documents(
        config,
        ["fema-proof-of-loss", "nvidia-cfo-commentary"],
    )

    assert [document.slug for document in selected] == [
        "fema-proof-of-loss",
        "nvidia-cfo-commentary",
    ]


def test_select_documents_rejects_unknown_slug() -> None:
    with pytest.raises(ValueError, match="Unknown document slug"):
        select_documents(load_config(), ["missing"])


def test_sie_url_takes_precedence_with_legacy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_CLUSTER_URL", "https://legacy.example/prefix/")
    monkeypatch.setenv("SIE_URL", "https://chosen.example/gateway/")
    assert load_config().cluster.url == "https://chosen.example/gateway"

    monkeypatch.delenv("SIE_URL")
    assert load_config().cluster.url == "https://legacy.example/prefix"
