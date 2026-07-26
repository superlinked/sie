from __future__ import annotations

from insurance_claims.config import load_config


def test_source_set_uses_public_fema_documents() -> None:
    config = load_config()

    assert [source.slug for source in config.sources] == [
        "nfip-appeal-b8",
        "sfip-dwelling-policy",
    ]
    assert all(
        source.rights.startswith("U.S. federal government work")
        for source in config.sources
    )
    assert config.models.parse == "docling"
    assert config.models.extract == "fastino/gliner2-large-v1"
    assert config.models.rerank == "BAAI/bge-reranker-v2-m3"
    assert config.models.review == "Qwen/Qwen3.5-4B:no-spec"
