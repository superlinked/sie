from __future__ import annotations

import hashlib
import json

from insurance_claims.config import ROOT, load_config


def test_source_set_uses_public_fema_documents() -> None:
    config = load_config()

    assert [source.slug for source in config.sources] == [
        "nfip-appeal-b8",
        "sfip-dwelling-policy",
    ]
    assert all(source.rights.startswith("U.S. federal government work") for source in config.sources)
    assert config.models.parse == "docling"
    assert config.models.extract == "fastino/gliner2-large-v1"
    assert config.models.rerank == "BAAI/bge-reranker-v2-m3"
    assert config.models.review == "Qwen/Qwen3.5-4B:no-spec"


def test_bundled_sources_match_the_verified_source_manifest() -> None:
    config = load_config()
    manifest_path = ROOT / "verified-run" / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {row["slug"]: row for row in manifest["sources"]}

    assert set(expected) == {source.slug for source in config.sources}
    for source in config.sources:
        assert source.fixture_path is not None
        payload = source.fixture_path.read_bytes()
        assert len(payload) == expected[source.slug]["bytes"]
        assert hashlib.sha256(payload).hexdigest() == expected[source.slug]["sha256"]
