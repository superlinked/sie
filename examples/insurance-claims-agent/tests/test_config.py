from __future__ import annotations

from insurance_claims.config import load_claim, load_config


def test_source_set_uses_public_government_and_photo_sources() -> None:
    config = load_config()

    assert [source.slug for source in config.sources] == [
        "nfip-proof-of-loss",
        "sfip-dwelling-policy",
        "flooded-house-interior",
    ]
    assert config.models.parse == "docling"
    assert config.models.vision == "IDEA-Research/grounding-dino-tiny"
    assert config.models.review == "Qwen/Qwen3.5-4B:no-spec"


def test_claim_fixture_is_explicitly_fictional() -> None:
    claim = load_claim()

    assert claim["fictional"] is True
    assert "SAMPLE" in claim["claim_number"]
    assert claim["contact"]["email"].endswith(".invalid")
