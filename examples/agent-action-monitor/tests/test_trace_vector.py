"""Tests for SIE-backed similarity search in dusk.trace.vector."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from dusk.config import Config
from dusk.trace import vector
from dusk.trace.models import TraceDecision

DEFAULT_CONFIG = Config()


def _decisions() -> list[TraceDecision]:
    return [
        TraceDecision(
            agent_id="netops-agent",
            action="firewall_rule_change fw-corp-https",
            score=10,
            reasoning="opened port 443 on the corp https rule",
        ),
        TraceDecision(
            agent_id="netops-agent",
            action="role_assignment fw-restricted",
            score=90,
            reasoning="granted owner role on a restricted segment",
        ),
    ]


def _inject_fake_item_type(monkeypatch) -> None:
    fake_types = types.ModuleType("sie_sdk.types")
    fake_types.Item = lambda text=None, id=None, **_kw: {  # type: ignore[attr-defined]  # noqa: A002
        "text": text,
        "id": id,
    }
    monkeypatch.setitem(sys.modules, "sie_sdk.types", fake_types)


def test_find_similar_returns_empty_below_two_decisions() -> None:
    assert vector.find_similar("firewall_rule_change fw-corp-https", "netops-agent", []) == []


def test_find_similar_falls_back_to_ngram_when_sie_sdk_missing(monkeypatch) -> None:
    monkeypatch.setattr(vector, "_sie_client", lambda config: None)
    results = vector.find_similar(
        "firewall_rule_change fw-corp-https", "netops-agent", _decisions()
    )
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, vector.SimilarDecision)


def test_stable_hash_is_deterministic_across_calls() -> None:
    """_stable_hash must not depend on Python's per-process hash randomization.

    A gate restart while running the no-SIE fallback path must not make
    previously recorded embeddings incomparable to freshly computed ones.
    """
    assert vector._stable_hash("firewall_rule_change") == vector._stable_hash(
        "firewall_rule_change"
    )
    assert vector._stable_hash("a") != vector._stable_hash("b")


def test_stable_hash_does_not_use_builtin_hash() -> None:
    """Guards against a regression back to Python's randomized hash()."""
    token = "netops-agent"
    assert vector._stable_hash(token) != hash(token) % (2**64)


def test_ngram_fallback_is_deterministic_across_calls() -> None:
    assert vector._ngram_fallback("firewall_rule_change fw-corp-https") == vector._ngram_fallback(
        "firewall_rule_change fw-corp-https"
    )


def test_sie_encode_uses_sdk_dense_vector_when_available(monkeypatch) -> None:
    fake_client = MagicMock()
    fake_client.encode.return_value = {"dense": [1.0, 0.0, 0.0]}
    monkeypatch.setattr(vector, "_sie_client", lambda config: fake_client)
    _inject_fake_item_type(monkeypatch)

    embedding = vector.sie_encode("hello world")

    assert embedding == [1.0, 0.0, 0.0]
    fake_client.encode.assert_called_once()
    assert fake_client.encode.call_args[0][0] == DEFAULT_CONFIG.sie_encode_model


def test_sie_encode_returns_none_and_does_not_raise_on_sdk_error(monkeypatch) -> None:
    fake_client = MagicMock()
    fake_client.encode.side_effect = RuntimeError("connection refused")
    monkeypatch.setattr(vector, "_sie_client", lambda config: fake_client)
    _inject_fake_item_type(monkeypatch)

    assert vector.sie_encode("hello world") is None


def test_sie_client_returns_none_when_sie_sdk_not_installed(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sie_sdk", None)
    assert vector._sie_client(DEFAULT_CONFIG) is None


def test_sie_client_passes_configured_timeout(monkeypatch) -> None:
    """sie_timeout_ms must actually reach the SDK client, not just live in Config."""
    captured: dict[str, object] = {}

    class FakeSIEClient:
        def __init__(self, base_url: str, **kwargs: object) -> None:
            captured["base_url"] = base_url
            captured.update(kwargs)

    fake_sie_sdk = types.ModuleType("sie_sdk")
    fake_sie_sdk.SIEClient = FakeSIEClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sie_sdk", fake_sie_sdk)

    config = Config(sie_endpoint="http://sie:8080", sie_timeout_ms=5000)
    client = vector._sie_client(config)

    assert isinstance(client, FakeSIEClient)
    assert captured["timeout_s"] == 5.0
    assert captured["base_url"] == "http://sie:8080"


def test_sie_score_returns_none_without_candidates() -> None:
    assert vector.sie_score("query", []) is None


def test_sie_score_returns_none_when_sie_sdk_missing(monkeypatch) -> None:
    monkeypatch.setattr(vector, "_sie_client", lambda config: None)
    assert vector.sie_score("query", ["a", "b"]) is None


def test_sie_score_preserves_input_order(monkeypatch) -> None:
    fake_client = MagicMock()
    # SDK returns entries out of input order (by rank); sie_score must map
    # them back by item_id to the same order the candidates were given in.
    fake_client.score.return_value = {
        "scores": [
            {"item_id": "1", "score": 0.9, "rank": 0},
            {"item_id": "0", "score": 0.2, "rank": 1},
        ]
    }
    monkeypatch.setattr(vector, "_sie_client", lambda config: fake_client)
    _inject_fake_item_type(monkeypatch)

    scores = vector.sie_score("query", ["candidate-a", "candidate-b"])

    # Raw logits (0.2, 0.9) pass through sigmoid before returning, so the
    # order is preserved but the values are calibrated probabilities.
    assert scores is not None
    assert scores == [pytest.approx(vector._sigmoid(0.2)), pytest.approx(vector._sigmoid(0.9))]
    assert scores[0] < scores[1]
    fake_client.score.assert_called_once()
    assert fake_client.score.call_args[0][0] == DEFAULT_CONFIG.sie_score_model


def test_sie_score_returns_none_and_does_not_raise_on_sdk_error(monkeypatch) -> None:
    fake_client = MagicMock()
    fake_client.score.side_effect = RuntimeError("connection refused")
    monkeypatch.setattr(vector, "_sie_client", lambda config: fake_client)
    _inject_fake_item_type(monkeypatch)

    assert vector.sie_score("query", ["a", "b"]) is None


def test_sie_extract_returns_empty_when_sie_sdk_missing(monkeypatch) -> None:
    monkeypatch.setattr(vector, "_sie_client", lambda config: None)
    assert vector.sie_extract("granted owner role") == []


def test_sie_extract_returns_entity_texts_when_available(monkeypatch) -> None:
    fake_client = MagicMock()
    fake_client.extract.return_value = {
        "entities": [
            {"text": "administrator", "label": "role", "score": 0.9},
            {"text": "0.0.0.0", "label": "resource", "score": 0.8},
        ]
    }
    monkeypatch.setattr(vector, "_sie_client", lambda config: fake_client)
    _inject_fake_item_type(monkeypatch)

    terms = vector.sie_extract("grant administrator on 0.0.0.0")

    assert [t.text for t in terms] == ["administrator", "0.0.0.0"]
    assert [t.label for t in terms] == ["role", "resource"]
    assert [t.score for t in terms] == [0.9, 0.8]
    fake_client.extract.assert_called_once()
    assert fake_client.extract.call_args[0][0] == DEFAULT_CONFIG.sie_extract_model
    assert fake_client.extract.call_args[1]["labels"] == vector.DEFAULT_EXTRACT_LABELS


def test_sie_extract_returns_empty_and_does_not_raise_on_sdk_error(monkeypatch) -> None:
    fake_client = MagicMock()
    fake_client.extract.side_effect = RuntimeError("connection refused")
    monkeypatch.setattr(vector, "_sie_client", lambda config: fake_client)
    _inject_fake_item_type(monkeypatch)

    assert vector.sie_extract("grant administrator") == []


def test_find_similar_uses_sie_encode_when_available(monkeypatch) -> None:
    calls: list[str] = []

    def fake_encode(text: str, config: Config | None = None) -> list[float]:
        calls.append(text)
        return [1.0, 0.0] if "fw-corp-https" in text else [0.0, 1.0]

    monkeypatch.setattr(vector, "sie_encode", fake_encode)
    results = vector.find_similar(
        "firewall_rule_change fw-corp-https", "netops-agent", _decisions()
    )
    assert calls
    assert all(isinstance(r, vector.SimilarDecision) for r in results)


def test_find_similar_reranks_shortlist_with_sie_score(monkeypatch) -> None:
    """The rerank pass can override the cosine-similarity order of the shortlist."""
    decisions = [
        TraceDecision(agent_id="netops-agent", action="a", score=10, reasoning="r"),
        TraceDecision(agent_id="netops-agent", action="b", score=20, reasoning="r"),
        TraceDecision(agent_id="netops-agent", action="c", score=30, reasoning="r"),
    ]
    monkeypatch.setattr(vector, "sie_encode", lambda text, config=None: [1.0, 0.0])

    def fake_score(query: str, candidates: list[str]) -> list[float]:
        # Same order as candidates: force the last one to the front.
        return [0.1, 0.2, 0.9][: len(candidates)]

    monkeypatch.setattr(vector, "sie_score", fake_score)

    results = vector.find_similar("query-action", "netops-agent", decisions, top_k=3)

    assert [r.action for r in results] == ["c", "b", "a"]


def test_similar_decision_uses_the_recorded_verdict_not_a_score_guess() -> None:
    """Regression test: SimilarDecision.verdict must come from TraceDecision.verdict,
    not be reconstructed from a hardcoded score cutoff decoupled from
    gate_block_threshold and collapsing WOULD-BLOCK/BLOCK into one label."""
    decisions = [
        TraceDecision(
            agent_id="netops-agent", action="a", score=95, reasoning="r", verdict="WOULD-BLOCK"
        ),
        TraceDecision(
            agent_id="netops-agent", action="b", score=10, reasoning="r", verdict="ALLOW"
        ),
    ]

    scored = list(zip([0.9, 0.8], decisions, strict=True))
    results = vector._rank_candidates("query", scored, top_k=2)

    verdict_by_action = {r.action: r.verdict for r in results}
    assert verdict_by_action["a"] == "WOULD-BLOCK"
    assert verdict_by_action["b"] == "ALLOW"


def test_similar_decision_falls_back_for_legacy_decision_with_no_verdict() -> None:
    """A TraceDecision recorded before the verdict field existed has verdict=='' --
    must fall back to a labeled default, not silently claim ALLOW."""
    decisions = [TraceDecision(agent_id="netops-agent", action="a", score=50, reasoning="r")]
    scored = [(0.9, decisions[0])]

    results = vector._rank_candidates("query", scored, top_k=1)

    assert results[0].verdict == vector._UNKNOWN_VERDICT_FALLBACK
