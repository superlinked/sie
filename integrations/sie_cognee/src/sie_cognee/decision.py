"""Bounded evidence and validated native SIE choice probabilities."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sie_sdk import SIEClient
from sie_sdk.types import Item

from sie_cognee.retrieval import SourceChunk


class EvidenceOverflowError(ValueError):
    """Evidence needs explicit selection or a larger model, never truncation."""


@dataclass(frozen=True)
class Decision:
    choice: str
    probabilities: dict[str, float]


def evidence_packet(query: str, chunks: Sequence[SourceChunk], *, max_chars: int) -> str:
    """Render exactly the evidence sent; max_chars is not a tokenizer budget."""
    if not query.strip() or not chunks:
        raise ValueError("A query and at least one source chunk are required")
    packet = f"Question: {query}\n" + "\n".join(f"[{chunk.chunk_id}] {chunk.text}" for chunk in chunks)
    _check_size(packet, max_chars)
    return packet


def _check_size(text: str, max_chars: int) -> None:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if len(text) > max_chars:
        raise EvidenceOverflowError("Evidence exceeds max_chars; select fewer sources or request review")


def decide(
    client: SIEClient,
    packet: str,
    *,
    model: str,
    choices: Mapping[str, str],
    instructions: str,
    max_chars: int,
    question_id: str = "action",
    provision_timeout_s: float = 180,
) -> Decision:
    """Return model preferences, not guarantees; server token overflow is an error."""
    if not {"review", "unknown"}.intersection(choices):
        raise ValueError("Choices must include review or unknown")
    if not packet.strip() or not instructions.strip() or not question_id.strip():
        raise ValueError("packet, instructions and question_id must be nonempty")
    if any(not label.strip() or not criterion.strip() for label, criterion in choices.items()):
        raise ValueError("Choice labels and criteria must be nonempty")
    _check_size(packet, max_chars)
    reply = client.extract(
        model,
        Item(id="decision", text=packet),
        output_schema={question_id: {"type": "choice", "instructions": instructions, "criteria": dict(choices)}},
        options={"overflow_policy": "error"},
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )
    data = reply.get("data") if isinstance(reply, dict) else None
    answer = data.get(question_id) if isinstance(data, dict) else None
    if not isinstance(answer, dict):
        raise ValueError("Decision response is missing its question answer")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(choices):
        raise ValueError("Decision must return probabilities for exactly the requested choices")
    validated = {}
    for label, probability in probabilities.items():
        if (
            isinstance(probability, bool)
            or not isinstance(probability, int | float)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("Decision probabilities must be finite numbers in [0, 1]")
        validated[label] = float(probability)
    if not math.isclose(sum(validated.values()), 1.0, abs_tol=1e-4):
        raise ValueError("Choice probabilities must sum to one")
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in choices or validated[choice] != max(validated.values()):
        raise ValueError("Decision choice must match a highest probability label")
    return Decision(choice, validated)
