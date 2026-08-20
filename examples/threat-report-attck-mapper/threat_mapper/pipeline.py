from __future__ import annotations

import hashlib
import re
import time
from typing import Any

import numpy as np
from sie_sdk.types import Item

from .models import BehaviorEvidence, CandidateScore, MappingDecision, Technique
from .sie import SIEClientProtocol, jsonable, parse_generated_json, request_record

ENTITY_LABELS = [
    "threat actor",
    "malware",
    "security tool",
    "credential",
    "authentication token",
    "network protocol",
    "command or script",
    "cloud service",
]

BEHAVIOR_SCHEMA = {
    "type": "object",
    "properties": {
        "behaviors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["quote", "summary"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["behaviors"],
    "additionalProperties": False,
}


def verification_schema(candidate_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selected_index": {"type": "integer", "minimum": -1, "maximum": candidate_count - 1},
            "support": {"type": "string", "enum": ["supported", "ambiguous", "unsupported"]},
            "evidence_quote": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["selected_index", "support", "evidence_quote", "rationale"],
        "additionalProperties": False,
    }


def _ground_quote(source: str, quote: str) -> tuple[str, int, int] | None:
    quote = quote.strip()
    if not quote:
        return None
    exact = source.find(quote)
    if exact >= 0:
        return source[exact : exact + len(quote)], exact, exact + len(quote)
    tokens = quote.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, source)
    if match is None:
        return None
    return match.group(), match.start(), match.end()


def split_report(text: str, max_characters: int) -> list[tuple[int, int]]:
    if max_characters < 1:
        raise ValueError("max_characters must be positive")

    paragraph_spans: list[tuple[int, int]] = []
    cursor = 0
    for separator in re.finditer(r"(?:\r?\n[ \t]*){2,}", text):
        start, end = cursor, separator.start()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            paragraph_spans.append((start, end))
        cursor = separator.end()

    start, end = cursor, len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        paragraph_spans.append((start, end))

    if not paragraph_spans:
        return [(0, min(len(text), max_characters))]

    chunks: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end = 0
    for paragraph_start, paragraph_end in paragraph_spans:
        if paragraph_end - paragraph_start > max_characters:
            if current_start is not None:
                chunks.append((current_start, current_end))
                current_start = None
            chunks.extend(
                (chunk_start, min(chunk_start + max_characters, paragraph_end))
                for chunk_start in range(paragraph_start, paragraph_end, max_characters)
            )
            continue
        if current_start is None:
            current_start, current_end = paragraph_start, paragraph_end
        elif paragraph_end - current_start <= max_characters:
            current_end = paragraph_end
        else:
            chunks.append((current_start, current_end))
            current_start, current_end = paragraph_start, paragraph_end
    if current_start is not None:
        chunks.append((current_start, current_end))
    return chunks


def extract_behaviors(
    client: SIEClientProtocol,
    model: str,
    report_text: str,
    *,
    max_behaviors: int,
    chunk_characters: int,
    provision_timeout_s: float,
) -> tuple[list[BehaviorEvidence], list[dict[str, Any]]]:
    behaviors: list[BehaviorEvidence] = []
    calls: list[dict[str, Any]] = []
    seen_quotes: set[str] = set()
    for chunk_index, (chunk_start, chunk_end) in enumerate(split_report(report_text, chunk_characters)):
        chunk = report_text[chunk_start:chunk_end]
        prompt = (
            "You review a cyber threat report. Extract concrete adversary behaviors that can be mapped to MITRE "
            "ATT&CK Enterprise techniques. Copy each quote exactly from the report chunk. Skip product advice, "
            "background definitions, and unsupported guesses. Keep the shortest quote that still states the behavior.\n\n"
            f"REPORT CHUNK\n{chunk}"
        )
        started = time.perf_counter()
        response = client.generate(
            model,
            prompt,
            max_new_tokens=1800,
            temperature=0,
            grammar={"json_schema": BEHAVIOR_SCHEMA, "label": "threat_behaviors", "strict": True},
            wait_for_capacity=True,
            provision_timeout_s=provision_timeout_s,
        )
        calls.append(
            request_record(
                f"behavior_extract_{chunk_index}",
                model,
                response,
                (time.perf_counter() - started) * 1000,
                function="generate",
            )
        )
        payload = parse_generated_json(response)
        rows = payload.get("behaviors")
        if not isinstance(rows, list):
            raise TypeError("Behavior extractor omitted behaviors")
        for row in rows:
            if not isinstance(row, dict):
                continue
            grounded = _ground_quote(chunk, str(row.get("quote", "")))
            if grounded is None:
                continue
            quote, relative_start, relative_end = grounded
            start = chunk_start + relative_start
            end = chunk_start + relative_end
            normalized = " ".join(quote.casefold().split())
            if normalized in seen_quotes:
                continue
            seen_quotes.add(normalized)
            behaviors.append(
                BehaviorEvidence(
                    quote=quote,
                    summary=str(row.get("summary", "")).strip(),
                    source_start=start,
                    source_end=end,
                )
            )
            if len(behaviors) >= max_behaviors:
                return behaviors, calls
    return behaviors, calls


def enrich_entities(
    client: SIEClientProtocol,
    model: str,
    behaviors: list[BehaviorEvidence],
    *,
    provision_timeout_s: float,
) -> tuple[list[BehaviorEvidence], list[dict[str, Any]]]:
    enriched: list[BehaviorEvidence] = []
    calls: list[dict[str, Any]] = []
    for index, behavior in enumerate(behaviors):
        started = time.perf_counter()
        response = client.extract(
            model,
            Item(id=f"behavior-{index}", text=behavior.quote),
            labels=ENTITY_LABELS,
            wait_for_capacity=True,
            provision_timeout_s=provision_timeout_s,
        )
        calls.append(
            request_record(
                f"entities_{index}",
                model,
                response,
                (time.perf_counter() - started) * 1000,
                function="extract",
            )
        )
        payload = jsonable(response)
        entities = payload.get("entities", []) if isinstance(payload, dict) else []
        if not isinstance(entities, list):
            entities = []
        enriched.append(
            BehaviorEvidence(
                quote=behavior.quote,
                summary=behavior.summary,
                source_start=behavior.source_start,
                source_end=behavior.source_end,
                entities=tuple(entity for entity in entities if isinstance(entity, dict)),
            )
        )
    return enriched, calls


def retrieve(
    query_vector: np.ndarray,
    catalog_vectors: np.ndarray,
    techniques: list[Technique],
    candidate_count: int,
) -> list[CandidateScore]:
    if catalog_vectors.shape[0] != len(techniques):
        raise ValueError("Catalog vectors and techniques have different row counts")
    scores = catalog_vectors @ query_vector
    count = min(candidate_count, len(techniques))
    indexes = np.argsort(-scores, kind="stable")[:count]
    return [
        CandidateScore(
            technique_id=techniques[int(index)].technique_id,
            name=techniques[int(index)].name,
            dense_score=float(scores[int(index)]),
        )
        for index in indexes
    ]


def rerank(
    client: SIEClientProtocol,
    model: str,
    behavior: BehaviorEvidence,
    candidates: list[CandidateScore],
    technique_lookup: dict[str, Technique],
    *,
    rerank_count: int,
    provision_timeout_s: float,
) -> tuple[list[CandidateScore], dict[str, Any]]:
    selected = candidates[:rerank_count]
    query = Item(id="behavior", text=f"Observed adversary behavior: {behavior.quote}")
    items = [Item(id=row.technique_id, text=technique_lookup[row.technique_id].candidate_text) for row in selected]
    started = time.perf_counter()
    response = client.score(
        model,
        query,
        items,
        instruction="Determine whether the ATT&CK technique directly describes the observed adversary behavior.",
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )
    call = request_record("rerank", model, response, (time.perf_counter() - started) * 1000, function="score")
    payload = jsonable(response)
    rows = payload.get("scores", []) if isinstance(payload, dict) else []
    dense_by_id = {row.technique_id: row.dense_score for row in selected}
    ranked: list[CandidateScore] = []
    for rank, row in enumerate(sorted(rows, key=lambda item: int(item.get("rank", 0)))):
        technique_id = str(row.get("item_id", ""))
        if technique_id not in dense_by_id:
            raise RuntimeError(f"Reranker returned an unknown technique ID: {technique_id}")
        ranked.append(
            CandidateScore(
                technique_id=technique_id,
                name=technique_lookup[technique_id].name,
                dense_score=dense_by_id[technique_id],
                rerank_score=float(row.get("score", 0)),
                rerank_rank=rank,
            )
        )
    if len(ranked) != len(selected):
        raise RuntimeError(f"Reranker returned {len(ranked)} rows for {len(selected)} candidates")
    return ranked, call


def _verify_once(
    client: SIEClientProtocol,
    model: str,
    behavior: BehaviorEvidence,
    candidates: list[CandidateScore],
    technique_lookup: dict[str, Technique],
    *,
    provision_timeout_s: float,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_text = "\n\n".join(
        f"CANDIDATE {index}\n{technique_lookup[row.technique_id].candidate_text}"
        for index, row in enumerate(candidates)
    )
    prompt = (
        "You verify one proposed ATT&CK mapping. Select a candidate only when its definition directly matches the "
        "observed behavior. Copy the supporting words exactly from OBSERVED BEHAVIOR. Mark ambiguous when two "
        "candidates remain plausible. Mark unsupported and select -1 when none is supported. The analyst, not this "
        "model, accepts or rejects the final mapping.\n\n"
        f"OBSERVED BEHAVIOR\n{behavior.quote}\n\n{candidate_text}"
    )
    started = time.perf_counter()
    response = client.generate(
        model,
        prompt,
        max_new_tokens=700,
        temperature=0,
        grammar={"json_schema": verification_schema(len(candidates)), "label": "attck_verification", "strict": True},
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )
    call = request_record(stage, model, response, (time.perf_counter() - started) * 1000, function="generate")
    return parse_generated_json(response), call


def verify_mapping(
    client: SIEClientProtocol,
    verify_model: str,
    escalation_model: str,
    behavior: BehaviorEvidence,
    candidates: list[CandidateScore],
    technique_lookup: dict[str, Technique],
    *,
    verifier_count: int,
    use_escalation: bool,
    provision_timeout_s: float,
) -> tuple[MappingDecision, list[dict[str, Any]]]:
    finalists = candidates[:verifier_count]
    if not finalists:
        raise ValueError("Verifier requires at least one candidate")
    result, call = _verify_once(
        client,
        verify_model,
        behavior,
        finalists,
        technique_lookup,
        provision_timeout_s=provision_timeout_s,
        stage="verify",
    )
    calls = [call]
    escalated = False
    verifier_model = verify_model
    if result.get("support") == "ambiguous" and use_escalation:
        result, escalation_call = _verify_once(
            client,
            escalation_model,
            behavior,
            finalists,
            technique_lookup,
            provision_timeout_s=provision_timeout_s,
            stage="escalate",
        )
        calls.append(escalation_call)
        escalated = True
        verifier_model = escalation_model
    support = str(result.get("support", "unsupported"))
    selected_index = result.get("selected_index", -1)
    evidence_quote = str(result.get("evidence_quote", "")).strip()
    grounded = _ground_quote(behavior.quote, evidence_quote)
    if type(selected_index) is not int or not (-1 <= selected_index < len(finalists)):
        raise ValueError(f"Verifier returned invalid selected_index: {selected_index!r}")
    if support == "unsupported":
        selected_index = -1
    elif selected_index < 0 or grounded is None:
        support = "unsupported"
        selected_index = -1
        evidence_quote = ""
    selected_id = finalists[selected_index].technique_id if selected_index >= 0 else None
    route = {"supported": "suggested_mapping", "ambiguous": "analyst_review", "unsupported": "abstain"}[support]
    return (
        MappingDecision(
            behavior=behavior,
            route=route,
            status="needs_analyst_review" if selected_id is not None else "no_mapping_suggested",
            selected_technique_id=selected_id,
            support=support,
            evidence_quote=grounded[0] if grounded is not None and selected_id is not None else "",
            rationale=str(result.get("rationale", "")).strip(),
            candidates=tuple(finalists),
            verifier_model=verifier_model,
            escalated=escalated,
        ),
        calls,
    )


def evidence_sha256(behavior: BehaviorEvidence) -> str:
    return hashlib.sha256(behavior.quote.encode("utf-8")).hexdigest()
