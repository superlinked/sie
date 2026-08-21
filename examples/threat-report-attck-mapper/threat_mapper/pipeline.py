from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from typing import Any

import numpy as np
from sie_sdk.scoring import maxsim
from sie_sdk.types import Item

from .models import (
    BehaviorEvidence,
    CandidateScore,
    ExemplarScore,
    LabeledTechniqueExample,
    MappingDecision,
    Technique,
)
from .sie import (
    SIEClientProtocol,
    jsonable,
    parse_generated_json,
    request_record,
    traced_request_record,
)

ENTITY_LABELS = [
    "threat actor",
    "malware",
    "software tool",
    "vulnerability",
    "credential",
    "authentication token",
    "network protocol",
    "command or script",
    "cloud service",
    "organization",
    "infrastructure",
    "target system",
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
                    "actor": {"type": "string"},
                    "action": {"type": "string"},
                    "object": {"type": "string"},
                    "tool": {"type": "string"},
                    "target": {"type": "string"},
                    "assertion": {
                        "type": "string",
                        "enum": ["observed", "capability", "background", "defensive"],
                    },
                },
                "required": ["quote", "summary", "actor", "action", "object", "tool", "target", "assertion"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["behaviors"],
    "additionalProperties": False,
}


def behavior_schema(max_items: int) -> dict[str, Any]:
    schema = copy.deepcopy(BEHAVIOR_SCHEMA)
    schema["properties"]["behaviors"]["maxItems"] = max_items
    return schema


def verification_schema(candidate_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selected_index": {"type": "integer", "minimum": -1, "maximum": candidate_count - 1},
            "support": {"type": "string", "enum": ["supported", "ambiguous", "unsupported"]},
            "evidence_quote": {"type": "string", "maxLength": 300},
            "rationale": {"type": "string", "maxLength": 240},
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


def split_report(text: str, max_characters: int, overlap_characters: int = 0) -> list[tuple[int, int]]:
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    if overlap_characters < 0 or overlap_characters >= max_characters:
        raise ValueError("overlap_characters must be between zero and max_characters")

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
    if overlap_characters:
        chunks = [
            (max(0, start - overlap_characters) if index else start, end) for index, (start, end) in enumerate(chunks)
        ]
    return chunks


def extract_document_entities(
    client: SIEClientProtocol,
    model: str,
    report_text: str,
    *,
    chunk_characters: int,
    overlap_characters: int,
    provision_timeout_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entities: dict[tuple[int, int, str], dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []
    spans = split_report(report_text, chunk_characters, overlap_characters)
    for chunk_index, (chunk_start, chunk_end) in enumerate(spans):
        chunk = report_text[chunk_start:chunk_end]
        started = time.perf_counter()
        response = client.extract(
            model,
            Item(id=f"report-{chunk_index}", text=chunk),
            labels=ENTITY_LABELS,
            wait_for_capacity=True,
            provision_timeout_s=provision_timeout_s,
        )
        calls.append(
            traced_request_record(
                f"document_entities_{chunk_index}",
                model,
                response,
                (time.perf_counter() - started) * 1000,
                function="extract",
                request_payload={
                    "item": {"id": f"report-{chunk_index}", "text": chunk},
                    "labels": ENTITY_LABELS,
                },
            )
        )
        payload = jsonable(response)
        rows = payload.get("entities", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            relative_start = row.get("start")
            relative_end = row.get("end")
            if type(relative_start) is not int or type(relative_end) is not int:
                continue
            start = chunk_start + relative_start
            end = chunk_start + relative_end
            if not (0 <= start < end <= len(report_text)):
                continue
            text = report_text[start:end]
            if _normalized_text(text) != _normalized_text(str(row.get("text", ""))):
                continue
            label = str(row.get("label", ""))
            entity = {
                "text": text,
                "label": label,
                "score": float(row.get("score", 0.0)),
                "start": start,
                "end": end,
            }
            key = (start, end, label)
            if key not in entities or entity["score"] > entities[key]["score"]:
                entities[key] = entity
    return sorted(entities.values(), key=lambda row: (int(row["start"]), int(row["end"]), str(row["label"]))), calls


def _normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _entities_for_chunk(entities: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return [row for row in entities if int(row["start"]) < end and int(row["end"]) > start]


def extract_behaviors(
    client: SIEClientProtocol,
    model: str,
    report_text: str,
    *,
    max_behaviors: int,
    chunk_characters: int,
    provision_timeout_s: float,
    overlap_characters: int = 0,
    document_entities: list[dict[str, Any]] | None = None,
) -> tuple[list[BehaviorEvidence], list[dict[str, Any]]]:
    behaviors: list[BehaviorEvidence] = []
    calls: list[dict[str, Any]] = []
    seen_behaviors: set[tuple[int, int, str, str]] = set()
    entities = document_entities or []
    pending_chunks = [
        (chunk_start, chunk_end, str(chunk_index), max_behaviors)
        for chunk_index, (chunk_start, chunk_end) in enumerate(
            split_report(report_text, chunk_characters, overlap_characters)
        )
    ]
    while pending_chunks:
        chunk_start, chunk_end, chunk_label, behavior_limit = pending_chunks.pop(0)
        chunk = report_text[chunk_start:chunk_end]
        anchors = _entities_for_chunk(entities, chunk_start, chunk_end)
        anchor_text = (
            "\n".join(f"- {row['label']}: {row['text']}" for row in anchors[:40]) or "- No entity anchors found"
        )
        response_schema = behavior_schema(behavior_limit)
        prompt = (
            "You review a cyber threat report. Extract atomic technical behaviors that can map to MITRE ATT&CK "
            "Enterprise. Create a separate row for every distinct action. A compound sentence that says malware steals "
            "credentials, records the screen, and downloads another payload needs separate rows with the shortest exact "
            "source clause for each action. Include observed activity and stated malware or tool capabilities. Exclude "
            "sales, advertising, prices, possession claims, marketplace roles, and generic criminal activity unless the "
            "same clause states the technical action used against a system, account, network, or data. Use observed when "
            "the report says an attacker or malware performed the action. Use capability when a tool can perform it. Use "
            "background for historical adversary activity. Use defensive only for an action performed or recommended by "
            "a defender, such as blocking, patching, detection, or incident response. Malware evasion is adversary "
            "activity. Skip headings, image captions, predictions, and generic security prose. Copy one "
            "exact, self-contained quote for each behavior. Keep enough words to identify the actor, action, object, tool, "
            "and target when the report states them. Use an empty string for a missing event field. Return compact "
            "JSON on one line, with no indentation or whitespace outside string values. "
            f"Return at most {behavior_limit} behaviors from this chunk. Entity anchors are hints, not requirements.\n\n"
            f"ENTITY ANCHORS\n{anchor_text}\n\n"
            f"REPORT CHUNK\n{chunk}"
        )
        started = time.perf_counter()
        response = client.generate(
            model,
            prompt,
            max_new_tokens=4096,
            temperature=0,
            grammar={"json_schema": response_schema, "label": "threat_behaviors", "strict": True},
            wait_for_capacity=True,
            provision_timeout_s=provision_timeout_s,
        )
        calls.append(
            traced_request_record(
                f"behavior_extract_{chunk_label}",
                model,
                response,
                (time.perf_counter() - started) * 1000,
                function="generate",
                request_payload={
                    "prompt": prompt,
                    "max_new_tokens": 4096,
                    "temperature": 0,
                    "grammar": {
                        "json_schema": response_schema,
                        "label": "threat_behaviors",
                        "strict": True,
                    },
                },
            )
        )
        try:
            payload = parse_generated_json(response)
        except ValueError:
            retry_size = max(800, len(chunk) // 2)
            retry_spans = split_report(chunk, retry_size)
            retry_limit = max(1, behavior_limit // 2)
            if len(retry_spans) < 2 and retry_limit == behavior_limit:
                raise
            calls[-1]["outcome"] = "invalid_json_retried_with_smaller_request"
            if len(retry_spans) < 2:
                pending_chunks.insert(
                    0,
                    (chunk_start, chunk_end, f"{chunk_label}.0", retry_limit),
                )
            else:
                pending_chunks[0:0] = [
                    (
                        chunk_start + start,
                        chunk_start + end,
                        f"{chunk_label}.{index}",
                        retry_limit,
                    )
                    for index, (start, end) in enumerate(retry_spans)
                ]
            continue
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
            action = str(row.get("action", "")).strip()
            object_ = str(row.get("object", "")).strip()
            behavior_key = (start, end, _normalized_text(action), _normalized_text(object_))
            if behavior_key in seen_behaviors:
                continue
            seen_behaviors.add(behavior_key)
            overlapping_entities = tuple(
                entity for entity in entities if int(entity["start"]) < end and int(entity["end"]) > start
            )
            behaviors.append(
                BehaviorEvidence(
                    quote=quote,
                    summary=str(row.get("summary", "")).strip(),
                    source_start=start,
                    source_end=end,
                    entities=overlapping_entities,
                    actor=str(row.get("actor", "")).strip(),
                    action=action,
                    object=object_,
                    tool=str(row.get("tool", "")).strip(),
                    target=str(row.get("target", "")).strip(),
                    assertion=str(row.get("assertion", "observed")),
                )
            )
    return sorted(behaviors, key=lambda row: (row.source_start, row.source_end)), calls


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
                actor=behavior.actor,
                action=behavior.action,
                object=behavior.object,
                tool=behavior.tool,
                target=behavior.target,
                assertion=behavior.assertion,
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
            dense_rank=rank,
        )
        for rank, index in enumerate(indexes)
    ]


def retrieve_hybrid(
    query_vector: np.ndarray,
    catalog_vectors: np.ndarray,
    query_multivector: np.ndarray,
    catalog_multivectors: list[np.ndarray],
    techniques: list[Technique],
    *,
    dense_count: int,
    late_interaction_count: int,
    candidate_count: int,
    exemplar_candidates: list[ExemplarScore] | None = None,
    exemplar_count: int = 0,
    exemplar_rrf_weight: float = 1.0,
) -> list[CandidateScore]:
    if catalog_vectors.shape[0] != len(techniques) or len(catalog_multivectors) != len(techniques):
        raise ValueError("Catalog embeddings and techniques have different row counts")
    dense_scores = catalog_vectors @ query_vector
    late_scores = np.asarray(maxsim(query_multivector, catalog_multivectors), dtype=np.float32)
    dense_order = np.argsort(-dense_scores, kind="stable")
    late_order = np.argsort(-late_scores, kind="stable")
    dense_ranks = {int(index): rank for rank, index in enumerate(dense_order)}
    late_ranks = {int(index): rank for rank, index in enumerate(late_order)}
    pool = {int(index) for index in dense_order[:dense_count]}
    pool.update(int(index) for index in late_order[:late_interaction_count])
    technique_indexes = {technique.technique_id: index for index, technique in enumerate(techniques)}
    exemplar_by_id = {
        row.technique_id: row
        for row in (exemplar_candidates or [])[:exemplar_count]
        if row.technique_id in technique_indexes
    }
    pool.update(technique_indexes[technique_id] for technique_id in exemplar_by_id)

    rows: list[CandidateScore] = []
    for index in pool:
        dense_rank = dense_ranks[index]
        late_rank = late_ranks[index]
        fusion_score = 1.0 / (60 + dense_rank + 1) + 1.0 / (60 + late_rank + 1)
        exemplar = exemplar_by_id.get(techniques[index].technique_id)
        if exemplar is not None:
            fusion_score += exemplar_rrf_weight / (60 + exemplar.rank + 1)
        rows.append(
            CandidateScore(
                technique_id=techniques[index].technique_id,
                name=techniques[index].name,
                dense_score=float(dense_scores[index]),
                late_interaction_score=float(late_scores[index]),
                dense_rank=dense_rank,
                late_interaction_rank=late_rank,
                fusion_score=fusion_score,
                exemplar_score=exemplar.score if exemplar is not None else None,
                exemplar_rank=exemplar.rank if exemplar is not None else None,
                exemplar_quote=exemplar.quote if exemplar is not None else None,
                exemplar_document=exemplar.document if exemplar is not None else None,
            )
        )
    return sorted(
        rows,
        key=lambda row: (
            -(row.fusion_score or 0.0),
            row.dense_rank if row.dense_rank is not None else len(techniques),
            row.technique_id,
        ),
    )[:candidate_count]


def retrieve_exemplars(
    query_vector: np.ndarray,
    example_vectors: np.ndarray,
    examples: list[LabeledTechniqueExample],
    technique_lookup: dict[str, Technique],
    candidate_count: int,
) -> list[ExemplarScore]:
    if example_vectors.shape[0] != len(examples):
        raise ValueError("Example vectors and labeled examples have different row counts")
    scores = example_vectors @ query_vector
    order = np.argsort(-scores, kind="stable")
    result: list[ExemplarScore] = []
    seen: set[str] = set()
    for index in order:
        example = examples[int(index)]
        if example.technique_id not in technique_lookup or example.technique_id in seen:
            continue
        seen.add(example.technique_id)
        result.append(
            ExemplarScore(
                technique_id=example.technique_id,
                score=float(scores[int(index)]),
                rank=len(result),
                quote=example.context,
                document=example.document,
            )
        )
        if len(result) == candidate_count:
            break
    return result


def _candidate_text(candidate: CandidateScore, technique_lookup: dict[str, Technique]) -> str:
    text = technique_lookup[candidate.technique_id].candidate_text
    if candidate.exemplar_quote:
        text += f"\nLabeled report example: {candidate.exemplar_quote}"
    return text


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
    items = [Item(id=row.technique_id, text=_candidate_text(row, technique_lookup)) for row in selected]
    started = time.perf_counter()
    response = client.score(
        model,
        query,
        items,
        instruction="Determine whether the ATT&CK technique directly describes the observed adversary behavior.",
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )
    call = traced_request_record(
        "rerank",
        model,
        response,
        (time.perf_counter() - started) * 1000,
        function="score",
        request_payload={
            "query": {"id": query["id"], "text": query["text"]},
            "items": [{"id": item["id"], "text": item["text"]} for item in items],
            "instruction": "Determine whether the ATT&CK technique directly describes the observed adversary behavior.",
        },
    )
    payload = jsonable(response)
    rows = payload.get("scores", []) if isinstance(payload, dict) else []
    selected_by_id = {row.technique_id: row for row in selected}
    ranked: list[CandidateScore] = []
    for rank, row in enumerate(sorted(rows, key=lambda item: int(item.get("rank", 0)))):
        technique_id = str(row.get("item_id", ""))
        if technique_id not in selected_by_id:
            raise RuntimeError(f"Reranker returned an unknown technique ID: {technique_id}")
        source = selected_by_id[technique_id]
        ranked.append(
            CandidateScore(
                technique_id=technique_id,
                name=technique_lookup[technique_id].name,
                dense_score=source.dense_score,
                rerank_score=float(row.get("score", 0)),
                rerank_rank=rank,
                late_interaction_score=source.late_interaction_score,
                dense_rank=source.dense_rank,
                late_interaction_rank=source.late_interaction_rank,
                fusion_score=source.fusion_score,
                exemplar_score=source.exemplar_score,
                exemplar_rank=source.exemplar_rank,
                exemplar_quote=source.exemplar_quote,
                exemplar_document=source.exemplar_document,
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
    max_new_tokens: int = 500,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    candidate_text = "\n\n".join(
        f"CANDIDATE {index}\n{_candidate_text(row, technique_lookup)}" for index, row in enumerate(candidates)
    )
    event_fields = "\n".join(
        field
        for field in [
            f"Actor: {behavior.actor}" if behavior.actor else "",
            f"Action: {behavior.action}" if behavior.action else "",
            f"Object: {behavior.object}" if behavior.object else "",
            f"Tool: {behavior.tool}" if behavior.tool else "",
            f"Target: {behavior.target}" if behavior.target else "",
        ]
        if field
    )
    prompt = (
        "You verify one ATT&CK mapping from a threat report. Match the action and object in EXTRACTED EVENT to one "
        "candidate definition. SOURCE QUOTE is the evidence boundary. Do not infer a delivery channel, protocol, access "
        "method, or tool that the quote does not state. Select a sub-technique only when the quote states its defining "
        "detail; otherwise prefer a supported parent technique. Respect actor direction and operation phase. Downloading a "
        "payload into a compromised system is tool transfer. Acquiring malware means an adversary obtains a capability for "
        "a future operation. Advertising or selling malware proves neither action. Defensive and law-enforcement actions "
        "do not become adversary techniques. A labeled report example shows how the candidate has been used in public "
        "training data; it is guidance, not evidence for this quote. Independently decide whether the quote describes "
        "adversary activity. Copy the supporting words exactly from SOURCE QUOTE. Mark ambiguous when two candidates remain "
        "plausible. Mark unsupported and select -1 when none is supported. The analyst accepts or rejects the final mapping. "
        "Return one JSON object with exactly selected_index, support, evidence_quote, and rationale. Do not return an array. "
        "Write compact one-line JSON with no whitespace outside string values. Keep the rationale to one sentence under 30 "
        "words.\n\n"
        f"EXTRACTED EVENT\n{event_fields}\n\nSOURCE QUOTE\n{behavior.quote}\n\n{candidate_text}"
    )
    started = time.perf_counter()
    response = client.generate(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0,
        grammar={"json_schema": verification_schema(len(candidates)), "label": "attck_verification", "strict": True},
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )
    call = traced_request_record(
        stage,
        model,
        response,
        (time.perf_counter() - started) * 1000,
        function="generate",
        request_payload={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
            "grammar": {
                "json_schema": verification_schema(len(candidates)),
                "label": "attck_verification",
                "strict": True,
            },
        },
    )
    try:
        return parse_generated_json(response), call
    except ValueError:
        call["outcome"] = "invalid_json"
        return None, call


def _verify_with_retry(
    client: SIEClientProtocol,
    model: str,
    behavior: BehaviorEvidence,
    candidates: list[CandidateScore],
    technique_lookup: dict[str, Technique],
    *,
    provision_timeout_s: float,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result, first_call = _verify_once(
        client,
        model,
        behavior,
        candidates,
        technique_lookup,
        provision_timeout_s=provision_timeout_s,
        stage=stage,
    )
    calls = [first_call]
    if result is not None:
        return result, calls

    result, retry_call = _verify_once(
        client,
        model,
        behavior,
        candidates,
        technique_lookup,
        provision_timeout_s=provision_timeout_s,
        stage=f"{stage}_retry",
        max_new_tokens=1200,
    )
    calls.append(retry_call)
    if result is not None:
        return result, calls

    return (
        {
            "selected_index": -1,
            "support": "unsupported",
            "evidence_quote": "",
            "rationale": "Verifier returned invalid JSON twice; no mapping emitted.",
        },
        calls,
    )


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
    result, calls = _verify_with_retry(
        client,
        verify_model,
        behavior,
        finalists,
        technique_lookup,
        provision_timeout_s=provision_timeout_s,
        stage="verify",
    )
    escalated = False
    verifier_model = verify_model
    if result.get("support") == "ambiguous" and use_escalation:
        result, escalation_calls = _verify_with_retry(
            client,
            escalation_model,
            behavior,
            finalists,
            technique_lookup,
            provision_timeout_s=provision_timeout_s,
            stage="escalate",
        )
        calls.extend(escalation_calls)
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
    exemplar_agreement = support == "supported" and selected_index >= 0 and finalists[selected_index].exemplar_rank == 0
    if selected_id is None:
        route = "abstain"
    elif exemplar_agreement:
        route = "suggested_mapping"
    else:
        route = "analyst_review"
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
            exemplar_agreement=exemplar_agreement,
        ),
        calls,
    )


def evidence_sha256(behavior: BehaviorEvidence) -> str:
    payload = json.dumps(
        behavior.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
