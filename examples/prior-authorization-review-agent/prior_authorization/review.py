from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.console import Console
from sie_sdk import SIEClient
from sie_sdk.types import Item

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
DOCUMENT_PATH = ROOT / "fixtures" / "cms-l1851-insufficient-documentation.html"
CONFIG_PATH = ROOT / "config.yaml"
SOURCE_URL = (
    "https://www.cms.gov/training-education/medicare-learning-networkr-mln/"
    "compliance/medicare-provider-compliance-tips/lower-limb-orthoses"
)
console = Console()

ENTITY_LABELS = [
    "procedure code",
    "administrative requirement",
    "time requirement",
    "submitted document",
    "documentation gap",
    "review conclusion",
    "payment action",
]
REQUIREMENT_FIELDS = (
    "hcpcs_code",
    "authorization_requirement_text",
    "face_to_face_requirement_text",
    "written_order_requirement_text",
    "face_to_face_window_text",
)
SUBMISSION_FIELDS = (
    "submitted_order",
    "submitted_medical_record",
    "submitted_proof_of_delivery",
    "documented_face_to_face_age",
)
OUTCOME_FIELDS = ("missing_documentation", "review_conclusion", "payment_action")
FIELD_SPECS: dict[str, dict[str, Any]] = {
    "hcpcs_code": {
        "description": "Return only the exact HCPCS code billed by the supplier in the example.",
        "scope_tokens": ("supplier bills the claim", "l1851"),
    },
    "authorization_requirement_text": {
        "description": "Return the shortest exact source text naming prior authorization as required for L1851.",
        "scope_tokens": ("prior authorization", "l1851"),
    },
    "face_to_face_requirement_text": {
        "description": "Return the shortest exact source text naming a face-to-face encounter as required for L1851.",
        "scope_tokens": ("face-to-face encounter", "l1851"),
    },
    "written_order_requirement_text": {
        "description": "Return the shortest exact source text naming the written-order-prior-to-delivery requirement.",
        "scope_tokens": ("written order prior to delivery", "l1851"),
    },
    "face_to_face_window_text": {
        "description": "Return only the exact source text stating the face-to-face encounter timing window.",
        "scope_tokens": ("within the 6 months", "prescribing the item"),
    },
    "submitted_order": {
        "description": "Return the exact source text describing the submitted standard written order.",
        "scope_tokens": ("standard written order", "correct hcpcs"),
    },
    "submitted_medical_record": {
        "description": "Return the exact source text describing the submitted treating practitioner's medical record.",
        "scope_tokens": ("medical record", "adequate medical necessity"),
    },
    "submitted_proof_of_delivery": {
        "description": "Return the exact source text describing the submitted proof of delivery and encounter timing.",
        "scope_tokens": ("proof of delivery", "face-to-face encounter 7 months ago"),
    },
    "documented_face_to_face_age": {
        "description": "Return only the exact source text stating the age of the face-to-face encounter.",
        "scope_tokens": ("face-to-face encounter 7 months ago",),
    },
    "missing_documentation": {
        "description": "Return the exact source text stating what documentation was missing.",
        "scope_tokens": ("document the face-to-face encounter within 6 months", "proof of delivery"),
    },
    "review_conclusion": {
        "description": "Return only the exact source text naming the review contractor's conclusion.",
        "scope_tokens": ("insufficient documentation error",),
    },
    "payment_action": {
        "description": "Return only the exact source text naming the MAC's payment action.",
        "scope_tokens": ("mac", "recoups", "payment"),
    },
}
GROUP_FIELDS: dict[str, tuple[str, ...]] = {
    "requirements": (
        "authorization_requirement_text",
        "face_to_face_requirement_text",
        "written_order_requirement_text",
        "face_to_face_window_text",
    ),
    "submission": (
        "hcpcs_code",
        "submitted_order",
        "submitted_medical_record",
        "submitted_proof_of_delivery",
        "documented_face_to_face_age",
    ),
    "outcome": ("missing_documentation", "review_conclusion", "payment_action"),
}
GLINER2_GROUP_LABELS = {
    "requirements": ["administrative requirement", "time requirement", "procedure code"],
    "submission": ["procedure code", "standard written order", "medical record", "proof of delivery", "time period"],
    "outcome": ["missing documentation", "claim result", "payment action", "organization", "time period"],
}
GLINER2_REQUIRED_SPANS = {
    "requirements": ("prior authorization", "face-to-face encounter", "written order", "6 months"),
    "submission": ("l1851", "medical record", "proof of delivery", "7 months"),
    "outcome": ("face-to-face encounter", "6 months", "insufficient documentation", "recoups payment"),
}


def load_config() -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["cluster"]["url"] = os.getenv("SIE_CLUSTER_URL", config["cluster"]["url"])
    config["cluster"]["api_key"] = os.getenv("SIE_API_KEY", config["cluster"]["api_key"])
    return config


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _charged_request_rows(value: Any) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if isinstance(value, dict):
        request = value.get("request")
        if isinstance(request, dict) and request.get("credits_debited"):
            usage = value.get("usage")
            rows.append((request, usage if isinstance(usage, dict) else {}))
        for child in value.values():
            rows.extend(_charged_request_rows(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_charged_request_rows(child))
    return rows


def _rate_book_provenance(raw_dir: Path) -> dict[str, Any]:
    versions: set[str] = set()
    source_artifacts: list[str] = []
    request_ids: list[str] = []
    request_versions: dict[str, str] = {}
    for path in sorted(raw_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        charged_rows = _charged_request_rows(result)
        if charged_rows:
            source_artifacts.append(f"raw/{path.name}")
        for request, usage in charged_rows:
            request_id = request.get("id")
            if not isinstance(request_id, str) or not request_id:
                raise RuntimeError(f"{path.name} has a charged request without an ID")
            request_ids.append(request_id)
            version = request.get("rate_book_version")
            if not isinstance(version, str) or not version:
                version = usage.get("rate_book_version")
            if not isinstance(version, str) or not version:
                raise RuntimeError(f"{path.name} has a charged request without a rate-book version")
            versions.add(version)
            request_versions[request_id] = version
    if len(request_ids) != len(set(request_ids)):
        raise RuntimeError("Run contains duplicate charged request IDs")
    if len(versions) != 1 or not request_ids:
        raise RuntimeError("Run does not establish one settled rate book for charged requests")
    version = versions.pop()
    return {
        "version": version,
        "source_artifacts": source_artifacts,
        "request_ids": request_ids,
        "request_versions": request_versions,
    }


def _dense(result: dict[str, Any]) -> list[float]:
    values: Any = result.get("dense")
    if isinstance(values, dict):
        values = values.get("values") or values.get("vector")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list):
        raise TypeError("Embedding response has no dense vector")
    return [float(value) for value in values]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise RuntimeError("Embedding response has a zero-length vector")
    return dot / (left_norm * right_norm)


def _chunks(markdown: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append(" ".join(current))
            current.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            flush()
            chunks.append(line)
            continue
        if line.startswith(("-", "*", "+")):
            flush()
            chunks.append(line)
            continue
        current.append(line)
        if line.endswith((".", ":", "?", "!")):
            flush()
    flush()
    chunks = [chunk for chunk in chunks if len(chunk) > 24]
    if not chunks:
        raise RuntimeError("Docling returned no usable CMS source chunks")
    return chunks


def _require_fields(data: dict[str, Any], required: tuple[str, ...], stage: str) -> None:
    missing = sorted(set(required) - set(data))
    if missing:
        raise RuntimeError(f"Mapped {stage} evidence omitted required fields: {missing}")


def _source_fragments(text: str) -> list[str]:
    fragments = [text.strip()]
    for raw_line in text.splitlines():
        inline_bullets = re.split(r"\s+(?=[-*+]\s+[A-Z])", raw_line.strip())
        for raw_fragment in inline_bullets:
            line = re.sub(r"^(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)", "", raw_fragment).strip()
            if not line:
                continue
            fragments.append(line)
            fragments.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if part.strip())
    return list(dict.fromkeys(fragments))


def _source_scope(ranked: list[dict[str, Any]], field: str) -> dict[str, Any]:
    tokens = tuple(FIELD_SPECS[field]["scope_tokens"])
    matches: list[dict[str, Any]] = []
    for row in ranked:
        if not row.get("chunk_id"):
            continue
        for fragment in _source_fragments(str(row.get("text", ""))):
            if all(token in fragment.casefold() for token in tokens):
                matches.append({**row, "text": fragment})
    if not matches:
        raise RuntimeError(f"Reranked CMS evidence omitted the exact source scope for {field}: {tokens}")
    return min(matches, key=lambda row: (len(str(row["text"])), int(row.get("rank", 0))))


def _group_source_scope(ranked: list[dict[str, Any]], group: str) -> dict[str, Any]:
    field_scopes = {canonical_field: _source_scope(ranked, canonical_field) for canonical_field in GROUP_FIELDS[group]}
    fragments: list[str] = []
    seen: set[str] = set()
    for scope in field_scopes.values():
        fragment = str(scope["text"]).strip()
        normalized = _normalized_source_text(fragment)
        if normalized not in seen:
            fragments.append(fragment)
            seen.add(normalized)
    return {
        "text": "\n\n".join(fragments),
        "field_scopes": field_scopes,
        "chunk_ids": list(dict.fromkeys(str(scope["chunk_id"]) for scope in field_scopes.values())),
    }


def _normalized_source_text(value: str) -> str:
    return " ".join(value.replace("’", "'").split()).casefold()


def _require_gliner2_group_evidence(response: dict[str, Any], group: str) -> None:
    observed = _normalized_source_text(" ".join(str(entity.get("text", "")) for entity in response.get("entities", [])))
    missing = [span for span in GLINER2_REQUIRED_SPANS[group] if span not in observed]
    if missing:
        raise RuntimeError(f"GLiNER2 {group} extraction omitted exact CMS source spans: {missing}")


def _require_tokens(value: str, tokens: tuple[str, ...], field: str) -> None:
    normalized = value.casefold()
    missing = [token for token in tokens if token not in normalized]
    if missing:
        raise RuntimeError(f"GLiNER2 {field} output omitted source terms {missing}: {value!r}")


def _month_count(value: str, field: str) -> int:
    match = re.search(r"\b(\d+)\s+months?\b", value.casefold())
    if match is None:
        raise RuntimeError(f"GLiNER2 {field} output has no month count: {value!r}")
    return int(match.group(1))


def _require_entity_evidence(result: dict[str, Any]) -> None:
    observed = " ".join(str(entity.get("text", "")) for entity in result.get("entities", [])).casefold()
    required = ("l1851", "6 months", "7 months")
    missing = [token for token in required if token not in observed]
    if missing:
        raise RuntimeError(f"GLiNER did not recover required CMS source spans: {missing}")


def _require_ranked_evidence(ranked: list[dict[str, Any]]) -> None:
    if not ranked:
        raise RuntimeError("Reranker returned no evidence")
    texts: list[str] = []
    for row in ranked:
        text = str(row.get("text", "")).strip()
        if not row.get("chunk_id") or not text:
            raise RuntimeError("Reranker evidence must retain its chunk identity and source text")
        texts.append(text)
    joined = "\n".join(texts).casefold()
    required = (
        "l1851",
        "prior authorization",
        "written order prior to delivery",
        "within the 6 months",
        "face-to-face encounter 7 months ago",
        "insufficient documentation error",
        "mac recoups payment",
    )
    missing = [token for token in required if token not in joined]
    if missing:
        raise RuntimeError(f"Reranker evidence omitted required CMS text: {missing}")


def _select_evidence_rows(
    ranked: list[dict[str, Any]],
    terms: tuple[str, ...],
    stage: str,
) -> list[dict[str, Any]]:
    selected = [row for row in ranked if any(term in str(row["text"]).casefold() for term in terms)]
    if not selected:
        raise RuntimeError(f"No reranked CMS evidence remained for {stage}")
    return sorted(selected, key=lambda row: int(str(row["chunk_id"]).split("-")[-1]))


def build_review(
    requirements: dict[str, Any],
    submission: dict[str, Any],
    outcome: dict[str, Any],
    ranked: list[dict[str, Any]],
) -> dict[str, Any]:
    _require_ranked_evidence(ranked)
    _require_fields(requirements, REQUIREMENT_FIELDS, "requirements")
    _require_fields(submission, SUBMISSION_FIELDS, "submission")
    _require_fields(outcome, OUTCOME_FIELDS, "outcome")

    code = str(requirements["hcpcs_code"]).strip().upper()
    if code != "L1851":
        raise RuntimeError(f"Mapped evidence returned the wrong HCPCS code: {code!r}")
    _require_tokens(
        str(requirements["authorization_requirement_text"]),
        ("prior authorization",),
        "authorization_requirement_text",
    )
    _require_tokens(
        str(requirements["face_to_face_requirement_text"]),
        ("face-to-face",),
        "face_to_face_requirement_text",
    )
    _require_tokens(
        str(requirements["written_order_requirement_text"]),
        ("written order", "prior to delivery"),
        "written_order_requirement_text",
    )

    required_months = _month_count(
        str(requirements["face_to_face_window_text"]),
        "face_to_face_window_text",
    )
    observed_months = _month_count(
        str(submission["documented_face_to_face_age"]),
        "documented_face_to_face_age",
    )
    if required_months != 6:
        raise RuntimeError(f"Mapped evidence returned the wrong face-to-face window: {required_months}")
    overdue_by_months = observed_months - required_months
    if overdue_by_months <= 0:
        raise RuntimeError("Published example does not establish an overdue face-to-face encounter")
    if observed_months != 7:
        raise RuntimeError(f"Mapped evidence returned the wrong encounter age: {observed_months}")

    _require_tokens(str(submission["submitted_order"]), ("standard written order", "hcpcs"), "submitted_order")
    _require_tokens(
        str(submission["submitted_medical_record"]),
        ("medical", "adequate", "necessity"),
        "submitted_medical_record",
    )
    _require_tokens(
        str(submission["submitted_proof_of_delivery"]),
        ("proof of delivery", "face-to-face"),
        "submitted_proof_of_delivery",
    )
    _require_tokens(
        str(outcome["missing_documentation"]),
        ("face-to-face", "within 6 months", "proof of delivery"),
        "missing_documentation",
    )
    _require_tokens(str(outcome["review_conclusion"]), ("insufficient documentation",), "review_conclusion")
    _require_tokens(str(outcome["payment_action"]), ("mac", "recoup", "payment"), "payment_action")

    return {
        "scope": "Reproduction of CMS's published L1851 insufficient-documentation example",
        "route": "insufficient_documentation",
        "headline": "The face-to-face encounter fell one month outside CMS's six-month window",
        "hcpcs_code": code,
        "required_face_to_face_within_months": required_months,
        "documented_face_to_face_age_months": observed_months,
        "overdue_by_months": overdue_by_months,
        "submitted_documentation": {
            "order": str(submission["submitted_order"]),
            "medical_record": str(submission["submitted_medical_record"]),
            "proof_of_delivery": str(submission["submitted_proof_of_delivery"]),
        },
        "missing_documentation": [str(outcome["missing_documentation"]).strip()],
        "review_conclusion": str(outcome["review_conclusion"]).strip(),
        "payment_action": str(outcome["payment_action"]).strip(),
        "source": {
            "publisher": "Centers for Medicare & Medicaid Services",
            "title": "Lower Limb Orthoses",
            "url": SOURCE_URL,
            "page_last_modified": "2026-02-11",
        },
        "ranked_source_evidence": ranked,
        "published_example": True,
        "coverage_decision": None,
        "medical_decision": None,
    }


def _extract_gliner2_group(
    client: SIEClient,
    config: dict[str, Any],
    group: str,
    scope: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_text = str(scope["text"])
    stage = f"gliner2_{group}"
    started = time.perf_counter()
    response = client.extract(
        config["models"]["extract"],
        Item(id=f"cms-l1851-{group}", text=source_text),
        labels=GLINER2_GROUP_LABELS[group],
        wait_for_capacity=True,
        provision_timeout_s=config["cluster"]["provision_timeout_s"],
    )
    call = {
        "stage": stage,
        "model": config["models"]["extract"],
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "source_chunk_ids": scope["chunk_ids"],
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "source_fragment_sha256": {
            canonical_field: hashlib.sha256(str(field_scope["text"]).encode("utf-8")).hexdigest()
            for canonical_field, field_scope in scope["field_scopes"].items()
        },
    }
    return response, call


def run(run_id: str) -> Path:
    config = load_config()
    parse_model = str(config["models"]["parse"])
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        raise SystemExit(
            f"Run directory already exists: {run_dir}. Choose a new --run-id or remove the existing run directory."
        )
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)
    calls: list[dict[str, Any]] = []
    timeout = config["cluster"]["provision_timeout_s"]

    with SIEClient(config["cluster"]["url"], api_key=config["cluster"]["api_key"] or None, timeout_s=timeout) as client:
        started = time.perf_counter()
        parsed = client.extract(
            parse_model,
            Item(id="cms-l1851-published-example", document=DOCUMENT_PATH),
            options={"profile": "default"},
            wait_for_capacity=True,
            provision_timeout_s=timeout,
        )
        calls.append(
            {
                "stage": "parse",
                "model": parse_model,
                "configured_model": config["models"]["parse"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        _write_json(raw_dir / "parse.json", parsed)
        markdown = str(parsed.get("data", {}).get("markdown", ""))
        (run_dir / "parsed.md").write_text(markdown, encoding="utf-8")
        chunks = _chunks(markdown)

        started = time.perf_counter()
        query_embedding = client.encode(
            config["models"]["retrieve"],
            Item(id="cms-l1851-query", text=config["review"]["query"]),
            wait_for_capacity=True,
            provision_timeout_s=timeout,
        )
        chunk_embeddings = [
            client.encode(
                config["models"]["retrieve"],
                Item(id=f"chunk-{index}", text=text),
                wait_for_capacity=True,
                provision_timeout_s=timeout,
            )
            for index, text in enumerate(chunks)
        ]
        retrieval = sorted(
            [
                {"chunk_id": f"chunk-{index}", "text": text, "score": _cosine(_dense(query_embedding), _dense(result))}
                for index, (text, result) in enumerate(zip(chunks, chunk_embeddings, strict=True))
            ],
            key=lambda row: row["score"],
            reverse=True,
        )[: config["review"]["candidate_chunks"]]
        calls.append(
            {
                "stage": "retrieve",
                "model": config["models"]["retrieve"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        _write_json(
            raw_dir / "retrieve.json",
            {"query": query_embedding, "chunks": chunk_embeddings, "ranking": retrieval},
        )

        started = time.perf_counter()
        rerank_query = Item(id="cms-l1851-query", text=config["review"]["query"])
        rerank_items = [Item(id=row["chunk_id"], text=row["text"]) for row in retrieval]
        _write_json(
            raw_dir / "rerank-request.json",
            {
                "model": config["models"]["rerank"],
                "query": rerank_query,
                "items": rerank_items,
            },
        )
        rerank_raw = client.score(
            config["models"]["rerank"],
            rerank_query,
            rerank_items,
            wait_for_capacity=True,
            provision_timeout_s=timeout,
        )
        by_id = {row["chunk_id"]: row["text"] for row in retrieval}
        ranked = [
            {
                "chunk_id": row["item_id"],
                "rank": row["rank"],
                "score": row["score"],
                "text": by_id[row["item_id"]],
            }
            for row in sorted(rerank_raw["scores"], key=lambda item: item["rank"])
        ][: config["review"]["top_k"]]
        calls.append(
            {
                "stage": "rerank",
                "model": config["models"]["rerank"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        _write_json(raw_dir / "rerank.json", rerank_raw)
        requirements_rows = _select_evidence_rows(
            ranked,
            ("we require prior authorization", "within the 6 months"),
            "requirements",
        )
        submission_rows = _select_evidence_rows(
            ranked,
            (
                "standard written order",
                "medical necessity information",
                "proof of delivery with face-to-face",
            ),
            "submission",
        )
        outcome_rows = _select_evidence_rows(
            ranked,
            (
                "document the face-to-face encounter within 6 months",
                "insufficient documentation error",
                "mac recoups",
            ),
            "outcome",
        )
        entity_outputs: list[dict[str, Any]] = []
        requirement_labels = ["procedure code", "administrative requirement", "time requirement"]
        case_labels = [
            "procedure code",
            "submitted document",
            "time period",
            "documentation gap",
            "review conclusion",
            "payment action",
        ]
        entity_inputs = [
            (f"entities_requirement_{index}", str(row["text"]), requirement_labels)
            for index, row in enumerate(requirements_rows)
        ]
        entity_inputs.extend(
            (f"entities_case_{index}", str(row["text"]), case_labels)
            for index, row in enumerate([*submission_rows, *outcome_rows])
        )
        for stage, text, labels in entity_inputs:
            started = time.perf_counter()
            response = client.extract(
                config["models"]["entities"],
                Item(id=f"cms-l1851-{stage}", text=text),
                labels=labels,
                wait_for_capacity=True,
                provision_timeout_s=timeout,
            )
            calls.append(
                {
                    "stage": stage,
                    "model": config["models"]["entities"],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            )
            _write_json(raw_dir / f"{stage.replace('_', '-')}.json", response)
            entity_outputs.append(response)
        entities = {
            "model": config["models"]["entities"],
            "entities": [entity for response in entity_outputs for entity in response.get("entities", [])],
        }
        _write_json(raw_dir / "entities.json", entities)
        _require_entity_evidence(entities)

        structured: dict[str, str] = {}
        source_scopes: dict[str, dict[str, Any]] = {}
        for group in GROUP_FIELDS:
            scope = _group_source_scope(ranked, group)
            response, call = _extract_gliner2_group(client, config, group, scope)
            calls.append(call)
            _write_json(raw_dir / f"gliner2-{group}.json", response)
            _require_gliner2_group_evidence(response, group)
            for canonical_field, field_scope in scope["field_scopes"].items():
                source_text = str(field_scope["text"]).strip()
                if canonical_field == "hcpcs_code":
                    code_match = re.search(r"\bL1851\b", source_text, flags=re.IGNORECASE)
                    if code_match is None:
                        raise RuntimeError("The ranked CMS claim scope omitted L1851")
                    source_text = code_match.group()
                structured[canonical_field] = source_text
                source_scopes[canonical_field] = {
                    "group": group,
                    "chunk_id": field_scope["chunk_id"],
                    "source_sha256": call["source_fragment_sha256"][canonical_field],
                }
        _write_json(
            raw_dir / "mapped.json",
            {
                "method": "Exact ranked CMS fragments validated against GLiNER2 source spans",
                "data": structured,
                "source_scopes": source_scopes,
            },
        )

        review = build_review(
            {field: structured[field] for field in REQUIREMENT_FIELDS},
            {field: structured[field] for field in SUBMISSION_FIELDS},
            {field: structured[field] for field in OUTCOME_FIELDS},
            ranked,
        )

    _write_json(run_dir / "review.json", review)
    artifact_paths = [run_dir / "parsed.md", run_dir / "review.json", *sorted(raw_dir.glob("*.json"))]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "endpoint": config["cluster"]["url"],
        "models": config["models"],
        "fixture": {"path": str(DOCUMENT_PATH.relative_to(ROOT)), "sha256": sha256(DOCUMENT_PATH)},
        "rate_book_provenance": _rate_book_provenance(raw_dir),
        "artifacts": [{"path": str(path.relative_to(run_dir)), "sha256": sha256(path)} for path in artifact_paths],
        "source": review["source"],
        "calls": calls,
        "pipeline": [
            "parse",
            "retrieve",
            "rerank",
            "entities_requirement",
            "entities_case",
            *[f"gliner2_{group}" for group in GROUP_FIELDS],
            "deterministic_source_mapping",
            "deterministic_validation",
        ],
        "decision_boundary": (
            "This run reproduces one example published by CMS. It does not make a coverage, medical-necessity, "
            "diagnosis, treatment, or prospective payment decision."
        ),
    }
    _write_json(run_dir / "manifest.json", manifest)
    console.print(f"[green]Wrote[/] {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce CMS's published L1851 insufficient-documentation example")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    run(args.run_id)


if __name__ == "__main__":
    main()
