from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.console import Console
from sie_sdk import SIEClient
from sie_sdk.types import Item

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
DOCUMENT_PATH = ROOT / "fixtures" / "pathward-filing-packet.html"
CONFIG_PATH = ROOT / "config.yaml"
console = Console()

FIGURE_ENTITY_LABELS = ["company", "reporting period", "filing", "money amount"]
STATUS_ENTITY_LABELS = ["company", "filing", "reliance status"]
REQUIRED_FIELDS = (
    "company",
    "period",
    "original_net_income",
    "restated_net_income",
    "original_diluted_eps",
    "restated_diluted_eps",
    "reliance_status",
)
GLINER2_FIGURE_LABELS = [
    "reporting period",
    "previously reported net income",
    "as restated net income",
    "previously reported diluted EPS",
    "as restated diluted EPS",
]
GLINER2_ORIGINAL_FIGURE_LABELS = ["reporting period", "net income", "diluted EPS"]
GLINER2_STATUS_LABELS = ["company", "prior filing reliance status"]
CAVEAT_SOURCE_TEXT = (
    "The change from net to gross basis presentation does not impact net income "
    "over the life of the portfolio, but changes the timing of when elements of "
    "the programs are recognized for accounting purposes."
)


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
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## ") and current:
            chunks.append(" ".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append(" ".join(current))
    chunks = [chunk for chunk in chunks if len(chunk) > 24]
    if not chunks:
        raise RuntimeError("Docling returned no usable filing chunks")
    return chunks


def _number(value: str) -> Decimal:
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    if not match:
        raise RuntimeError(f"Structured field contains no number: {value!r}")
    return Decimal(match.group())


def _millions_from_thousands(value: str) -> Decimal:
    amount = _number(value)
    if abs(amount) < 1000:
        raise RuntimeError(f"Expected a source-table value in thousands, received: {value!r}")
    return amount / Decimal(1000)


def _normalize_source_text(value: str) -> str:
    return " ".join(value.replace("$ ", "$").split()).casefold()


def _require_entity_evidence(result: dict[str, Any]) -> None:
    observed = _normalize_source_text(" ".join(str(entity.get("text", "")) for entity in result.get("entities", [])))
    required_spans = ("45,096", "36,080", "$1.68")
    missing = sorted(span for span in required_spans if span.casefold() not in observed)
    if missing:
        raise RuntimeError(f"GLiNER omitted required source spans: {missing}")
    if "pathward financial" not in observed or "june 30, 2023" not in observed:
        raise RuntimeError("GLiNER omitted the company or reporting-period source span")


def _require_ranked_evidence(ranked: list[dict[str, Any]]) -> None:
    if not ranked:
        raise RuntimeError("Reranker returned no evidence")
    texts: list[str] = []
    for row in ranked:
        text = str(row.get("text", "")).strip()
        if not row.get("chunk_id") or not text:
            raise RuntimeError("Reranker evidence must retain its chunk identity and source text")
        texts.append(text)
    joined = _normalize_source_text("\n".join(texts))
    required = (
        "pathward financial",
        "$45,096",
        "$36,080",
        "$1.68",
        "$1.34",
        "no longer be relied upon",
        "over the life of the portfolio",
    )
    missing = [token for token in required if token not in joined]
    if missing:
        raise RuntimeError(f"Reranker evidence omitted required filing source text: {missing}")


def _select_evidence_rows(
    ranked: list[dict[str, Any]],
    terms: tuple[str, ...],
    stage: str,
) -> list[dict[str, Any]]:
    selected = [row for row in ranked if any(term in str(row["text"]).casefold() for term in terms)]
    if not selected:
        raise RuntimeError(f"No reranked evidence remained for {stage}")
    return sorted(selected, key=lambda row: int(str(row["chunk_id"]).split("-")[-1]))


def _select_source_sentences(text: str, terms: tuple[str, ...], stage: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    selected = [sentence for sentence in sentences if any(term in sentence.casefold() for term in terms)]
    if not selected:
        raise RuntimeError(f"No exact source sentence remained for {stage}")
    return " ".join(selected)


def _single_source_row(ranked: list[dict[str, Any]], term: str, stage: str) -> str:
    rows = _select_evidence_rows(ranked, (term,), stage)
    exact_rows = [str(row["text"]) for row in rows if term in str(row["text"]).casefold()]
    if not exact_rows:
        raise RuntimeError(f"No exact reranked source row remained for {stage}")
    return min(exact_rows, key=len)


def _table_source_values(text: str, row_label: str) -> tuple[str, str]:
    marker = f"| {row_label}"
    start = text.casefold().find(marker.casefold())
    if start < 0:
        raise RuntimeError(f"The reranked restated table omitted the {row_label} row")
    cells = [cell.strip() for cell in text[start:].split("|")]
    if len(cells) < 6 or cells[1].casefold() != row_label.casefold():
        raise RuntimeError(f"Docling returned an invalid {row_label} table row")
    original, restated = cells[2], cells[5]
    if not original or not restated:
        raise RuntimeError(f"Docling omitted the original or restated {row_label} value")
    return original, restated


def _original_table_source_value(text: str, row_label: str) -> str:
    marker = f"| {row_label}"
    start = text.casefold().find(marker.casefold())
    if start < 0:
        raise RuntimeError(f"The reranked original Form 10-Q omitted the {row_label} row")
    cells = [cell.strip() for cell in text[start:].split("|")]
    if len(cells) < 3 or cells[1].casefold() != row_label.casefold():
        raise RuntimeError(f"Docling returned an invalid original Form 10-Q {row_label} row")
    value = cells[2]
    if not value:
        raise RuntimeError(f"Docling omitted the original Form 10-Q {row_label} value")
    return value


def _require_matching_source_values(original: str, previously_reported: str, stage: str) -> None:
    if _normalize_source_text(original) != _normalize_source_text(previously_reported):
        raise RuntimeError(
            f"The original Form 10-Q and the Form 10-K/A 'As Previously Reported' {stage} values disagree"
        )


def _entity_span(result: dict[str, Any], label: str, tokens: tuple[str, ...], stage: str) -> str:
    matches = [
        str(entity.get("text", "")).strip()
        for entity in result.get("entities", [])
        if str(entity.get("label", "")).casefold() == label.casefold()
        and all(token in str(entity.get("text", "")).casefold() for token in tokens)
    ]
    if not matches:
        raise RuntimeError(f"GLiNER2 omitted the exact {stage} source span")
    return max(matches, key=len)


def _require_gliner2_figure_spans(result: dict[str, Any]) -> None:
    observed = _normalize_source_text(" ".join(str(entity.get("text", "")) for entity in result.get("entities", [])))
    required = ("three months ended june 30, 2023", "$45,096", "$36,080", "$1.68", "$1.34")
    missing = [span for span in required if span not in observed]
    if missing:
        raise RuntimeError(f"GLiNER2 omitted required filing table spans: {missing}")


def _require_gliner2_original_figure_spans(result: dict[str, Any]) -> None:
    observed = _normalize_source_text(" ".join(str(entity.get("text", "")) for entity in result.get("entities", [])))
    required = ("$45,096", "$1.68")
    missing = [span for span in required if span not in observed]
    if missing:
        raise RuntimeError(f"GLiNER2 omitted required original Form 10-Q spans: {missing}")


def build_review(data: dict[str, Any], ranked: list[dict[str, Any]]) -> dict[str, Any]:
    _require_ranked_evidence(ranked)
    missing = sorted(set(REQUIRED_FIELDS) - set(data))
    if missing:
        raise RuntimeError(f"Mapped source evidence omitted required fields: {missing}")
    original_income = _millions_from_thousands(str(data["original_net_income"]))
    restated_income = _millions_from_thousands(str(data["restated_net_income"]))
    original_eps = _number(str(data["original_diluted_eps"]))
    restated_eps = _number(str(data["restated_diluted_eps"]))
    expected = {
        "original_income": Decimal("45.096"),
        "restated_income": Decimal("36.080"),
        "original_eps": Decimal("1.68"),
        "restated_eps": Decimal("1.34"),
    }
    observed = {
        "original_income": original_income,
        "restated_income": restated_income,
        "original_eps": original_eps,
        "restated_eps": restated_eps,
    }
    if observed != expected:
        raise RuntimeError(f"Mapped values do not match the cited source packet: {observed}")
    period = str(data["period"])
    if "june 30, 2023" not in period.casefold():
        raise RuntimeError("Mapped period does not match the source table")
    reliance = str(data["reliance_status"])
    if "no longer" not in reliance.casefold() or "relied" not in reliance.casefold():
        raise RuntimeError("Mapped reliance status does not identify the superseded source")
    ranked_text = " ".join(str(row["text"]).strip() for row in ranked)
    if CAVEAT_SOURCE_TEXT.casefold() not in ranked_text.casefold():
        raise RuntimeError("Reranked source evidence omitted Pathward's life-of-portfolio caveat")

    delta = restated_income - original_income
    percent = (delta / original_income * Decimal(100)).quantize(Decimal("0.1"))
    return {
        "route": "superseded_figure",
        "company": str(data["company"]),
        "period": period,
        "metric": "net income attributable to parent",
        "original": {
            "value_millions": float(original_income),
            "diluted_eps": float(original_eps),
            "source_id": "2023-10q",
        },
        "restated": {
            "value_millions": float(restated_income),
            "diluted_eps": float(restated_eps),
            "source_id": "2025-10ka",
        },
        "change": {
            "value_millions": float(delta),
            "diluted_eps": float(restated_eps - original_eps),
            "percent": float(percent),
        },
        "source_status": reliance,
        "company_caveat": CAVEAT_SOURCE_TEXT,
        "controlling_sources": ["2023-10q", "2025-8k-item-4-02", "2025-10ka"],
        "ranked_evidence": ranked,
        "claims_excluded": ["fraud", "misconduct", "investment recommendation"],
    }


def run(run_id: str) -> Path:
    config = load_config()
    parse_model = str(config["models"]["parse"])
    run_dir = RUNS_DIR / run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)
    calls: list[dict[str, Any]] = []
    timeout = config["cluster"]["provision_timeout_s"]

    with SIEClient(config["cluster"]["url"], api_key=config["cluster"]["api_key"] or None, timeout_s=900) as client:
        started = time.perf_counter()
        parsed = client.extract(
            parse_model,
            Item(id="pathward-source-packet", document=DOCUMENT_PATH),
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
            Item(id="filing-query", text=config["review"]["query"]),
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
        rerank_raw = client.score(
            config["models"]["rerank"],
            Item(id="filing-query", text=config["review"]["query"]),
            [Item(id=row["chunk_id"], text=row["text"]) for row in retrieval],
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
        original_table_text = _single_source_row(ranked, "original form 10-q", "original Form 10-Q extraction")
        restated_table_text = _single_source_row(ranked, "restated form 10-k/a", "restated Form 10-K/A extraction")
        status_rows = _select_evidence_rows(ranked, ("item 4.02 form 8-k",), "filing status")
        status_text = "\n\n".join(row["text"] for row in status_rows)
        status_model_text = _select_source_sentences(
            status_text,
            ('affected periods") should no longer be relied upon',),
            "filing status extraction",
        )

        entity_outputs: list[dict[str, Any]] = []
        entity_inputs = [
            ("entities_original_10q", original_table_text, FIGURE_ENTITY_LABELS),
            ("entities_restated_10ka", restated_table_text, FIGURE_ENTITY_LABELS),
        ]
        entity_inputs.append(("entities_status", status_model_text, STATUS_ENTITY_LABELS))
        for stage, text, labels in entity_inputs:
            started = time.perf_counter()
            response = client.extract(
                config["models"]["entities"],
                Item(id=stage, text=text),
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

        started = time.perf_counter()
        gliner2_original = client.extract(
            config["models"]["extract"],
            Item(id="ranked-original-10q-table-evidence", text=original_table_text),
            labels=GLINER2_ORIGINAL_FIGURE_LABELS,
            wait_for_capacity=True,
            provision_timeout_s=timeout,
        )
        calls.append(
            {
                "stage": "gliner2_original_10q",
                "model": config["models"]["extract"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        _write_json(raw_dir / "gliner2-original-10q.json", gliner2_original)
        _require_gliner2_original_figure_spans(gliner2_original)

        started = time.perf_counter()
        gliner2_restated = client.extract(
            config["models"]["extract"],
            Item(id="ranked-restated-table-evidence", text=restated_table_text),
            labels=GLINER2_FIGURE_LABELS,
            wait_for_capacity=True,
            provision_timeout_s=timeout,
        )
        calls.append(
            {
                "stage": "gliner2_restated_10ka",
                "model": config["models"]["extract"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        _write_json(raw_dir / "gliner2-restated-10ka.json", gliner2_restated)
        _require_gliner2_figure_spans(gliner2_restated)

        started = time.perf_counter()
        gliner2_status = client.extract(
            config["models"]["extract"],
            Item(id="ranked-filing-status-evidence", text=status_text),
            labels=GLINER2_STATUS_LABELS,
            wait_for_capacity=True,
            provision_timeout_s=timeout,
        )
        calls.append(
            {
                "stage": "gliner2_status",
                "model": config["models"]["extract"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        _write_json(raw_dir / "gliner2-status.json", gliner2_status)

        original_income = _original_table_source_value(original_table_text, "Net income attributable to parent")
        original_eps = _original_table_source_value(original_table_text, "Diluted")
        previously_reported_income, restated_income = _table_source_values(
            restated_table_text, "Net income attributable to parent"
        )
        previously_reported_eps, restated_eps = _table_source_values(restated_table_text, "Diluted")
        _require_matching_source_values(original_income, previously_reported_income, "net income")
        _require_matching_source_values(original_eps, previously_reported_eps, "diluted EPS")
        structured_data = {
            "company": " ".join(_entity_span(gliner2_status, "company", ("pathward financial",), "company").split()),
            "period": _entity_span(
                gliner2_restated,
                "reporting period",
                ("june 30, 2023",),
                "reporting period",
            ),
            "original_net_income": original_income,
            "restated_net_income": restated_income,
            "original_diluted_eps": original_eps,
            "restated_diluted_eps": restated_eps,
            "reliance_status": _entity_span(
                gliner2_status,
                "prior filing reliance status",
                ("no longer", "relied"),
                "reliance status",
            ),
        }
        structured = {
            "model": config["models"]["extract"],
            "method": "Source-specific Docling table coordinates validated against source-specific GLiNER2 spans",
            "data": structured_data,
            "source_fields": {
                "original_net_income": {"source_id": "2023-10q", "value": original_income},
                "original_diluted_eps": {"source_id": "2023-10q", "value": original_eps},
                "restated_net_income": {"source_id": "2025-10ka", "value": restated_income},
                "restated_diluted_eps": {"source_id": "2025-10ka", "value": restated_eps},
            },
        }
        _write_json(raw_dir / "mapped.json", structured)
        review = build_review(dict(structured.get("data", {})), ranked)

    _write_json(run_dir / "review.json", review)
    artifact_paths = [run_dir / "parsed.md", run_dir / "review.json", *sorted(raw_dir.glob("*.json"))]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "endpoint": config["cluster"]["url"],
        "sie_server_commit": os.getenv("SIE_SERVER_COMMIT"),
        "run_command": os.getenv("SIE_RUN_COMMAND"),
        "models": config["models"],
        "fixture": {"path": str(DOCUMENT_PATH.relative_to(ROOT)), "sha256": sha256(DOCUMENT_PATH)},
        "artifacts": [{"path": str(path.relative_to(run_dir)), "sha256": sha256(path)} for path in artifact_paths],
        "calls": calls,
        "pipeline": [
            "parse",
            "retrieve",
            "rerank",
            "entities_original_10q",
            "entities_restated_10ka",
            "entities_status",
            "gliner2_original_10q",
            "gliner2_restated_10ka",
            "gliner2_status",
            "source_specific_table_mapping",
            "cross_filing_value_check",
            "deterministic_validation",
        ],
        "decision_boundary": (
            "The review fails closed unless GLiNER and GLiNER2 recover source spans from both filing tables, "
            "the original Form 10-Q values match the Form 10-K/A's previously reported column, and reranked "
            "evidence retains the filing-status span and Pathward's exact life-of-portfolio caveat."
        ),
    }
    _write_json(run_dir / "manifest.json", manifest)
    console.print(f"[green]Wrote[/] {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace a restated Pathward financial fact through SEC filings")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    run(args.run_id)


if __name__ == "__main__":
    main()
