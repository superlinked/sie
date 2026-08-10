from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
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
DOCUMENT_PATH = ROOT / "fixtures" / "east-palestine-bearing-spread.pdf"
CONFIG_PATH = ROOT / "config.yaml"
SCHEMA_PROBE_PATHS = (
    ROOT / "fixtures" / "output-schema-probe-request.json",
    ROOT / "fixtures" / "output-schema-probe-response.json",
)
console = Console()

MAPPED_FIELDS = (
    "bearing",
    "sebring_time",
    "sebring_temperature",
    "salem_time",
    "salem_temperature",
    "east_palestine_time",
    "east_palestine_temperature",
    "sebring_alert_status_text",
    "salem_alert_level_text",
    "salem_alert_recipient_text",
    "salem_crew_notification_text",
    "salem_camera_observation_text",
    "east_palestine_alert_text",
    "engineer_action",
    "derailment_statement",
    "cause_statement",
)
GLINER2_LABELS = {
    "detector": [
        "location",
        "event time",
        "bearing",
        "degrees above ambient",
        "alert status",
        "alert recipient",
        "camera observation",
    ],
    "cause": ["bearing failure"],
    "engineer": ["operator action", "event time"],
    "derailment": ["derailed railcars"],
}
GLINER2_REQUIRED_SPANS = {
    "sebring": ("7:37 p.m.", "sebring hbd", "38°f", "l1 bearing"),
    "salem": (
        "8:13 p.m.",
        "salem hbd",
        "l1 bearing",
        "103°f",
        "noncritical",
        "wayside help desk",
        "fire near the bearing",
    ),
    "east_palestine": ("8:52 p.m", "east palestine hbd", "253°f", "l1", "critical alarm", "locomotive cab"),
    "cause": ("overheated bearing",),
    "engineer": ("slow the train", "8:54 p.m."),
    "derailment": ("hopper car", "37 others"),
}
SOURCE_CAUSE_STATEMENT = (
    "The East Palestine derailment began when an overheated bearing burned off the accident hopper car."
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


def _charged_request_rows(value: Any) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if isinstance(value, dict):
        request = value.get("request")
        if isinstance(request, dict) and request.get("credits_debited"):
            usage = request.get("usage")
            if not isinstance(usage, dict):
                usage = value.get("usage")
            rows.append((request, usage if isinstance(usage, dict) else {}))
        for child in value.values():
            rows.extend(_charged_request_rows(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_charged_request_rows(child))
    return rows


def _request_rate_book_version(request: dict[str, Any], usage: dict[str, Any], source: str) -> str:
    direct_present = "rate_book_version" in request
    usage_present = "rate_book_version" in usage
    direct = request.get("rate_book_version")
    nested = usage.get("rate_book_version")
    if direct_present and (not isinstance(direct, str) or not direct):
        raise RuntimeError(f"{source} has a charged request without a rate-book version")
    if usage_present and (not isinstance(nested, str) or not nested):
        raise RuntimeError(f"{source} has a charged request without a rate-book version")
    if direct_present and usage_present and direct != nested:
        raise RuntimeError(f"{source} has conflicting rate-book versions for one request")
    version = direct if direct_present else nested
    if not isinstance(version, str) or not version:
        raise RuntimeError(f"{source} has a charged request without a rate-book version")
    return version


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
            version = _request_rate_book_version(request, usage, path.name)
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
    chunks = [part.strip() for part in re.split(r"\n\s*\n", markdown) if len(part.strip()) > 24]
    if not chunks:
        raise RuntimeError("Docling returned no usable NTSB chunks")
    return chunks


def _number(value: str) -> int:
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    if not match:
        raise RuntimeError(f"Structured field contains no number: {value!r}")
    return round(float(match.group()))


def _minute_of_day(value: str) -> int:
    normalized = value.casefold().replace(".", "")
    match = re.search(r"(\d{1,2}):(\d{2})\s*([ap]m)", normalized)
    if not match:
        raise RuntimeError(f"Structured field contains no clock time: {value!r}")
    hour = int(match.group(1)) % 12
    if match.group(3) == "pm":
        hour += 12
    return hour * 60 + int(match.group(2))


def _normalize_source_text(value: str) -> str:
    return " ".join(value.replace(" °", "°").split()).casefold()


def _require_entity_evidence(result: dict[str, Any]) -> None:
    observed = " ".join(str(entity.get("text", "")) for entity in result.get("entities", []))
    normalized = _normalize_source_text(observed)
    required = (
        "sebring",
        "salem",
        "east palestine",
        "7:37",
        "8:13",
        "8:52",
        "l1 bearing",
        "noncritical alert",
        "critical alarm",
        "37 others",
    )
    missing = [token for token in required if token not in normalized]
    if missing:
        raise RuntimeError(f"GLiNER omitted required NTSB source spans: {missing}")


def _require_ranked_evidence(ranked: list[dict[str, Any]]) -> str:
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
        "sebring",
        "salem",
        "east palestine",
        "38°f",
        "103°f",
        "253°f",
        "wayside help desk",
        "not to the crew",
        "fire near the bearing",
        "37 others derailed",
        "engineer began to slow",
        "8:54 p.m.",
    )
    missing = [token for token in required if token not in joined]
    if missing:
        raise RuntimeError(f"Reranker evidence omitted required NTSB source text: {missing}")
    if _normalize_source_text(SOURCE_CAUSE_STATEMENT) not in joined:
        raise RuntimeError("Reranked evidence omitted the NTSB's exact cause statement")
    return joined


def _select_evidence_rows(
    ranked: list[dict[str, Any]],
    terms: tuple[str, ...],
    stage: str,
) -> list[dict[str, Any]]:
    selected = [row for row in ranked if any(term in str(row["text"]).casefold() for term in terms)]
    if not selected:
        raise RuntimeError(f"No reranked NTSB evidence remained for {stage}")
    return sorted(selected, key=lambda row: int(str(row["chunk_id"]).split("-")[-1]))


def _require_gliner2_evidence(response: dict[str, Any], stage: str) -> None:
    observed = _normalize_source_text(" ".join(str(entity.get("text", "")) for entity in response.get("entities", [])))
    missing = [span for span in GLINER2_REQUIRED_SPANS[stage] if span not in observed]
    if missing:
        raise RuntimeError(f"GLiNER2 {stage} extraction omitted exact NTSB source spans: {missing}")


def _row_for_location(rows: list[dict[str, Any]], location: str) -> dict[str, Any]:
    matches = [row for row in rows if location in str(row["text"]).casefold()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one ranked NTSB detector row for {location}, found {len(matches)}")
    return matches[0]


def _shortest_row_with(rows: list[dict[str, Any]], term: str) -> dict[str, Any]:
    matches = [row for row in rows if term in str(row["text"]).casefold()]
    if not matches:
        raise RuntimeError(f"Ranked NTSB evidence omitted the source row containing {term!r}")
    return min(matches, key=lambda row: (len(str(row["text"])), int(row["rank"])))


def _source_match(text: str, pattern: str, field: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"Ranked NTSB source omitted {field}")
    return match.group(0)


def _exact_source_phrase(text: str, phrase: str, field: str) -> str:
    start = text.casefold().find(phrase.casefold())
    if start < 0:
        raise RuntimeError(f"Ranked NTSB source omitted the exact phrase for {field}: {phrase!r}")
    return text[start : start + len(phrase)]


def _map_exact_source_fields(
    detector_rows: list[dict[str, Any]],
    outcome_rows: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    detector_sources = {
        "sebring": _row_for_location(detector_rows, "sebring hbd"),
        "salem": _row_for_location(detector_rows, "salem hbd"),
        "east_palestine": _row_for_location(detector_rows, "east palestine hbd"),
    }
    outcome_text = "\n\n".join(str(row["text"]) for row in outcome_rows)
    sebring_text = str(detector_sources["sebring"]["text"])
    salem_text = str(detector_sources["salem"]["text"])
    east_palestine_text = str(detector_sources["east_palestine"]["text"])
    data = {
        "bearing": _source_match(sebring_text, r"\bL1\b", "bearing"),
        "sebring_time": _source_match(sebring_text, r"\b7:37\s+p\.m\.", "Sebring time"),
        "sebring_temperature": _source_match(
            sebring_text,
            r"\b38°F\s+above\s+ambient\b",
            "Sebring temperature",
        ),
        "salem_time": _source_match(salem_text, r"\b8:13\s+p\.m\.", "Salem time"),
        "salem_temperature": _source_match(
            salem_text,
            r"\b103°F\s+above\s+ambient\b",
            "Salem temperature",
        ),
        "east_palestine_time": _source_match(
            east_palestine_text,
            r"\b8:52\s+p\.m\s*\.",
            "East Palestine time",
        ),
        "east_palestine_temperature": _source_match(
            east_palestine_text,
            r"\b253°F\s+above\s+ambient\b",
            "East Palestine temperature",
        ),
        "sebring_alert_status_text": _exact_source_phrase(
            sebring_text,
            "not high enough to trigger an alert",
            "Sebring alert status",
        ),
        "salem_alert_level_text": _exact_source_phrase(
            salem_text,
            "noncritical alert",
            "Salem alert level",
        ),
        "salem_alert_recipient_text": _exact_source_phrase(
            salem_text,
            "Wayside Help Desk",
            "Salem alert recipient",
        ),
        "salem_crew_notification_text": _exact_source_phrase(
            salem_text,
            "not to the crew",
            "Salem crew notification",
        ),
        "salem_camera_observation_text": _exact_source_phrase(
            salem_text,
            "fire near the bearing",
            "Salem camera observation",
        ),
        "east_palestine_alert_text": _exact_source_phrase(
            east_palestine_text,
            "critical alarm, which was broadcast in the locomotive cab",
            "East Palestine alert",
        ),
        "engineer_action": _exact_source_phrase(
            outcome_text,
            "The engineer began to slow the train before 8:54 p.m.",
            "engineer action",
        ),
        "derailment_statement": _exact_source_phrase(
            outcome_text,
            "the hopper car and 37 others derailed as the train's emergency braking system activated",
            "derailment statement",
        ),
        "cause_statement": _exact_source_phrase(
            outcome_text,
            SOURCE_CAUSE_STATEMENT,
            "NTSB cause statement",
        ),
    }
    group_for_field = {
        **{
            field: "sebring"
            for field in ("bearing", "sebring_time", "sebring_temperature", "sebring_alert_status_text")
        },
        **{
            field: "salem"
            for field in (
                "salem_time",
                "salem_temperature",
                "salem_alert_level_text",
                "salem_alert_recipient_text",
                "salem_crew_notification_text",
                "salem_camera_observation_text",
            )
        },
        **{
            field: "east_palestine"
            for field in ("east_palestine_time", "east_palestine_temperature", "east_palestine_alert_text")
        },
    }
    source_scopes = {
        field: {
            "stage": group,
            "chunk_id": str(detector_sources[group]["chunk_id"]),
            "source_sha256": hashlib.sha256(str(detector_sources[group]["text"]).encode("utf-8")).hexdigest(),
        }
        for field, group in group_for_field.items()
    }
    outcome_hash = hashlib.sha256(outcome_text.encode("utf-8")).hexdigest()
    outcome_chunks = ",".join(str(row["chunk_id"]) for row in outcome_rows)
    for field in ("engineer_action", "derailment_statement", "cause_statement"):
        source_scopes[field] = {
            "stage": "outcome",
            "chunk_id": outcome_chunks,
            "source_sha256": outcome_hash,
        }
    return data, source_scopes


def _require_fields(data: dict[str, Any]) -> None:
    required = set(MAPPED_FIELDS)
    missing = sorted(required - set(data))
    if missing:
        raise RuntimeError(f"Mapped source evidence omitted required fields: {missing}")


def _derailed_car_count(value: str) -> int:
    match = re.search(r"\b(?:the\s+)?hopper car and (\d+) others derailed\b", value, flags=re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"Mapped derailment statement has no derivable railcar count: {value!r}")
    return 1 + int(match.group(1))


def build_review(data: dict[str, Any], ranked: list[dict[str, Any]]) -> dict[str, Any]:
    _require_ranked_evidence(ranked)
    _require_fields(data)

    if str(data["bearing"]).strip().casefold() != "l1":
        raise RuntimeError("Mapped source evidence did not identify the L1 bearing")
    temperatures = [
        _number(str(data["sebring_temperature"])),
        _number(str(data["salem_temperature"])),
        _number(str(data["east_palestine_temperature"])),
    ]
    if temperatures != [38, 103, 253]:
        raise RuntimeError(f"Mapped detector temperatures do not match the NTSB source: {temperatures}")
    sebring_minute = _minute_of_day(str(data["sebring_time"]))
    salem_minute = _minute_of_day(str(data["salem_time"]))
    east_palestine_minute = _minute_of_day(str(data["east_palestine_time"]))
    if [sebring_minute, salem_minute, east_palestine_minute] != [
        19 * 60 + 37,
        20 * 60 + 13,
        20 * 60 + 52,
    ]:
        raise RuntimeError("Mapped detector time does not match the NTSB source")

    alert_checks = {
        "sebring": ("not", "alert"),
        "salem": ("noncritical", "alert"),
        "recipient": ("wayside help desk",),
        "crew": ("not", "crew"),
        "camera": ("fire", "bearing"),
        "east_palestine": ("critical", "cab"),
    }
    alert_values = {
        "sebring": str(data["sebring_alert_status_text"]).casefold(),
        "salem": str(data["salem_alert_level_text"]).casefold(),
        "recipient": str(data["salem_alert_recipient_text"]).casefold(),
        "crew": str(data["salem_crew_notification_text"]).casefold(),
        "camera": str(data["salem_camera_observation_text"]).casefold(),
        "east_palestine": str(data["east_palestine_alert_text"]).casefold(),
    }
    for name, tokens in alert_checks.items():
        if not all(token in alert_values[name] for token in tokens):
            raise RuntimeError(f"Mapped {name} alert evidence does not match the NTSB source")

    engineer_action = str(data["engineer_action"])
    derailment = str(data["derailment_statement"])
    cause = str(data["cause_statement"])
    if not all(token in engineer_action.casefold() for token in ("slow", "8:54")):
        raise RuntimeError("Mapped source evidence omitted the engineer's action before 8:54 p.m.")
    if not all(token in derailment.casefold() for token in ("hopper", "37", "derail")):
        raise RuntimeError("Mapped source evidence omitted the hopper car and 37 other derailed cars")
    total_derailed_cars = _derailed_car_count(derailment)
    if total_derailed_cars != 38:
        raise RuntimeError(f"Mapped source evidence returned the wrong derailed railcar count: {total_derailed_cars}")
    if not all(token in cause.casefold() for token in ("overheated bearing", "burned off", "hopper car")):
        raise RuntimeError("Mapped source evidence changed the NTSB's stated derailment cause")

    detector_deltas = [
        temperatures[1] - temperatures[0],
        temperatures[2] - temperatures[1],
    ]
    return {
        "route": "read_only_detector_trend_review",
        "source": "NTSB Illustrated Digest SPC-24-06, printed pages 4–5",
        "bearing": str(data["bearing"]),
        "detector_readings": [
            {
                "location": "Sebring",
                "time": str(data["sebring_time"]),
                "degrees_f_above_ambient": temperatures[0],
                "alert": str(data["sebring_alert_status_text"]),
            },
            {
                "location": "Salem",
                "time": str(data["salem_time"]),
                "degrees_f_above_ambient": temperatures[1],
                "alert": str(data["salem_alert_level_text"]),
                "alert_recipient": str(data["salem_alert_recipient_text"]),
                "crew_notification": str(data["salem_crew_notification_text"]),
                "camera_observation": str(data["salem_camera_observation_text"]),
            },
            {
                "location": "East Palestine",
                "time": str(data["east_palestine_time"]),
                "degrees_f_above_ambient": temperatures[2],
                "alert": str(data["east_palestine_alert_text"]),
            },
        ],
        "trend": {
            "successive_increases_degrees_f": detector_deltas,
            "total_increase_degrees_f": temperatures[-1] - temperatures[0],
            "sebring_to_salem_minutes": salem_minute - sebring_minute,
            "salem_to_east_palestine_minutes": east_palestine_minute - salem_minute,
        },
        "engineer_action": engineer_action,
        "derailment": {
            "total_cars": total_derailed_cars,
            "statement": derailment,
        },
        "ntsb_cause_statement": cause,
        "ranked_source_evidence": ranked,
        "new_causal_inferences": [],
        "control_writes": [],
        "safety_boundary": "Read-only reconstruction of the NTSB's published detector sequence.",
    }


def run(run_id: str) -> Path:
    config = load_config()
    final_run_dir = RUNS_DIR / run_id
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    reservation_dir = RUNS_DIR / f".{run_id}.lock"
    try:
        reservation_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"Run ID is already reserved: {run_id}") from exc
    try:
        if final_run_dir.exists():
            raise FileExistsError(f"Run evidence already exists at {final_run_dir}")
        staging_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=RUNS_DIR))
        try:
            _write_run(staging_dir, config)
            staging_dir.rename(final_run_dir)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
    finally:
        reservation_dir.rmdir()
    console.print(f"[green]Wrote[/] {final_run_dir}")
    return final_run_dir


def _write_run(run_dir: Path, config: dict[str, Any]) -> None:
    parse_model = str(config["models"]["parse"])
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)
    calls: list[dict[str, Any]] = []
    timeout = config["cluster"]["provision_timeout_s"]

    with SIEClient(config["cluster"]["url"], api_key=config["cluster"]["api_key"] or None, timeout_s=timeout) as client:
        started = time.perf_counter()
        parsed = client.extract(
            parse_model,
            Item(id="ntsb-east-palestine-bearing-spread", document=DOCUMENT_PATH),
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
            Item(id="bearing-trend-query", text=config["review"]["query"]),
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
        rerank_query = Item(id="bearing-trend-query", text=config["review"]["query"])
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
        detector_rows = _select_evidence_rows(
            ranked,
            ("38°f above ambient", "103°f above ambient", "253°f above ambient"),
            "detector readings",
        )
        outcome_rows = _select_evidence_rows(
            ranked,
            (
                "overheated bearing burned off",
                "engineer began to slow",
                "hopper car and 37 others derailed",
            ),
            "published outcome",
        )
        outcome_text = "\n\n".join(row["text"] for row in outcome_rows)

        entity_outputs: list[dict[str, Any]] = []
        for index, row in enumerate(detector_rows):
            started = time.perf_counter()
            response = client.extract(
                config["models"]["entities"],
                Item(id=f"ntsb-detector-{index}", text=str(row["text"])),
                labels=["location", "event time", "bearing", "temperature reading", "alert"],
                wait_for_capacity=True,
                provision_timeout_s=timeout,
            )
            calls.append(
                {
                    "stage": f"entities_detector_{index}",
                    "model": config["models"]["entities"],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            )
            _write_json(raw_dir / f"entities-detector-{index}.json", response)
            entity_outputs.append(response)

        started = time.perf_counter()
        outcome_entities = client.extract(
            config["models"]["entities"],
            Item(id="ntsb-outcome", text=outcome_text),
            labels=["location", "bearing", "event time", "railcar count", "engineer action"],
            wait_for_capacity=True,
            provision_timeout_s=timeout,
        )
        calls.append(
            {
                "stage": "entities_outcome",
                "model": config["models"]["entities"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        _write_json(raw_dir / "entities-outcome.json", outcome_entities)
        entity_outputs.append(outcome_entities)
        entities = {
            "model": config["models"]["entities"],
            "entities": [entity for response in entity_outputs for entity in response.get("entities", [])],
        }
        _write_json(raw_dir / "entities.json", entities)
        _require_entity_evidence(entities)

        detector_sources = {
            "sebring": _row_for_location(detector_rows, "sebring hbd"),
            "salem": _row_for_location(detector_rows, "salem hbd"),
            "east_palestine": _row_for_location(detector_rows, "east palestine hbd"),
        }
        for stage, row in detector_sources.items():
            started = time.perf_counter()
            response = client.extract(
                config["models"]["extract"],
                Item(id=f"ranked-ntsb-{stage}-evidence", text=str(row["text"])),
                labels=GLINER2_LABELS["detector"],
                wait_for_capacity=True,
                provision_timeout_s=timeout,
            )
            calls.append(
                {
                    "stage": f"gliner2_{stage}",
                    "model": config["models"]["extract"],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "source_chunk_ids": [str(row["chunk_id"])],
                    "source_sha256": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
                }
            )
            _write_json(raw_dir / f"gliner2-{stage.replace('_', '-')}.json", response)
            _require_gliner2_evidence(response, stage)

        outcome_sources = {
            "cause": _shortest_row_with(outcome_rows, "overheated bearing burned off"),
            "engineer": _shortest_row_with(outcome_rows, "engineer began to slow"),
            "derailment": _shortest_row_with(outcome_rows, "hopper car and 37 others derailed"),
        }
        for stage, row in outcome_sources.items():
            source_text = str(row["text"])
            started = time.perf_counter()
            response = client.extract(
                config["models"]["extract"],
                Item(id=f"ranked-ntsb-{stage}-evidence", text=source_text),
                labels=GLINER2_LABELS[stage],
                wait_for_capacity=True,
                provision_timeout_s=timeout,
            )
            calls.append(
                {
                    "stage": f"gliner2_{stage}",
                    "model": config["models"]["extract"],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "source_chunk_ids": [str(row["chunk_id"])],
                    "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                }
            )
            _write_json(raw_dir / f"gliner2-{stage}.json", response)
            _require_gliner2_evidence(response, stage)

        mapped_data, source_scopes = _map_exact_source_fields(detector_rows, outcome_rows)
        _write_json(
            raw_dir / "mapped.json",
            {
                "method": "Exact ranked NTSB fragments validated against GLiNER2 source spans",
                "data": mapped_data,
                "source_scopes": source_scopes,
            },
        )
        review = build_review(mapped_data, ranked)

    _write_json(run_dir / "review.json", review)
    artifact_paths = [run_dir / "parsed.md", run_dir / "review.json", *sorted(raw_dir.glob("*.json"))]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "endpoint": config["cluster"]["url"],
        "models": config["models"],
        "rate_book_provenance": _rate_book_provenance(raw_dir),
        "fixture": {"path": str(DOCUMENT_PATH.relative_to(ROOT)), "sha256": sha256(DOCUMENT_PATH)},
        "diagnostic_fixtures": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in SCHEMA_PROBE_PATHS
        ],
        "artifacts": [{"path": str(path.relative_to(run_dir)), "sha256": sha256(path)} for path in artifact_paths],
        "source_document": {
            "url": "https://www.ntsb.gov/investigations/AccidentReports/Reports/SPC2406.pdf",
            "source_page_index": 2,
            "printed_pages": "4–5",
        },
        "calls": calls,
        "pipeline": [
            "parse",
            "retrieve",
            "rerank",
            "entities_detector_sections",
            "entities_outcome",
            "gliner2_detector_sections",
            "gliner2_outcome_sections",
            "deterministic_source_mapping",
            "deterministic_validation",
        ],
        "decision_boundary": (
            "The review reconstructs the NTSB's published detector sequence. "
            "It makes no new causal determination and performs no control write."
        ),
    }
    _write_json(run_dir / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct the NTSB East Palestine bearing-detector sequence")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    run(args.run_id)


if __name__ == "__main__":
    main()
