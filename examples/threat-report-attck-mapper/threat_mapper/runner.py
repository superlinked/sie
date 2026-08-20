from __future__ import annotations

import contextlib
import json
import re
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sie_sdk import SIEClient
from sie_sdk.types import Item

from .catalog import catalog_by_id, load_annoctr_catalog, load_catalog
from .config import RUNS_DIR
from .data import ensure_sources, find_annoctr_catalog, find_annoctr_report, load_linking_cases, sha256
from .evaluation import evaluate_predictions
from .models import BehaviorEvidence, Technique
from .pipeline import enrich_entities, evidence_sha256, extract_behaviors, rerank, retrieve, verify_mapping
from .sie import encode_texts, jsonable, request_record


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) or run_id in {".", ".."}:
        raise ValueError("run_id must be one safe directory name")
    return run_id


def _begin_run(run_id: str) -> tuple[Path, Path, Path]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    final_dir = RUNS_DIR / run_id
    reservation = RUNS_DIR / f".{run_id}.lock"
    try:
        reservation.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"Run ID is already reserved: {run_id}") from exc
    try:
        if final_dir.exists():
            raise FileExistsError(f"Run already exists: {final_dir}")
        staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=RUNS_DIR))
    except BaseException:
        with contextlib.suppress(OSError):
            reservation.rmdir()
        raise
    return final_dir, staging, reservation


def _artifact_rows(run_dir: Path) -> list[dict[str, str]]:
    return [
        {"path": path.relative_to(run_dir).as_posix(), "sha256": sha256(path)}
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def _rate_book_provenance(calls: list[dict[str, Any]]) -> dict[str, Any]:
    charged = [call for call in calls if call.get("credits_debited")]
    if not charged:
        return {"version": None, "request_ids": [], "execution_identity_sha256": []}
    request_ids: list[str] = []
    versions: set[str] = set()
    execution_identities: set[str] = set()
    for call in charged:
        request_id = call.get("request_id")
        version = call.get("rate_book_version")
        execution_identity = call.get("execution_identity_sha256")
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError("A charged SIE request has no request ID")
        if not isinstance(version, str) or not version:
            raise RuntimeError(f"Charged SIE request {request_id} has no rate-book version")
        if not isinstance(execution_identity, str) or not execution_identity:
            raise RuntimeError(f"Charged SIE request {request_id} has no execution identity")
        request_ids.append(request_id)
        versions.add(version)
        execution_identities.add(execution_identity)
    if len(request_ids) != len(set(request_ids)):
        raise RuntimeError("The run contains duplicate charged SIE request IDs")
    if len(versions) != 1:
        raise RuntimeError("The run spans more than one SIE rate-book version")
    return {
        "version": versions.pop(),
        "request_ids": request_ids,
        "execution_identity_sha256": sorted(execution_identities),
    }


def _eligible_gold(gold_ids: tuple[str, ...], lookup: dict[str, Technique]) -> tuple[list[str], list[str]]:
    active = [value for value in gold_ids if value in lookup]
    excluded = [value for value in gold_ids if value not in lookup]
    return active, excluded


def _read_report(
    client: SIEClient,
    config: dict[str, Any],
    report_path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    if report_path.suffix.casefold() not in {".pdf", ".html", ".htm", ".docx"}:
        return report_path.read_text(encoding="utf-8"), []
    started = time.perf_counter()
    response = client.extract(
        config["models"]["parse"],
        Item(id=report_path.stem, document=report_path),
        options={"profile": "default"},
        wait_for_capacity=True,
        provision_timeout_s=float(config["cluster"]["provision_timeout_s"]),
    )
    payload = jsonable(response)
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    markdown = data.get("markdown") if isinstance(data, dict) else None
    if not isinstance(markdown, str) or not markdown.strip():
        raise RuntimeError("Document parser returned no Markdown")
    return markdown, [
        request_record(
            "parse",
            config["models"]["parse"],
            response,
            (time.perf_counter() - started) * 1000,
            function="extract",
        )
    ]


def map_report(config: dict[str, Any], *, report_path: Path, run_id: str) -> Path:
    run_id = _validate_run_id(run_id)
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    sources = ensure_sources(config)
    techniques = load_catalog(sources["attack"])
    lookup = catalog_by_id(techniques)
    final_dir, staging, reservation = _begin_run(run_id)
    calls: list[dict[str, Any]] = []
    try:
        timeout = float(config["cluster"]["provision_timeout_s"])
        with SIEClient(
            config["cluster"]["url"],
            api_key=config["cluster"]["api_key"] or None,
            timeout_s=timeout,
        ) as client:
            report_text, parse_calls = _read_report(client, config, report_path)
            calls.extend(parse_calls)
            (staging / "parsed-report.md").write_text(report_text, encoding="utf-8")
            behaviors, behavior_calls = extract_behaviors(
                client,
                config["models"]["behavior_extract"],
                report_text,
                max_behaviors=int(config["report"]["max_behaviors"]),
                chunk_characters=int(config["report"]["chunk_characters"]),
                provision_timeout_s=timeout,
            )
            calls.extend(behavior_calls)
            behaviors, entity_calls = enrich_entities(
                client,
                config["models"]["entities"],
                behaviors,
                provision_timeout_s=timeout,
            )
            calls.extend(entity_calls)
            catalog_vectors, catalog_calls = encode_texts(
                client,
                config["models"]["retrieve"],
                [technique.candidate_text for technique in techniques],
                instruction=None,
                is_query=False,
                batch_size=int(config["retrieval"]["embedding_batch_size"]),
                provision_timeout_s=timeout,
                stage="catalog_encode",
            )
            calls.extend(catalog_calls)
            query_vectors, query_calls = encode_texts(
                client,
                config["models"]["retrieve"],
                [f"Observed behavior: {row.quote}\nBehavior summary: {row.summary}" for row in behaviors],
                instruction=config["retrieval"]["instruction"],
                is_query=True,
                batch_size=int(config["retrieval"]["embedding_batch_size"]),
                provision_timeout_s=timeout,
                stage="behavior_encode",
            )
            calls.extend(query_calls)
            np.savez_compressed(
                staging / "embeddings.npz",
                catalog=catalog_vectors,
                queries=query_vectors,
                technique_ids=np.asarray([row.technique_id for row in techniques]),
                evidence_sha256=np.asarray([evidence_sha256(row) for row in behaviors]),
            )
            decisions: list[dict[str, Any]] = []
            for index, behavior in enumerate(behaviors):
                dense_candidates = retrieve(
                    query_vectors[index],
                    catalog_vectors,
                    techniques,
                    int(config["retrieval"]["candidate_count"]),
                )
                ranked, rerank_call = rerank(
                    client,
                    config["models"]["rerank"],
                    behavior,
                    dense_candidates,
                    lookup,
                    rerank_count=int(config["retrieval"]["rerank_count"]),
                    provision_timeout_s=timeout,
                )
                calls.append({**rerank_call, "evidence_sha256": evidence_sha256(behavior)})
                decision, verification_calls = verify_mapping(
                    client,
                    config["models"]["verify"],
                    config["models"]["escalate"],
                    behavior,
                    ranked,
                    lookup,
                    verifier_count=int(config["retrieval"]["verifier_count"]),
                    use_escalation=bool(config["report"]["use_escalation"]),
                    provision_timeout_s=timeout,
                )
                calls.extend({**call, "evidence_sha256": evidence_sha256(behavior)} for call in verification_calls)
                row = decision.to_dict()
                row["selected_technique"] = (
                    lookup[decision.selected_technique_id].to_dict()
                    if decision.selected_technique_id is not None
                    else None
                )
                decisions.append(row)

        review = {
            "report": {
                "path": report_path.name,
                "sha256": sha256(report_path),
                "characters": len(report_text),
            },
            "taxonomy": {
                "name": "MITRE ATT&CK Enterprise",
                "version": config["sources"]["attack"]["version"],
                "active_techniques": len(techniques),
            },
            "status": "analyst_review_required",
            "behavior_count": len(behaviors),
            "suggested_mapping_count": sum(row["selected_technique_id"] is not None for row in decisions),
            "mappings": decisions,
        }
        _write_json(staging / "review.json", review)
        _write_json(staging / "api-calls.json", calls)
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "endpoint": config["cluster"]["url"],
            "models": config["models"],
            "model_revisions": config["model_revisions"],
            "rate_book_provenance": _rate_book_provenance(calls),
            "pipeline_stage": "full_report_review",
            "source_report": review["report"],
            "taxonomy": {
                "name": "MITRE ATT&CK Enterprise",
                "version": config["sources"]["attack"]["version"],
                "commit": config["sources"]["attack"]["commit"],
                "sha256": sha256(sources["attack"]),
                "active_techniques": len(techniques),
            },
            "decision_boundary": (
                "The agent proposes source-backed mappings for analyst review. It cannot accept a mapping, detect an "
                "intrusion, or change a security control."
            ),
            "artifacts": _artifact_rows(staging),
        }
        _write_json(staging / "manifest.json", manifest)
        staging.rename(final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        with contextlib.suppress(OSError):
            reservation.rmdir()
    return final_dir


def map_annoctr_demo(
    config: dict[str, Any],
    *,
    document: str,
    split: str,
    run_id: str,
) -> Path:
    sources = ensure_sources(config)
    report_path = find_annoctr_report(sources["annoctr"], split, document)
    return map_report(config, report_path=report_path, run_id=run_id)


def benchmark(
    config: dict[str, Any],
    *,
    split: str,
    limit: int | None,
    stage: str,
    run_id: str,
) -> Path:
    run_id = _validate_run_id(run_id)
    if stage not in {"retrieve", "rerank", "verify"}:
        raise ValueError("stage must be retrieve, rerank, or verify")
    sources = ensure_sources(config)
    benchmark_catalog_path = find_annoctr_catalog(sources["annoctr"])
    techniques = load_annoctr_catalog(benchmark_catalog_path)
    lookup = catalog_by_id(techniques)
    cases = load_linking_cases(sources["annoctr"], split)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        cases = cases[:limit]

    final_dir, staging, reservation = _begin_run(run_id)
    calls: list[dict[str, Any]] = []
    try:
        timeout = float(config["cluster"]["provision_timeout_s"])
        with SIEClient(
            config["cluster"]["url"],
            api_key=config["cluster"]["api_key"] or None,
            timeout_s=timeout,
        ) as client:
            catalog_vectors, catalog_calls = encode_texts(
                client,
                config["models"]["retrieve"],
                [technique.candidate_text for technique in techniques],
                instruction=None,
                is_query=False,
                batch_size=int(config["retrieval"]["embedding_batch_size"]),
                provision_timeout_s=timeout,
                stage="catalog_encode",
            )
            query_vectors, query_calls = encode_texts(
                client,
                config["models"]["retrieve"],
                [case.query_text for case in cases],
                instruction=config["retrieval"]["instruction"],
                is_query=True,
                batch_size=int(config["retrieval"]["embedding_batch_size"]),
                provision_timeout_s=timeout,
                stage="query_encode",
            )
            calls.extend(catalog_calls)
            calls.extend(query_calls)
            np.savez_compressed(
                staging / "embeddings.npz",
                catalog=catalog_vectors,
                queries=query_vectors,
                technique_ids=np.asarray([row.technique_id for row in techniques]),
                case_ids=np.asarray([row.case_id for row in cases]),
            )
            predictions: list[dict[str, Any]] = []
            for index, case in enumerate(cases):
                active_gold, excluded_gold = _eligible_gold(case.gold_ids, lookup)
                dense_candidates = retrieve(
                    query_vectors[index],
                    catalog_vectors,
                    techniques,
                    int(config["retrieval"]["candidate_count"]),
                )
                row: dict[str, Any] = {
                    "case_id": case.case_id,
                    "document": case.document,
                    "mention": case.mention,
                    "evidence": case.evidence,
                    "annotation_classes": list(case.annotation_classes),
                    "gold_ids": active_gold,
                    "excluded_gold_ids": excluded_gold,
                    "retrieval": [candidate.to_dict() for candidate in dense_candidates],
                    "rerank": [],
                    "verification": None,
                }
                if stage in {"rerank", "verify"} and active_gold:
                    behavior = BehaviorEvidence(
                        quote=case.evidence,
                        summary=case.mention,
                        source_start=0,
                        source_end=len(case.evidence),
                    )
                    ranked, rerank_call = rerank(
                        client,
                        config["models"]["rerank"],
                        behavior,
                        dense_candidates,
                        lookup,
                        rerank_count=int(config["retrieval"]["rerank_count"]),
                        provision_timeout_s=timeout,
                    )
                    calls.append({**rerank_call, "case_id": case.case_id})
                    row["rerank"] = [candidate.to_dict() for candidate in ranked]
                    if stage == "verify":
                        decision, verify_calls = verify_mapping(
                            client,
                            config["models"]["verify"],
                            config["models"]["escalate"],
                            behavior,
                            ranked,
                            lookup,
                            verifier_count=int(config["retrieval"]["verifier_count"]),
                            use_escalation=bool(config["report"]["use_escalation"]),
                            provision_timeout_s=timeout,
                        )
                        calls.extend({**call, "case_id": case.case_id} for call in verify_calls)
                        row["verification"] = {
                            "support": decision.support,
                            "route": decision.route,
                            "selected_technique_id": decision.selected_technique_id,
                            "evidence_quote": decision.evidence_quote,
                            "verifier_model": decision.verifier_model,
                            "escalated": decision.escalated,
                        }
                predictions.append(row)

        _write_jsonl(staging / "predictions.jsonl", predictions)
        _write_json(staging / "evaluation.json", evaluate_predictions(predictions))
        _write_json(staging / "api-calls.json", calls)
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "endpoint": config["cluster"]["url"],
            "models": config["models"],
            "model_revisions": config["model_revisions"],
            "rate_book_provenance": _rate_book_provenance(calls),
            "pipeline_stage": stage,
            "dataset": {
                "name": "AnnoCTR",
                "split": split,
                "commit": config["sources"]["annoctr"]["commit"],
                "archive_sha256": sha256(sources["annoctr_archive"]),
                "selection": "all cases sorted by a stable case hash"
                if limit is None
                else f"first {limit} stable case hashes",
            },
            "taxonomy": {
                "name": "AnnoCTR bundled MITRE ATT&CK entity catalog",
                "version": "historical snapshot distributed with AnnoCTR",
                "commit": config["sources"]["annoctr"]["commit"],
                "sha256": sha256(benchmark_catalog_path),
                "techniques": len(techniques),
                "scope": "Historical label space used by the published AnnoCTR annotations",
            },
            "decision_boundary": (
                "Predictions are review suggestions. Only a human analyst can accept or reject an ATT&CK mapping. "
                "The benchmark evaluates linking from an annotated behavior span, not behavior detection over a full report."
            ),
            "artifacts": _artifact_rows(staging),
        }
        _write_json(staging / "manifest.json", manifest)
        staging.rename(final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        with contextlib.suppress(OSError):
            reservation.rmdir()
    return final_dir
