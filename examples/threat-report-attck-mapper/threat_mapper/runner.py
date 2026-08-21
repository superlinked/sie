from __future__ import annotations

import contextlib
import json
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sie_sdk import SIEClient
from sie_sdk.types import Item

from .catalog import catalog_by_id, load_annoctr_catalog, load_catalog
from .config import RUNS_DIR
from .data import (
    ensure_sources,
    find_annoctr_catalog,
    find_annoctr_report,
    list_annoctr_reports,
    load_gold_mentions,
    load_linking_cases,
    load_training_examples,
    sha256,
)
from .evaluation import evaluate_full_report_predictions, evaluate_predictions
from .models import BehaviorEvidence, Technique
from .pipeline import (
    evidence_sha256,
    extract_behaviors,
    extract_document_entities,
    rerank,
    retrieve,
    retrieve_exemplars,
    retrieve_hybrid,
    verify_mapping,
)
from .sie import encode_multivectors, encode_texts, jsonable, request_record


def write_json(path: Path, value: Any) -> None:
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


def _publish_failed_run(final_dir: Path, staging: Path, error: BaseException) -> None:
    failure_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "post_processing_failed",
        "error": {"type": type(error).__name__, "message": str(error)},
        "artifacts": _artifact_rows(staging),
    }
    with contextlib.suppress(OSError, TypeError, ValueError):
        write_json(staging / "manifest.json", failure_manifest)
    with contextlib.suppress(OSError):
        staging.rename(final_dir)


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


def _behavior_exemplar_text(behavior: BehaviorEvidence) -> str:
    return f"Span: {behavior.quote}\nSentence: {behavior.quote}"


def map_report(config: dict[str, Any], *, report_path: Path, run_id: str) -> Path:
    run_id = _validate_run_id(run_id)
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    sources = ensure_sources(config)
    techniques = load_catalog(sources["attack"])
    lookup = catalog_by_id(techniques)
    training_examples = [
        example for example in load_training_examples(sources["annoctr"]) if example.technique_id in lookup
    ]
    final_dir, staging, reservation = _begin_run(run_id)
    calls: list[dict[str, Any]] = []
    preserve_artifacts = False
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
            entities, entity_calls = extract_document_entities(
                client,
                config["models"]["entities"],
                report_text,
                chunk_characters=int(config["report"]["entity_chunk_characters"]),
                overlap_characters=int(config["report"]["chunk_overlap_characters"]),
                provision_timeout_s=timeout,
            )
            calls.extend(entity_calls)
            behaviors, behavior_calls = extract_behaviors(
                client,
                config["models"]["behavior_extract"],
                report_text,
                max_behaviors=int(config["report"]["max_behaviors"]),
                chunk_characters=int(config["report"]["chunk_characters"]),
                overlap_characters=int(config["report"]["chunk_overlap_characters"]),
                document_entities=entities,
                provision_timeout_s=timeout,
            )
            calls.extend(behavior_calls)
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
            catalog_multivectors, late_catalog_calls = encode_multivectors(
                client,
                config["models"]["late_interaction"],
                [technique.candidate_text for technique in techniques],
                is_query=False,
                batch_size=int(config["retrieval"]["multivector_batch_size"]),
                provision_timeout_s=timeout,
                stage="late_catalog_encode",
            )
            calls.extend(late_catalog_calls)
            query_texts = [row.event_text for row in behaviors]
            query_vectors, query_calls = encode_texts(
                client,
                config["models"]["retrieve"],
                query_texts,
                instruction=config["retrieval"]["instruction"],
                is_query=True,
                batch_size=int(config["retrieval"]["embedding_batch_size"]),
                provision_timeout_s=timeout,
                stage="behavior_encode",
            )
            calls.extend(query_calls)
            exemplar_vectors, exemplar_calls = encode_texts(
                client,
                config["models"]["retrieve"],
                [example.embedding_text for example in training_examples],
                instruction=config["retrieval"]["exemplar_instruction"],
                is_query=True,
                batch_size=int(config["retrieval"]["embedding_batch_size"]),
                provision_timeout_s=timeout,
                stage="exemplar_catalog_encode",
            )
            calls.extend(exemplar_calls)
            exemplar_query_vectors, exemplar_query_calls = encode_texts(
                client,
                config["models"]["retrieve"],
                [_behavior_exemplar_text(row) for row in behaviors],
                instruction=config["retrieval"]["exemplar_instruction"],
                is_query=True,
                batch_size=int(config["retrieval"]["embedding_batch_size"]),
                provision_timeout_s=timeout,
                stage="exemplar_behavior_encode",
            )
            calls.extend(exemplar_query_calls)
            query_multivectors, late_query_calls = encode_multivectors(
                client,
                config["models"]["late_interaction"],
                query_texts,
                is_query=True,
                batch_size=int(config["retrieval"]["multivector_batch_size"]),
                provision_timeout_s=timeout,
                stage="late_behavior_encode",
            )
            calls.extend(late_query_calls)
            np.savez_compressed(
                staging / "embeddings.npz",
                catalog=catalog_vectors,
                queries=query_vectors,
                technique_ids=np.asarray([row.technique_id for row in techniques]),
                evidence_sha256=np.asarray([evidence_sha256(row) for row in behaviors]),
            )
            np.savez_compressed(
                staging / "exemplar-embeddings.npz",
                examples=exemplar_vectors,
                queries=exemplar_query_vectors,
                technique_ids=np.asarray([row.technique_id for row in training_examples]),
                documents=np.asarray([row.document for row in training_examples]),
                evidence_sha256=np.asarray([evidence_sha256(row) for row in behaviors]),
            )
            np.savez_compressed(
                staging / "late-interaction.npz",
                **{f"catalog_{index}": value for index, value in enumerate(catalog_multivectors)},
                **{f"query_{index}": value for index, value in enumerate(query_multivectors)},
            )
            ranked_rows: list[list[Any]] = []
            for index, behavior in enumerate(behaviors):
                exemplar_candidates = retrieve_exemplars(
                    exemplar_query_vectors[index],
                    exemplar_vectors,
                    training_examples,
                    lookup,
                    int(config["retrieval"]["exemplar_pool_count"]),
                )
                dense_candidates = retrieve_hybrid(
                    query_vectors[index],
                    catalog_vectors,
                    query_multivectors[index],
                    catalog_multivectors,
                    techniques,
                    dense_count=int(config["retrieval"]["dense_pool_count"]),
                    late_interaction_count=int(config["retrieval"]["late_interaction_pool_count"]),
                    candidate_count=int(config["retrieval"]["candidate_count"]),
                    exemplar_candidates=exemplar_candidates,
                    exemplar_count=int(config["retrieval"]["exemplar_pool_count"]),
                    exemplar_rrf_weight=float(config["retrieval"]["exemplar_rrf_weight"]),
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
                ranked_rows.append(ranked)

            decisions: list[dict[str, Any]] = []
            for behavior, ranked in zip(behaviors, ranked_rows, strict=True):
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
            "labeled_example_index": {
                "dataset": "AnnoCTR",
                "split": "train",
                "commit": config["sources"]["annoctr"]["commit"],
                "examples": len(training_examples),
            },
            "status": "analyst_review_required",
            "behavior_count": len(behaviors),
            "suggested_mapping_count": sum(row["route"] == "suggested_mapping" for row in decisions),
            "analyst_review_count": sum(row["route"] == "analyst_review" for row in decisions),
            "mappings": decisions,
        }
        write_json(staging / "review.json", review)
        write_json(staging / "api-calls.json", calls)
        preserve_artifacts = True
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
        write_json(staging / "manifest.json", manifest)
        staging.rename(final_dir)
    except BaseException as exc:
        if preserve_artifacts:
            _publish_failed_run(final_dir, staging, exc)
        else:
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
    preserve_artifacts = False
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
        write_json(staging / "api-calls.json", calls)
        preserve_artifacts = True
        write_json(staging / "evaluation.json", evaluate_predictions(predictions))
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
        write_json(staging / "manifest.json", manifest)
        staging.rename(final_dir)
    except BaseException as exc:
        if preserve_artifacts:
            _publish_failed_run(final_dir, staging, exc)
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        with contextlib.suppress(OSError):
            reservation.rmdir()
    return final_dir


def full_report_benchmark(
    config: dict[str, Any],
    *,
    split: str,
    limit: int | None,
    run_id: str,
) -> Path:
    run_id = _validate_run_id(run_id)
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    sources = ensure_sources(config)
    catalog_path = find_annoctr_catalog(sources["annoctr"])
    techniques = load_annoctr_catalog(catalog_path)
    lookup = catalog_by_id(techniques)
    training_examples = [
        example for example in load_training_examples(sources["annoctr"]) if example.technique_id in lookup
    ]
    excluded_documents = {str(value) for value in config["evaluation"].get("excluded_documents", [])}
    report_paths = [
        path for path in list_annoctr_reports(sources["annoctr"], split) if path.stem not in excluded_documents
    ]
    if limit is not None:
        report_paths = report_paths[:limit]
    selected_documents = {path.stem for path in report_paths}
    all_gold = load_gold_mentions(sources["annoctr"], split)
    gold_mentions = [row for row in all_gold if row.document in selected_documents and row.technique_id in lookup]
    excluded_gold = [row for row in all_gold if row.document in selected_documents and row.technique_id not in lookup]

    final_dir, staging, reservation = _begin_run(run_id)
    calls: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    reports: dict[str, str] = {}
    preserve_artifacts = False
    try:
        timeout = float(config["cluster"]["provision_timeout_s"])
        with SIEClient(
            config["cluster"]["url"],
            api_key=config["cluster"]["api_key"] or None,
            timeout_s=timeout,
        ) as client:
            report_artifacts = staging / "reports"
            report_artifacts.mkdir()
            entities_by_document: dict[str, list[dict[str, Any]]] = {}
            behaviors_by_document: dict[str, list[BehaviorEvidence]] = {}
            report_sha256: dict[str, str] = {}

            for report_path in report_paths:
                document = report_path.stem
                reports[document] = report_path.read_text(encoding="utf-8")
                report_sha256[document] = sha256(report_path)

            # Start the longest reports first so one multi-chunk report does not
            # become a serial tail after every short report has finished.
            report_items = sorted(
                reports.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )

            # Keep each model resident while it processes every report. A single-GPU
            # development server can then run the same ensemble without reloading a
            # 27B verifier between individual behaviors.
            def extract_entities_job(item: tuple[str, str]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
                document, report_text = item
                entities, entity_calls = extract_document_entities(
                    client,
                    config["models"]["entities"],
                    report_text,
                    chunk_characters=int(config["report"]["entity_chunk_characters"]),
                    overlap_characters=int(config["report"]["chunk_overlap_characters"]),
                    provision_timeout_s=timeout,
                )
                return document, entities, entity_calls

            with ThreadPoolExecutor(max_workers=int(config["concurrency"]["documents"])) as pool:
                entity_results = pool.map(extract_entities_job, report_items)
                for document, entities, entity_calls in entity_results:
                    entities_by_document[document] = entities
                    calls.extend({**call, "document": document} for call in entity_calls)

            def extract_behaviors_job(
                item: tuple[str, str],
            ) -> tuple[str, list[BehaviorEvidence], list[dict[str, Any]]]:
                document, report_text = item
                behaviors, behavior_calls = extract_behaviors(
                    client,
                    config["models"]["behavior_extract"],
                    report_text,
                    max_behaviors=int(config["report"]["max_behaviors"]),
                    chunk_characters=int(config["report"]["chunk_characters"]),
                    overlap_characters=int(config["report"]["chunk_overlap_characters"]),
                    document_entities=entities_by_document[document],
                    provision_timeout_s=timeout,
                )
                return document, behaviors, behavior_calls

            with ThreadPoolExecutor(max_workers=int(config["concurrency"]["documents"])) as pool:
                behavior_results = pool.map(extract_behaviors_job, report_items)
                for document, behaviors, behavior_calls in behavior_results:
                    behaviors_by_document[document] = behaviors
                    calls.extend({**call, "document": document} for call in behavior_calls)

            for report_path in report_paths:
                document = report_path.stem
                write_json(
                    report_artifacts / f"{document}.json",
                    {
                        "document": document,
                        "sha256": report_sha256[document],
                        "characters": len(reports[document]),
                        "entities": entities_by_document[document],
                        "behaviors": [row.to_dict() for row in behaviors_by_document[document]],
                        "predictions": [],
                    },
                )
            write_json(staging / "api-calls.json", calls)
            preserve_artifacts = True

            actionable = [
                (document, behavior) for document, behaviors in behaviors_by_document.items() for behavior in behaviors
            ]
            query_texts = [behavior.event_text for _, behavior in actionable]
            catalog_texts = [technique.candidate_text for technique in techniques]
            catalog_vectors, catalog_calls = encode_texts(
                client,
                config["models"]["retrieve"],
                catalog_texts,
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
                query_texts,
                instruction=config["retrieval"]["instruction"],
                is_query=True,
                batch_size=int(config["retrieval"]["embedding_batch_size"]),
                provision_timeout_s=timeout,
                stage="behavior_encode",
            )
            calls.extend(query_calls)
            exemplar_vectors, exemplar_calls = encode_texts(
                client,
                config["models"]["retrieve"],
                [example.embedding_text for example in training_examples],
                instruction=config["retrieval"]["exemplar_instruction"],
                is_query=True,
                batch_size=int(config["retrieval"]["embedding_batch_size"]),
                provision_timeout_s=timeout,
                stage="exemplar_catalog_encode",
            )
            calls.extend(exemplar_calls)
            exemplar_query_vectors, exemplar_query_calls = encode_texts(
                client,
                config["models"]["retrieve"],
                [_behavior_exemplar_text(behavior) for _, behavior in actionable],
                instruction=config["retrieval"]["exemplar_instruction"],
                is_query=True,
                batch_size=int(config["retrieval"]["embedding_batch_size"]),
                provision_timeout_s=timeout,
                stage="exemplar_behavior_encode",
            )
            calls.extend(exemplar_query_calls)
            catalog_multivectors, late_catalog_calls = encode_multivectors(
                client,
                config["models"]["late_interaction"],
                catalog_texts,
                is_query=False,
                batch_size=int(config["retrieval"]["multivector_batch_size"]),
                provision_timeout_s=timeout,
                stage="late_catalog_encode",
            )
            calls.extend(late_catalog_calls)
            query_multivectors, late_query_calls = encode_multivectors(
                client,
                config["models"]["late_interaction"],
                query_texts,
                is_query=True,
                batch_size=int(config["retrieval"]["multivector_batch_size"]),
                provision_timeout_s=timeout,
                stage="late_behavior_encode",
            )
            calls.extend(late_query_calls)
            np.savez_compressed(
                staging / "catalog-embeddings.npz",
                dense=catalog_vectors,
                queries=query_vectors,
                technique_ids=np.asarray([row.technique_id for row in techniques]),
                evidence_sha256=np.asarray([evidence_sha256(row) for _, row in actionable]),
            )
            np.savez_compressed(
                staging / "exemplar-embeddings.npz",
                examples=exemplar_vectors,
                queries=exemplar_query_vectors,
                technique_ids=np.asarray([row.technique_id for row in training_examples]),
                documents=np.asarray([row.document for row in training_examples]),
                evidence_sha256=np.asarray([evidence_sha256(row) for _, row in actionable]),
            )
            np.savez_compressed(
                staging / "catalog-late-interaction.npz",
                **{f"technique_{index}": value for index, value in enumerate(catalog_multivectors)},
                **{f"query_{index}": value for index, value in enumerate(query_multivectors)},
            )

            candidate_rows: list[tuple[str, BehaviorEvidence, list[Any]]] = []
            for index, (document, behavior) in enumerate(actionable):
                exemplar_candidates = retrieve_exemplars(
                    exemplar_query_vectors[index],
                    exemplar_vectors,
                    training_examples,
                    lookup,
                    int(config["retrieval"]["exemplar_pool_count"]),
                )
                candidates = retrieve_hybrid(
                    query_vectors[index],
                    catalog_vectors,
                    query_multivectors[index],
                    catalog_multivectors,
                    techniques,
                    dense_count=int(config["retrieval"]["dense_pool_count"]),
                    late_interaction_count=int(config["retrieval"]["late_interaction_pool_count"]),
                    candidate_count=int(config["retrieval"]["candidate_count"]),
                    exemplar_candidates=exemplar_candidates,
                    exemplar_count=int(config["retrieval"]["exemplar_pool_count"]),
                    exemplar_rrf_weight=float(config["retrieval"]["exemplar_rrf_weight"]),
                )
                candidate_rows.append((document, behavior, candidates))

            def rerank_job(
                item: tuple[str, BehaviorEvidence, list[Any]],
            ) -> tuple[str, BehaviorEvidence, list[Any], dict[str, Any]]:
                document, behavior, candidates = item
                ranked, rerank_call = rerank(
                    client,
                    config["models"]["rerank"],
                    behavior,
                    candidates,
                    lookup,
                    rerank_count=int(config["retrieval"]["rerank_count"]),
                    provision_timeout_s=timeout,
                )
                return document, behavior, ranked, rerank_call

            ranked_by_key: dict[tuple[str, str], list[Any]] = {}
            with ThreadPoolExecutor(max_workers=int(config["concurrency"]["rerank"])) as pool:
                rerank_results = pool.map(rerank_job, candidate_rows)
                for document, behavior, ranked, rerank_call in rerank_results:
                    ranked_by_key[(document, evidence_sha256(behavior))] = ranked
                    calls.append({**rerank_call, "document": document, "evidence_sha256": evidence_sha256(behavior)})

            def verify_job(
                item: tuple[str, BehaviorEvidence],
            ) -> tuple[str, BehaviorEvidence, Any, list[dict[str, Any]]]:
                document, behavior = item
                key = (document, evidence_sha256(behavior))
                decision, verification_calls = verify_mapping(
                    client,
                    config["models"]["verify"],
                    config["models"]["escalate"],
                    behavior,
                    ranked_by_key[key],
                    lookup,
                    verifier_count=int(config["retrieval"]["verifier_count"]),
                    use_escalation=bool(config["report"]["use_escalation"]),
                    provision_timeout_s=timeout,
                )
                return document, behavior, decision, verification_calls

            decisions_by_key: dict[tuple[str, str], Any] = {}
            with ThreadPoolExecutor(max_workers=int(config["concurrency"]["verify"])) as pool:
                verification_results = pool.map(verify_job, actionable)
                for document, behavior, decision, verification_calls in verification_results:
                    key = (document, evidence_sha256(behavior))
                    decisions_by_key[key] = decision
                    calls.extend(
                        {**call, "document": document, "evidence_sha256": evidence_sha256(behavior)}
                        for call in verification_calls
                    )

            for report_path in report_paths:
                document = report_path.stem
                behaviors = behaviors_by_document[document]
                document_predictions: list[dict[str, Any]] = []
                for behavior in behaviors:
                    base = {"document": document, **behavior.to_dict()}
                    decision = decisions_by_key[(document, evidence_sha256(behavior))]
                    row = {
                        **base,
                        "route": decision.route,
                        "support": decision.support,
                        "selected_technique_id": decision.selected_technique_id,
                        "evidence_quote": decision.evidence_quote,
                        "rationale": decision.rationale,
                        "candidates": [candidate.to_dict() for candidate in decision.candidates],
                        "verifier_model": decision.verifier_model,
                        "escalated": decision.escalated,
                    }
                    document_predictions.append(row)
                predictions.extend(document_predictions)
                write_json(
                    report_artifacts / f"{document}.json",
                    {
                        "document": document,
                        "sha256": report_sha256[document],
                        "characters": len(reports[document]),
                        "entities": entities_by_document[document],
                        "behaviors": [row.to_dict() for row in behaviors],
                        "predictions": document_predictions,
                    },
                )
                _write_jsonl(staging / "predictions.jsonl", predictions)
                write_json(staging / "api-calls.json", calls)

        evaluation = evaluate_full_report_predictions(
            predictions,
            gold_mentions,
            reports,
            excluded_documents=excluded_documents,
            suggestion_annotated_span_precision_gate=float(
                config["evaluation"]["suggestion_annotated_span_precision_gate"]
            ),
            report_pair_behavior_recall_gate=float(config["evaluation"]["report_pair_behavior_recall_gate"]),
            report_pair_family_conditional_finalist_recall_gate=float(
                config["evaluation"]["report_pair_family_conditional_finalist_recall_gate"]
            ),
            dense_pool_count=int(config["retrieval"]["dense_pool_count"]),
            late_interaction_pool_count=int(config["retrieval"]["late_interaction_pool_count"]),
            exemplar_pool_count=int(config["retrieval"]["exemplar_pool_count"]),
        )
        write_json(staging / "evaluation.json", evaluation)
        write_json(staging / "gold-mentions.json", [row.to_dict() for row in gold_mentions])
        preserve_artifacts = True
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "endpoint": config["cluster"]["url"],
            "models": config["models"],
            "model_revisions": config["model_revisions"],
            "rate_book_provenance": _rate_book_provenance(calls),
            "pipeline_stage": "full_report_end_to_end",
            "dataset": {
                "name": "AnnoCTR",
                "split": split,
                "commit": config["sources"]["annoctr"]["commit"],
                "archive_sha256": sha256(sources["annoctr_archive"]),
                "documents": len(report_paths),
                "selection": "documents sorted by source filename after the frozen exclusions",
                "excluded_documents": sorted(excluded_documents),
                "active_gold_mentions": len(gold_mentions),
                "historical_gold_mentions_outside_catalog": len(excluded_gold),
                "training_examples": len(training_examples),
                "training_split": "train",
            },
            "taxonomy": {
                "name": "AnnoCTR bundled MITRE ATT&CK entity catalog",
                "version": "historical snapshot distributed with AnnoCTR",
                "commit": config["sources"]["annoctr"]["commit"],
                "sha256": sha256(catalog_path),
                "techniques": len(techniques),
            },
            "evaluation_contract": {
                "prediction_unit": "one ATT&CK technique suggestion per report with at least one exact cited span",
                "match": "same report and technique ID with at least one overlapping cited span",
                "implicit_annotation_class": "CI",
                "precision_denominator": "unique supported report-technique pairs",
                "gates": evaluation["gates"],
            },
            "decision_boundary": (
                "The agent produces source-backed ATT&CK suggestions. A human analyst accepts or rejects each mapping."
            ),
            "artifacts": _artifact_rows(staging),
        }
        write_json(staging / "manifest.json", manifest)
        staging.rename(final_dir)
    except BaseException as exc:
        if preserve_artifacts:
            with contextlib.suppress(OSError, TypeError, ValueError):
                write_json(staging / "api-calls.json", calls)
            if predictions:
                with contextlib.suppress(OSError, TypeError, ValueError):
                    _write_jsonl(staging / "predictions.jsonl", predictions)
            _publish_failed_run(final_dir, staging, exc)
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        with contextlib.suppress(OSError):
            reservation.rmdir()
    return final_dir
