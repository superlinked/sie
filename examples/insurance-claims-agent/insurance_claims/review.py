from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from sie_sdk import SIEClient
from sie_sdk.types import Item

from insurance_claims.config import (
    DATA_DIR,
    RUNS_DIR,
    load_config,
    source_by_slug,
)

console = Console()

POLICY_QUERY = (
    "What does the Standard Flood Insurance Policy cover for removal of non-owned flood debris "
    "on or in insured property, and which transport, disposal, or yard-removal costs are excluded?"
)

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": [
                "scope_review_required",
                "affirm_denial",
                "insufficient_record",
            ],
        },
        "headline": {"type": "string"},
        "appeal_summary": {
            "type": "object",
            "properties": {
                "proof_of_loss_amount": {"type": "number"},
                "removal_estimate": {"type": "number"},
                "barge_estimate": {"type": "number"},
                "debris_cubic_yards_min": {"type": "integer"},
                "debris_cubic_yards_max": {"type": "integer"},
            },
            "required": [
                "proof_of_loss_amount",
                "removal_estimate",
                "barge_estimate",
                "debris_cubic_yards_min",
                "debris_cubic_yards_max",
            ],
            "additionalProperties": False,
        },
        "decision": {
            "type": "object",
            "properties": {
                "covered_scope": {"type": "string"},
                "excluded_scope": {"type": "string"},
                "evidence_needed": {"type": "string"},
                "prior_claim_check": {"type": "string"},
            },
            "required": [
                "covered_scope",
                "excluded_scope",
                "evidence_needed",
                "prior_claim_check",
            ],
            "additionalProperties": False,
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "covered_removal",
                            "excluded_transport",
                            "price_support",
                            "prior_claim_overlap",
                            "other",
                        ],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["decision", "high", "medium", "low"],
                    },
                    "title": {"type": "string"},
                    "evidence": {"type": "string"},
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["appeal_decision", "policy"],
                        },
                    },
                },
                "required": [
                    "category",
                    "severity",
                    "title",
                    "evidence",
                    "sources",
                ],
                "additionalProperties": False,
            },
        },
        "next_actions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "route",
        "headline",
        "appeal_summary",
        "decision",
        "findings",
        "next_actions",
    ],
    "additionalProperties": False,
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _parse_document(
    client: SIEClient,
    model: str,
    path: Path,
    provision_timeout_s: float,
) -> tuple[dict[str, Any], str, float]:
    started = time.perf_counter()
    result = client.extract(
        model,
        Item(id=path.stem, document=path),
        options={"profile": "default"},
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    if result.get("error"):
        raise RuntimeError(f"{path.name}: {result['error']}")
    markdown = str(result.get("data", {}).get("markdown", ""))
    if not markdown.strip():
        raise RuntimeError(f"{path.name}: parser returned no Markdown")
    return result, markdown, duration_ms


def chunk_markdown(markdown: str, target_characters: int) -> list[str]:
    paragraphs = [
        paragraph.strip()
        for paragraph in markdown.split("\n\n")
        if paragraph.strip()
    ]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in paragraphs:
        paragraph_size = len(paragraph) + 2
        if current and current_size + paragraph_size > target_characters:
            chunks.append("\n\n".join(current))
            current = []
            current_size = 0
        current.append(paragraph)
        current_size += paragraph_size
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _policy_candidates(
    chunks: list[str],
    limit: int,
) -> list[tuple[int, str]]:
    terms = (
        "debris",
        "non-owned",
        "removal",
        "insured property",
        "disposal",
        "flood-borne",
        "building",
        "yard",
    )
    ranked = sorted(
        enumerate(chunks),
        key=lambda row: sum(term in row[1].casefold() for term in terms),
        reverse=True,
    )
    return ranked[:limit]


def _retrieve_policy(
    client: SIEClient,
    model: str,
    markdown: str,
    *,
    chunk_characters: int,
    candidate_limit: int,
    result_limit: int,
    provision_timeout_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    chunks = chunk_markdown(markdown, chunk_characters)
    candidates = _policy_candidates(chunks, candidate_limit)
    started = time.perf_counter()
    score_result = client.score(
        model,
        Item(id="debris-removal-scope", text=POLICY_QUERY),
        [Item(id=str(index), text=text) for index, text in candidates],
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    by_index = {str(index): text for index, text in candidates}
    selected = [
        {
            "chunk_id": score["item_id"],
            "rank": score["rank"],
            "score": score["score"],
            "text": by_index[score["item_id"]],
        }
        for score in sorted(
            score_result["scores"],
            key=lambda item: item["rank"],
        )[:result_limit]
    ]
    return selected, score_result, duration_ms


def _extract_claim_facts(
    client: SIEClient,
    model: str,
    markdown: str,
    provision_timeout_s: float,
) -> tuple[dict[str, Any], float]:
    labels = [
        "proof of loss amount",
        "debris removal estimate",
        "barge transportation estimate",
        "debris volume",
        "date of loss",
        "covered debris removal scope",
        "excluded debris cost",
    ]
    started = time.perf_counter()
    result = client.extract(
        model,
        Item(id="appeal-facts", text=markdown[:12000]),
        labels=labels,
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )
    return result, round((time.perf_counter() - started) * 1000, 1)


def _json_object_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Review model returned no JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise TypeError("Review model JSON must be an object")
    return value


def _final_review(
    client: SIEClient,
    model: str,
    *,
    appeal_markdown: str,
    claim_facts: dict[str, Any],
    policy_chunks: list[dict[str, Any]],
    provision_timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    policy_evidence = "\n\n".join(
        f"[policy chunk {chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in policy_chunks
    )
    prompt = f"""
Summarize FEMA Flood Insurance Appeal Decision B8 as a cited operations review.

Read the decision's background, rules, analysis, and conclusion. Separate the
physical work FEMA directed the insurer to cover from transport, handling,
disposal, yard work, and other costs outside that scope. Preserve the three
published dollar amounts and the debris-volume range exactly. Record the
additional price evidence and prior-claim checks FEMA requested.

This is a summary of a completed public appeal. Do not make a new coverage or
payment decision. Cite only the source identifiers allowed by the schema.

Entities extracted from the appeal:
{json.dumps(claim_facts, indent=2, default=_json_default)}

FEMA appeal decision:
{appeal_markdown[:18000]}

Retrieved Standard Flood Insurance Policy language:
{policy_evidence}

Required JSON schema:
{json.dumps(REVIEW_SCHEMA, indent=2)}
""".strip()
    generation_prompt = f"""<|im_start|>system
You summarize public insurance appeal records for an operations team. Return
the published outcome, its exact scope, cited facts, and unresolved evidence.
Never decide a live claim. Return only one JSON object matching the supplied
schema.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
    started = time.perf_counter()
    result = client.generate(
        model,
        generation_prompt,
        max_new_tokens=1800,
        temperature=0,
        top_p=1,
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    content = str(result.get("text", ""))
    return result, _json_object_from_text(content), duration_ms


def _require_sources() -> None:
    config = load_config()
    missing = [source.path for source in config.sources if not source.path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing source files: {names}. Run `uv run fetch-claim-sources` first."
        )


def run_default_stage(run_id: str) -> Path:
    config = load_config()
    _require_sources()
    run_dir = RUNS_DIR / run_id
    raw_dir = run_dir / "raw"
    markdown_dir = run_dir / "markdown"
    raw_dir.mkdir(parents=True, exist_ok=False)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    client = SIEClient(
        config.cluster.url,
        api_key=config.cluster.api_key or None,
        timeout_s=config.cluster.request_timeout_s,
    )
    timings: dict[str, float] = {}
    try:
        documents = {
            "appeal_decision": source_by_slug(config, "nfip-appeal-b8").path,
            "policy": source_by_slug(config, "sfip-dwelling-policy").path,
        }
        markdown: dict[str, str] = {}
        for name, path in documents.items():
            result, text, duration_ms = _parse_document(
                client,
                config.models.parse,
                path,
                config.cluster.provision_timeout_s,
            )
            _write_json(raw_dir / f"{name}-parse.json", result)
            (markdown_dir / f"{name}.md").write_text(
                text.rstrip() + "\n",
                encoding="utf-8",
            )
            markdown[name] = text
            timings[f"parse_{name}_ms"] = duration_ms

        facts_result, timings["extract_claim_facts_ms"] = _extract_claim_facts(
            client,
            config.models.extract,
            markdown["appeal_decision"],
            config.cluster.provision_timeout_s,
        )
        _write_json(raw_dir / "claim-facts.json", facts_result)
        _write_json(
            run_dir / "claim-facts.json",
            facts_result.get("data", facts_result),
        )

        policy_chunks, rerank_result, timings["rerank_policy_ms"] = (
            _retrieve_policy(
                client,
                config.models.rerank,
                markdown["policy"],
                chunk_characters=config.retrieval.chunk_characters,
                candidate_limit=config.retrieval.candidate_chunks,
                result_limit=config.retrieval.result_chunks,
                provision_timeout_s=config.cluster.provision_timeout_s,
            )
        )
        _write_json(raw_dir / "policy-rerank.json", rerank_result)
        _write_json(run_dir / "policy-evidence.json", policy_chunks)
    finally:
        client.close()

    _write_json(
        run_dir / "default-stage.json",
        {
            "endpoint": config.cluster.url,
            "models": {
                "parse": config.models.parse,
                "extract": config.models.extract,
                "rerank": config.models.rerank,
            },
            "timings_ms": timings,
        },
    )
    table = Table("Default-bundle call", "Latency")
    for name, duration_ms in timings.items():
        table.add_row(name, f"{duration_ms:,.1f} ms")
    console.print(table)
    console.print(f"Default stage: {run_dir}")
    return run_dir


def run_generation_stage(run_id: str) -> Path:
    config = load_config()
    _require_sources()
    run_dir = RUNS_DIR / run_id
    raw_dir = run_dir / "raw"
    markdown_dir = run_dir / "markdown"
    default_stage_path = run_dir / "default-stage.json"
    if not default_stage_path.exists():
        raise FileNotFoundError(
            f"Missing {default_stage_path}. Run the default stage first."
        )
    default_stage = json.loads(default_stage_path.read_text(encoding="utf-8"))
    appeal_markdown = (markdown_dir / "appeal_decision.md").read_text(
        encoding="utf-8"
    )
    claim_facts = json.loads(
        (run_dir / "claim-facts.json").read_text(encoding="utf-8")
    )
    policy_chunks = json.loads(
        (run_dir / "policy-evidence.json").read_text(encoding="utf-8")
    )
    timings = dict(default_stage["timings_ms"])

    client = SIEClient(
        config.cluster.generation_url,
        api_key=config.cluster.api_key or None,
        timeout_s=config.cluster.request_timeout_s,
    )
    try:
        review_raw, review, timings["synthesize_review_ms"] = _final_review(
            client,
            config.models.review,
            appeal_markdown=appeal_markdown,
            claim_facts=claim_facts,
            policy_chunks=policy_chunks,
            provision_timeout_s=config.cluster.provision_timeout_s,
        )
        _write_json(raw_dir / "review-completion.json", review_raw)
        _write_json(run_dir / "review.json", review)
    finally:
        client.close()

    manifest = {
        "run_id": run_id,
        "run_at": datetime.now(UTC).isoformat(),
        "public_record": True,
        "endpoints": {
            "cluster": default_stage["endpoint"],
            "generation": config.cluster.generation_url,
        },
        "models": {
            **default_stage["models"],
            "review": config.models.review,
        },
        "timings_ms": timings,
        "source_manifest": "source-manifest.json",
        "review": "review.json",
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(
        run_dir / "source-manifest.json",
        json.loads(
            (DATA_DIR / "source-manifest.json").read_text(encoding="utf-8")
        ),
    )

    table = Table("Model call", "Latency")
    for name, duration_ms in timings.items():
        table.add_row(name, f"{duration_ms:,.1f} ms")
    console.print(table)
    console.print(f"Route: {review['route']}")
    console.print(f"Finding: {review['headline']}")
    console.print(f"Run bundle: {run_dir}")
    return run_dir


def run_review(run_id: str | None = None) -> Path:
    selected_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_default_stage(selected_run_id)
    return run_generation_stage(selected_run_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review FEMA Flood Insurance Appeal Decision B8 through SIE"
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--stage",
        choices=("all", "default", "generation"),
        default="all",
        help="Run both stages, or release the GPU between the default and generation bundles",
    )
    args = parser.parse_args()
    selected_run_id = args.run_id or datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    if args.stage == "default":
        run_default_stage(selected_run_id)
    elif args.stage == "generation":
        if not args.run_id:
            parser.error("--run-id is required for --stage generation")
        run_generation_stage(selected_run_id)
    else:
        run_review(selected_run_id)


if __name__ == "__main__":
    main()
