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

from insurance_claims.config import PACKET_DIR, RUNS_DIR, load_claim, load_config, source_by_slug
from insurance_claims.prepare import reconciliation

console = Console()

POLICY_QUERY = (
    "What must the policyholder submit after a flood loss, including the signed proof of loss, "
    "supporting estimates, inventory, photographs, and filing deadline?"
)

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": ["manual_review", "ready_for_adjuster", "return_to_policyholder"],
        },
        "headline": {"type": "string"},
        "claim_summary": {
            "type": "object",
            "properties": {
                "claimed_total": {"type": "number"},
                "attachment_total": {"type": "number"},
                "difference": {"type": "number"},
            },
            "required": ["claimed_total", "attachment_total", "difference"],
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
                            "missing_signature",
                            "amount_mismatch",
                            "policy_deadline",
                            "photo_scope",
                            "other",
                        ],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["blocking", "high", "medium", "low"],
                    },
                    "title": {"type": "string"},
                    "evidence": {"type": "string"},
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "proof_of_loss",
                                "estimate",
                                "policy",
                                "damage_photo",
                                "claim_note",
                            ],
                        },
                    },
                },
                "required": ["category", "severity", "title", "evidence", "sources"],
                "additionalProperties": False,
            },
        },
        "next_actions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["route", "headline", "claim_summary", "findings", "next_actions"],
    "additionalProperties": False,
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")


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
    paragraphs = [paragraph.strip() for paragraph in markdown.split("\n\n") if paragraph.strip()]
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


def _policy_candidates(chunks: list[str], limit: int) -> list[tuple[int, str]]:
    terms = ("proof of loss", "signed", "sworn", "60 days", "estimate", "inventory", "photograph")
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
        Item(id="policy-requirements", text=POLICY_QUERY),
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
        for score in sorted(score_result["scores"], key=lambda item: item["rank"])[:result_limit]
    ]
    return selected, score_result, duration_ms


def _extract_claim_identity(
    client: SIEClient,
    model: str,
    markdown: str,
    provision_timeout_s: float,
) -> tuple[dict[str, Any], float]:
    labels = [
        "insured name",
        "flood insurance policy number",
        "date and time of loss",
        "insured property address",
    ]
    started = time.perf_counter()
    result = client.extract(
        model,
        Item(id="claim-identity", text=markdown[:5000]),
        labels=labels,
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )
    return result, round((time.perf_counter() - started) * 1000, 1)


def _analyze_photo(
    client: SIEClient,
    model: str,
    photo_path: Path,
    provision_timeout_s: float,
) -> tuple[dict[str, Any], str, float]:
    labels = [
        "standing water",
        "flooded room",
        "water damaged wall",
        "damaged furniture",
        "debris",
    ]
    started = time.perf_counter()
    result = client.extract(
        model,
        Item(id="damage-photo", images=[photo_path]),
        labels=labels,
        options={"score_threshold": 0.05},
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    objects = result.get("data", {}).get("objects", result.get("objects", []))
    content = "\n".join(
        f"- {item['label']}: confidence {float(item['score']):.3f}, bbox {item['bbox']}"
        for item in objects
    )
    if not content:
        content = "- No requested damage category exceeded the 0.05 confidence threshold."
    return result, content, duration_ms


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
    form_markdown: str,
    estimate_markdown: str,
    claim_identity: dict[str, Any],
    policy_chunks: list[dict[str, Any]],
    photo_analysis: str,
    claim_note: str,
    totals: dict[str, Any],
    provision_timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    policy_evidence = "\n\n".join(f"[policy chunk {chunk['chunk_id']}]\n{chunk['text']}" for chunk in policy_chunks)
    prompt = f"""
Review this fictional flood claim packet and return the required JSON.

Rules:
- Route the packet for manual review. This is evidence triage, not a coverage or payment decision.
- The proof of loss has no policyholder signature and no signature date.
- Report the arithmetic exactly as supplied below.
- Create a blocking missing_signature finding and a high-severity amount_mismatch finding.
- Cite only the source identifiers allowed by the schema.
- Treat the photograph as limited supporting evidence. Do not claim it proves hidden damage, quantities, or cost.

Reconciliation calculated from the form and attachment line items:
{json.dumps(totals, indent=2, default=_json_default)}

Structured claim identity:
{json.dumps(claim_identity, indent=2, default=_json_default)}

Claim note:
{claim_note}

Proof of loss:
{form_markdown[:10000]}

Repair estimate and contents inventory:
{estimate_markdown[:10000]}

Damage photograph analysis:
{photo_analysis}

Retrieved policy language:
{policy_evidence}

Required JSON schema:
{json.dumps(REVIEW_SCHEMA, indent=2)}
""".strip()
    generation_prompt = f"""<|im_start|>system
You review claim evidence for an insurance operations team. Return sourced discrepancies and the next evidence request. Never make a final coverage, fraud, liability, or payment decision. Return only one JSON object that matches the supplied schema.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
    started = time.perf_counter()
    result = client.generate(
        model,
        generation_prompt,
        max_new_tokens=1500,
        temperature=0,
        top_p=1,
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    content = str(result.get("text", ""))
    return result, _json_object_from_text(content), duration_ms


def _require_packet() -> Path:
    packet_manifest_path = PACKET_DIR / "manifest.json"
    if not packet_manifest_path.exists():
        raise FileNotFoundError("Missing prepared packet. Run `uv run prepare-claim` first.")
    return packet_manifest_path


def run_default_stage(run_id: str) -> Path:
    config = load_config()
    _require_packet()
    run_dir = RUNS_DIR / run_id
    raw_dir = run_dir / "raw"
    markdown_dir = run_dir / "markdown"
    raw_dir.mkdir(parents=True, exist_ok=False)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    parse_client = SIEClient(
        config.cluster.url,
        api_key=config.cluster.api_key or None,
        timeout_s=config.cluster.request_timeout_s,
    )
    timings: dict[str, float] = {}
    try:
        documents = {
            "proof_of_loss": PACKET_DIR / "filled-proof-of-loss.pdf",
            "estimate": PACKET_DIR / "repair-estimate-and-inventory.pdf",
            "policy": source_by_slug(config, "sfip-dwelling-policy").path,
        }
        markdown: dict[str, str] = {}
        for name, path in documents.items():
            result, text, duration_ms = _parse_document(
                parse_client,
                config.models.parse,
                path,
                config.cluster.provision_timeout_s,
            )
            _write_json(raw_dir / f"{name}-parse.json", result)
            (markdown_dir / f"{name}.md").write_text(text.rstrip() + "\n", encoding="utf-8")
            markdown[name] = text
            timings[f"parse_{name}_ms"] = duration_ms

        identity_result, timings["extract_claim_identity_ms"] = _extract_claim_identity(
            parse_client,
            config.models.extract,
            markdown["proof_of_loss"],
            config.cluster.provision_timeout_s,
        )
        _write_json(raw_dir / "claim-identity.json", identity_result)

        policy_chunks, rerank_result, timings["rerank_policy_ms"] = _retrieve_policy(
            parse_client,
            config.models.rerank,
            markdown["policy"],
            chunk_characters=config.retrieval.chunk_characters,
            candidate_limit=config.retrieval.candidate_chunks,
            result_limit=config.retrieval.result_chunks,
            provision_timeout_s=config.cluster.provision_timeout_s,
        )
        _write_json(raw_dir / "policy-rerank.json", rerank_result)
        _write_json(run_dir / "policy-evidence.json", policy_chunks)
    finally:
        parse_client.close()

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
            "claim_identity": identity_result.get("data", identity_result),
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
    claim = load_claim()
    packet_manifest_path = _require_packet()
    run_dir = RUNS_DIR / run_id
    raw_dir = run_dir / "raw"
    markdown_dir = run_dir / "markdown"
    default_stage_path = run_dir / "default-stage.json"
    if not default_stage_path.exists():
        raise FileNotFoundError(f"Missing {default_stage_path}. Run the default stage first.")
    default_stage = json.loads(default_stage_path.read_text(encoding="utf-8"))
    if not default_stage.get("claim_identity"):
        default_stage["claim_identity"] = json.loads(
            (raw_dir / "claim-identity.json").read_text(encoding="utf-8")
        )
    markdown = {
        name: (markdown_dir / f"{name}.md").read_text(encoding="utf-8")
        for name in ("proof_of_loss", "estimate", "policy")
    }
    policy_chunks = json.loads((run_dir / "policy-evidence.json").read_text(encoding="utf-8"))
    timings = dict(default_stage["timings_ms"])

    generation_client = SIEClient(
        config.cluster.generation_url,
        api_key=config.cluster.api_key or None,
        timeout_s=config.cluster.request_timeout_s,
    )
    try:
        photo_raw, photo_analysis, timings["analyze_photo_ms"] = _analyze_photo(
            generation_client,
            config.models.vision,
            PACKET_DIR / "damage-photo.jpg",
            config.cluster.provision_timeout_s,
        )
        _write_json(raw_dir / "photo-analysis.json", photo_raw)
        (run_dir / "photo-analysis.md").write_text(photo_analysis.rstrip() + "\n", encoding="utf-8")

        totals = {key: float(value) for key, value in reconciliation(claim).items()}
        review_raw, review, timings["synthesize_review_ms"] = _final_review(
            generation_client,
            config.models.review,
            form_markdown=markdown["proof_of_loss"],
            estimate_markdown=markdown["estimate"],
            claim_identity=default_stage["claim_identity"],
            policy_chunks=policy_chunks,
            photo_analysis=photo_analysis,
            claim_note=(PACKET_DIR / "claim-note.txt").read_text(encoding="utf-8"),
            totals=totals,
            provision_timeout_s=config.cluster.provision_timeout_s,
        )
        _write_json(raw_dir / "review-completion.json", review_raw)
        _write_json(run_dir / "review.json", review)
    finally:
        generation_client.close()

    manifest = {
        "run_id": run_id,
        "run_at": datetime.now(UTC).isoformat(),
        "fictional_claim": True,
        "endpoints": {
            "cluster": default_stage["endpoint"],
            "generation": config.cluster.generation_url,
        },
        "models": {
            **default_stage["models"],
            "vision": config.models.vision,
            "review": config.models.review,
        },
        "timings_ms": timings,
        "source_manifest": "source-manifest.json",
        "packet_manifest": "packet-manifest.json",
        "review": "review.json",
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(
        run_dir / "packet-manifest.json",
        json.loads(packet_manifest_path.read_text(encoding="utf-8")),
    )
    _write_json(
        run_dir / "source-manifest.json",
        json.loads((PACKET_DIR.parent / "source-manifest.json").read_text(encoding="utf-8")),
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
    parser = argparse.ArgumentParser(description="Review the fictional flood claim through SIE")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--stage",
        choices=("all", "default", "generation"),
        default="all",
        help="Run both stages, or release the GPU between the default and generation bundles",
    )
    args = parser.parse_args()
    selected_run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
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
