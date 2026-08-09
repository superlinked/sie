"""Run the contract-review agent over one contract and show the model fan-out."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from agents import InputGuardrailTripwireTriggered, set_tracing_disabled
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sie_sdk import SIEAsyncClient

from .app import (
    ContractReview,
    build_investigator,
    build_reasoning_agent,
    build_synthesizer,
    run_review,
)
from .config import load_config
from .data import make_sample
from .data.paths import CUAD_DIR, GENERATED_DIR, MANIFEST_PATH
from .evidence import validate_run_id, write_run_record
from .runtime import AppContext, Ledger, instruct_once, provision_timeout_from

console = Console()

# (config role, human job, SIE function) — drives the catalog table.
ROLE_INFO = [
    ("orchestrator", "Plan, call tools, assemble the review", "generate + tools"),
    ("triage", "Classify the document type (fast)", "generate"),
    ("vision", "Read the scanned signature page", "generate + image"),
    ("reasoning", "Clause-risk specialist (sub-agent)", "generate"),
    ("sql", "Text-to-SQL over the obligations DB", "generate"),
    ("guard", "Safety / prompt-injection guardrail", "generate"),
    ("ocr", "Scanned page → markdown", "extract"),
    ("embed", "Clause search (embeddings)", "encode"),
    ("rerank", "Rerank retrieved clauses", "score"),
    ("entities", "Entity extraction (parties, dates, $)", "extract"),
]


def _print_catalog(cfg: dict) -> None:
    table = Table(
        title="One SIE cluster · the right model for each job", title_style="bold"
    )
    table.add_column("Role", style="cyan")
    table.add_column("SIE catalog model", style="green")
    table.add_column("SIE function", style="magenta")
    table.add_column("Job")
    for role, job, fn in ROLE_INFO:
        table.add_row(role, cfg["models"][role], fn, job)
    console.print(table)


def _print_ledger(ledger: Ledger) -> None:
    table = Table(title="Per-model observability (in call order)", title_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Step")
    table.add_column("Model", style="green")
    table.add_column("SIE fn", style="magenta")
    table.add_column("Warm-up", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Sent", justify="right")
    table.add_column("Got", justify="right")
    table.add_column("Throughput", justify="right")
    warmup_total = 0.0
    call_total = 0.0
    for i, e in enumerate(ledger.entries, 1):
        warmup_total += e.warmup_s
        call_total += e.latency_s
        table.add_row(
            str(i),
            e.step,
            e.model.split("/")[-1],
            e.sie_fn,
            f"{e.warmup_s:.1f}s" if e.warmup_s else "—",
            f"{e.latency_s:.2f}s" if e.latency_s else "—",
            e.sent or "—",
            e.got or "—",
            e.throughput or "—",
        )
    console.print(table)
    used = {e.model for e in ledger.entries}
    console.print(
        f"[bold]{len(used)} distinct SIE models[/] handled this request — "
        f"{warmup_total:.0f}s cold-start (model warm-up) + {call_total:.0f}s warm calls."
    )


def _print_summary(cfg: dict, usage, wall_s: float) -> None:
    parts = [f"end-to-end wall time [bold]{wall_s:.1f}s[/]"]
    reqs = getattr(usage, "requests", None)
    if reqs is not None:
        it = getattr(usage, "input_tokens", 0) or 0
        ot = getattr(usage, "output_tokens", 0) or 0
        orch = cfg["models"]["orchestrator"].split("/")[-1]
        parts.append(
            f"investigator {orch}: {reqs} LLM calls, {it:,} in / {ot:,} out tok"
        )
    console.print("Run summary — " + " · ".join(parts))


def _print_review(review: ContractReview) -> None:
    lines = [
        f"[bold]Document type[/]: {review.document_type}",
        f"[bold]Parties[/]: {', '.join(review.parties) or '—'}",
        f"[bold]Effective date[/]: {review.effective_date or '—'}",
        f"[bold]Governing law[/]: {review.governing_law or '—'}",
        f"[bold]Executed[/]: {review.executed}",
        f"[bold]Renewal terms[/]: {review.renewal_terms}",
        "",
        "[bold]Key obligations[/]:",
        *[f"  • {o}" for o in review.key_obligations],
        "",
        "[bold]Recommendation[/]:",
        f"  {review.recommendation}",
    ]
    console.print(Panel("\n".join(lines), title="Contract review", border_style="blue"))

    if review.risk_flags:
        risks = Table(title="Risk flags", title_style="bold red")
        risks.add_column("Severity", style="bold")
        risks.add_column("Clause")
        risks.add_column("Issue")
        risks.add_column("Suggested redline")
        sev_color = {"high": "red", "medium": "yellow", "low": "green"}
        for f in review.risk_flags:
            color = sev_color.get(f.severity.lower(), "white")
            risks.add_row(
                f"[{color}]{f.severity}[/]", f.clause, f.issue, f.suggested_redline
            )
        console.print(risks)


def _resolve_corpus(args) -> tuple[str, str, str, str]:
    """Return (contract_text, scan_path, db_path, label)."""
    # Explicit file path wins.
    if args.contract and Path(args.contract).is_file():
        p = Path(args.contract)
        scan = args.scan or str(GENERATED_DIR / "acme-msa-signature.png")
        return p.read_text(), scan, str(GENERATED_DIR / "obligations.db"), p.name

    if MANIFEST_PATH.exists():  # real CUAD corpus
        manifest = json.loads(MANIFEST_PATH.read_text())
        slug = args.contract or manifest["primary"]
        text = (CUAD_DIR / f"{slug}.txt").read_text()
        scan = args.scan or str(GENERATED_DIR / manifest["scan_path"])
        return text, scan, str(GENERATED_DIR / manifest["db_path"]), f"CUAD · {slug}"

    # Offline fallback: synthetic corpus (generate it if missing).
    if not (GENERATED_DIR / "acme-msa.md").exists():
        console.print(
            "[yellow]No corpus found — generating the synthetic one. "
            "Run `uv run fetch-contracts` for real CUAD contracts.[/]"
        )
        make_sample.main()
    name = args.contract or "acme-msa"
    text = (GENERATED_DIR / f"{name}.md").read_text()
    scan = args.scan or str(GENERATED_DIR / "acme-msa-signature.png")
    return text, scan, str(GENERATED_DIR / "obligations.db"), f"synthetic · {name}"


def _list_contracts() -> None:
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        console.print(f"[bold]CUAD corpus[/] ({manifest['license']}):")
        for c in manifest["contracts"]:
            console.print(
                f"  {c['slug']}  [dim]{c['type']} · {c['char_len']:,} chars[/]"
            )
    elif (GENERATED_DIR / "acme-msa.md").exists():
        console.print("[bold]Synthetic corpus[/]: acme-msa, mutual-nda, acme-sow")
    else:
        console.print(
            "No corpus yet. Run `uv run fetch-contracts` or `uv run make-sample`."
        )


async def _warm(app: AppContext) -> None:
    """Provision agent models through the same native route used by the run."""
    models = app.cfg["models"]
    for model in dict.fromkeys([models["orchestrator"], models["reasoning"]]):
        with console.status(
            f"Warming {model} (first call provisions it on a cold cluster)..."
        ):
            try:
                await instruct_once(
                    app,
                    model,
                    [{"role": "user", "content": "ok"}],
                    stage="warmup",
                    max_tokens=1,
                )
            except Exception as exc:  # noqa: BLE001 - warm-up is best effort.
                console.print(
                    f"[yellow]warm-up: {model} not ready "
                    f"({type(exc).__name__}); will retry during the run.[/]"
                )
    console.print("[green]Warm-up done.[/]\n")


async def _run(args) -> None:
    set_tracing_disabled(True)
    if args.run_id is not None:
        validate_run_id(args.run_id)
    cfg = load_config()
    text, scan_path, db_path, label = _resolve_corpus(args)

    _print_catalog(cfg)
    console.print(
        f"Reviewing [bold]{label}[/] against SIE at [bold]{cfg['cluster']['url']}[/]\n"
    )

    async with SIEAsyncClient(
        cfg["cluster"]["url"],
        api_key=cfg["cluster"]["api_key"] or None,
        timeout_s=provision_timeout_from(cfg),
    ) as sie:
        ledger = Ledger()
        api_calls: list[dict] = []
        app = AppContext(
            sie=sie,
            cfg=cfg,
            ledger=ledger,
            contract_text=text,
            scan_path=scan_path,
            db_path=db_path,
            api_calls=api_calls,
            reasoning_agent=build_reasoning_agent(cfg, sie, api_calls),
        )
        investigator = build_investigator(cfg, sie, api_calls)
        synthesizer = build_synthesizer(cfg, sie, api_calls)
        if not args.no_warm:
            await _warm(app)
        t0 = time.monotonic()
        try:
            gather, result = await run_review(
                app, investigator, synthesizer, args.instruction
            )
        except InputGuardrailTripwireTriggered:
            console.print(
                Panel(
                    "Request blocked by the granite-guardian safety guardrail.",
                    border_style="red",
                    title="Guardrail tripped",
                )
            )
            _print_ledger(ledger)
            return
        except Exception as exc:
            console.print(
                Panel(
                    f"{type(exc).__name__}: {exc}",
                    border_style="red",
                    title="Run failed",
                )
            )
            _print_ledger(ledger)
            raise
        wall = time.monotonic() - t0

        try:
            review = result.final_output_as(ContractReview)
        except Exception:  # noqa: BLE001 - preserve unstructured model output.
            review = result.final_output

        console.print()
        if isinstance(review, ContractReview):
            _print_review(review)
        else:
            console.print(Panel(str(review), title="Agent output (unstructured)"))
        console.print()
        _print_ledger(ledger)
        usage = getattr(getattr(gather, "context_wrapper", None), "usage", None)
        _print_summary(cfg, usage, wall)
        if args.run_id is not None:
            if not isinstance(review, ContractReview):
                raise RuntimeError("Cannot record an unstructured contract review")
            run_dir = write_run_record(
                run_id=args.run_id,
                endpoint=cfg["cluster"]["url"],
                cfg=cfg,
                label=label,
                contract_text=text,
                scan_path=scan_path,
                db_path=db_path,
                findings=str(gather.final_output),
                review=review,
                ledger=ledger,
                api_calls=api_calls,
                wall_s=wall,
            )
            console.print(f"Checked run evidence: [bold]{run_dir}[/]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review a contract with a multi-model SIE agent."
    )
    parser.add_argument(
        "--contract",
        default=None,
        help="contract slug (CUAD), synthetic name, or a path to a .txt/.md file",
    )
    parser.add_argument(
        "--scan", default=None, help="path to a signature-page image (png/jpg)"
    )
    parser.add_argument(
        "--instruction",
        default="Review this contract. Identify the parties and key terms, flag the "
        "biggest risks to the Customer with severity and redlines, confirm it is "
        "executed, and surface upcoming obligations and deadlines.",
        help="what to ask the agent to do",
    )
    parser.add_argument(
        "--list", action="store_true", help="list available contracts and exit"
    )
    parser.add_argument(
        "--no-warm",
        action="store_true",
        help="skip pre-warming models (faster when the cluster is already warm)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        type=validate_run_id,
        help="write reproducible evidence under runs/<run-id> after a passing run",
    )
    args = parser.parse_args()

    if args.list:
        _list_contracts()
        return
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
