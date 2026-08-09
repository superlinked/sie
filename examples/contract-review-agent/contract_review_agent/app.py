"""Assemble the multi-agent app: an orchestrator on one model, a risk-analyst
sub-agent on another, SIE-backed tools, a safety guardrail, and a structured
output type."""

from __future__ import annotations

from typing import Any

from agents import Agent, ModelSettings, Runner, RunResult
from pydantic import BaseModel

from .guardrails import safety_guardrail
from .runtime import AppContext, model_for, provision_timeout_from
from .tools import ALL_TOOLS


class RiskFlag(BaseModel):
    clause: str
    issue: str
    severity: str  # low | medium | high
    suggested_redline: str


class ContractReview(BaseModel):
    """The structured deliverable the orchestrator must produce."""

    document_type: str
    parties: list[str]
    effective_date: str  # "unknown" if not stated
    renewal_terms: str
    governing_law: str  # "unknown" if not stated
    executed: bool  # is the signature page signed and dated?
    key_obligations: list[str]
    risk_flags: list[RiskFlag]
    recommendation: str


# The investigator has NO output_type on purpose: a structured output_type gives a
# weak model an escape hatch to emit the schema immediately instead of using tools.
# With only tools available, it must call them to do its job.
_INVESTIGATOR_INSTRUCTIONS = """\
You are a contract investigator. You have NO prior knowledge of this contract — the
ONLY way to learn anything is to CALL YOUR TOOLS. Investigate thoroughly: call EVERY
one of these tools, one after another, before you write anything.

- classify_document() — the document type
- ocr_signature_page() — read the executed signature page (signatories, titles, date)
- extract_entities() — parties, dates, amounts, governing law
- read_signature_page("Are both parties' signatures present and dated?") — visual execution check
- search_clauses("automatic renewal"), then search_clauses("limitation of liability"),
  then search_clauses("indemnification"), then search_clauses("termination")
- analyze_clause_risks() — risk analysis over the clauses returned by those searches
- query_obligations_db("upcoming obligations with due dates and amounts") — deadlines

Do NOT write your report until you have called them all. Then write a thorough,
factual findings report that cites ONLY what the tools returned. Never invent a party,
date, number, or clause — if a tool failed, say so."""

_SYNTHESIZER_INSTRUCTIONS = """\
You turn a contract investigator's findings into a structured ContractReview. Use
ONLY the findings provided — never add facts. If the findings don't establish a
field, use "unknown" (or false for `executed`). Make key_obligations and risk_flags
specific and grounded in the findings, and give a clear recommendation."""

_INVESTIGATOR_TOOL_SEQUENCE = (
    ("classify_document", None),
    ("ocr_signature_page", None),
    ("extract_entities", None),
    ("read_signature_page", "Are both parties' signatures present and dated?"),
    ("search_clauses", "automatic renewal"),
    ("search_clauses", "limitation of liability"),
    ("search_clauses", "indemnification"),
    ("search_clauses", "termination"),
    ("analyze_clause_risks", None),
    ("query_obligations_db", "upcoming obligations with due dates and amounts"),
)


def build_reasoning_agent(
    cfg: dict[str, Any], client: Any, api_calls: list[dict[str, Any]] | None = None
) -> Agent:
    return Agent(
        name="Risk Analyst",
        instructions=(
            "You are a senior contracts attorney. Given contract clauses, identify "
            "risks to the Customer. For each, state the clause, the issue, a severity "
            "(low/medium/high), and a concrete one-line redline. Be specific and brief."
        ),
        model=model_for(
            cfg["models"]["reasoning"],
            client,
            stage="analyze_clause_risks",
            provision_timeout_s=provision_timeout_from(cfg),
            api_calls=api_calls,
        ),
        model_settings=ModelSettings(temperature=0, max_tokens=1200),
    )


def build_investigator(
    cfg: dict[str, Any], client: Any, api_calls: list[dict[str, Any]] | None = None
) -> Agent:
    """Autonomous tool-using agent (no output_type) that gathers grounded findings."""
    return Agent(
        name="Contract Investigator",
        instructions=_INVESTIGATOR_INSTRUCTIONS,
        model=model_for(
            cfg["models"]["orchestrator"],
            client,
            stage="investigator_report",
            provision_timeout_s=provision_timeout_from(cfg),
            required_tool_sequence=_INVESTIGATOR_TOOL_SEQUENCE,
            api_calls=api_calls,
        ),
        model_settings=ModelSettings(temperature=0, max_tokens=2048),
        tools=ALL_TOOLS,
        input_guardrails=[safety_guardrail],
    )


def build_synthesizer(
    cfg: dict[str, Any], client: Any, api_calls: list[dict[str, Any]] | None = None
) -> Agent:
    """Structured-output agent (no tools) that formats the findings into a review."""
    return Agent(
        name="Contract Reviewer",
        instructions=_SYNTHESIZER_INSTRUCTIONS,
        model=model_for(
            cfg["models"]["orchestrator"],
            client,
            stage="synthesize_review",
            provision_timeout_s=provision_timeout_from(cfg),
            api_calls=api_calls,
        ),
        model_settings=ModelSettings(temperature=0, max_tokens=2400),
        output_type=ContractReview,
    )


async def run_review(
    app: AppContext, investigator: Agent, synthesizer: Agent, instruction: str
) -> tuple[RunResult, RunResult]:
    """Investigate with tools (autonomous fan-out), then synthesize the structured review."""
    gather = await Runner.run(
        investigator,
        f"{instruction}\n\nInvestigate the contract using your tools, then report your findings.",
        context=app,
        max_turns=20,
    )
    synth = await Runner.run(
        synthesizer,
        f"Investigator findings:\n\n{gather.final_output}\n\nProduce the ContractReview.",
        context=app,
    )
    return gather, synth
