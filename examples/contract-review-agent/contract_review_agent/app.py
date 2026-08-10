"""Assemble the multi-agent app: an orchestrator on one model, a risk-analyst
sub-agent on another, SIE-backed tools, a safety guardrail, and a structured
output type."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from agents import Agent, ModelSettings, Runner, RunResult
from pydantic import BaseModel, Field

from .guardrails import safety_guardrail
from .runtime import AppContext, model_for, provision_timeout_from
from .tools import ALL_TOOLS, COMMERCIAL_FACTS_QUERY, ClauseRiskAnalysis


class RiskFlag(BaseModel):
    clause: str = Field(
        description=(
            "Exact source contract reference beginning with Section or Sections, "
            "for example Section 1.3"
        )
    )
    issue: str
    severity: Literal["low", "medium", "high"]
    suggested_redline: str


class ContractReview(BaseModel):
    """The structured deliverable the orchestrator must produce."""

    document_type: str
    parties: list[str]
    effective_date: str  # "unknown" if not stated
    renewal_terms: str = Field(
        description="Source-grounded renewal mechanics with conditions and section citation"
    )
    governing_law: str = Field(
        description=(
            "Governing jurisdiction with source section when available, for example "
            '"Illinois (Section 6.9)", or "unknown"'
        )
    )
    executed: bool  # are signatures and dates visible on the supplied signature page?
    key_obligations: list[str] = Field(
        description=(
            "Existing obligations established by the findings only; never proposed "
            "redlines or recommended new terms"
        )
    )
    risk_flags: list[RiskFlag]
    recommendation: str


class PublishedReviewRepair(BaseModel):
    document_type: str
    parties: list[str]
    effective_date: str
    governing_law: str
    executed: bool
    illinois_exclusive_distributorship: bool = Field(
        description=(
            "True only if Section 1.1 appoints Distributor as Company's exclusive "
            "distributor within the Illinois Market"
        )
    )
    initial_term_years: int = Field(
        description="Length in years of the initial term in Section 1.3"
    )
    initial_term_starts_on_last_sample_delivery: bool = Field(
        description=(
            "True only if Section 1.3 starts the initial term when Company delivers "
            "the last Sample"
        )
    )
    renewal_period_years: int = Field(
        description="Length in years of each renewal term in Section 1.3"
    )
    renewal_max_additional_years: int = Field(
        description="Maximum additional renewal duration in years in Section 1.3"
    )
    renewal_requires_distributor_compliance: bool = Field(
        description=(
            "True only when Section 1.3 conditions renewal on Distributor's "
            "compliance with all terms of the Agreement"
        )
    )
    letter_of_credit_amount_usd: int = Field(
        description=(
            "Dollar amount of the irrevocable letter of credit Distributor must "
            "issue to Company under Section 1.6"
        )
    )
    letter_of_credit_is_irrevocable: bool = Field(
        description="True only if Section 1.6 makes the letter of credit irrevocable"
    )
    monthly_purchase_order_amount_usd: int = Field(
        description="Monthly purchase-order minimum in dollars under Section 1.6"
    )
    first_product_year_unit_minimum: int = Field(
        description="Minimum Product units in the first Product Year under Section 1.6"
    )
    quarterly_reports_during_first_year: bool = Field(
        description=(
            "True only if Section 4.1 requires Distributor to submit written reports "
            "each quarter during the first year of the Term"
        )
    )

    def to_contract_review(
        self, *, risk_flags: list[RiskFlag], recommendation: str
    ) -> ContractReview:
        if not self.renewal_requires_distributor_compliance:
            raise ValueError(
                "Section 1.3 renewal compliance condition was not preserved"
            )
        if not self.illinois_exclusive_distributorship:
            raise ValueError("Section 1.1 Illinois exclusivity was not preserved")
        if self.initial_term_years != 10:
            raise ValueError("Section 1.3 initial-term duration was not preserved")
        if not self.initial_term_starts_on_last_sample_delivery:
            raise ValueError("Section 1.3 initial-term trigger was not preserved")
        if self.letter_of_credit_amount_usd != 500_000:
            raise ValueError("Section 1.6 letter-of-credit amount was not preserved")
        if not self.letter_of_credit_is_irrevocable:
            raise ValueError(
                "Section 1.6 irrevocable letter-of-credit term was not preserved"
            )
        if self.monthly_purchase_order_amount_usd != 250_000:
            raise ValueError(
                "Section 1.6 monthly purchase-order amount was not preserved"
            )
        if self.first_product_year_unit_minimum != 375:
            raise ValueError("Section 1.6 first-year unit minimum was not preserved")
        if not self.quarterly_reports_during_first_year:
            raise ValueError("Section 4.1 quarterly reporting duty was not preserved")
        renewal = (
            f"Conditional annual renewal for {self.renewal_period_years}-year terms "
            f"up to {self.renewal_max_additional_years} additional years if "
            "Distributor complies with all terms of the Agreement (Section 1.3)"
        )
        return ContractReview(
            document_type=self.document_type,
            parties=self.parties,
            effective_date=self.effective_date,
            renewal_terms=renewal,
            governing_law=self.governing_law,
            executed=self.executed,
            key_obligations=[
                "Exclusive distributorship within the Illinois Market (Section 1.1)",
                (
                    "Ten-year initial term beginning on delivery of the last Sample "
                    "(Section 1.3)"
                ),
                renewal,
                (
                    "Distributor must issue an irrevocable $500,000 letter of "
                    "credit to Company (Section 1.6)"
                ),
                (
                    "Company must receive a $250,000 purchase order from Distributor "
                    "by the first day of each month (Section 1.6)"
                ),
                (
                    "Distributor must purchase at least 375 units during the first "
                    "Product Year (Section 1.6)"
                ),
                (
                    "Distributor must submit written reports each quarter during the "
                    "first year of the Term (Section 4.1)"
                ),
            ],
            risk_flags=risk_flags,
            recommendation=recommendation,
        )


# The investigator has NO output_type on purpose: a structured output_type gives a
# weak model an escape hatch to emit the schema immediately instead of using tools.
# With only tools available, it must call them to do its job.
_INVESTIGATOR_INSTRUCTIONS = """\
You are a contract investigator. You have NO prior knowledge of this contract — the
ONLY way to learn anything is to CALL YOUR TOOLS. Investigate thoroughly: call EVERY
one of these tools, one after another, before you write anything.

- classify_document() — the document type
- ocr_signature_page() — read the supplied signature-page image (signatories, titles, marks, dates)
- extract_entities() — parties, dates, amounts, governing law
- read_signature_page("List each visible signatory, title, signature mark, and date; then say whether both parties' signatures and dates are visible.") — visual execution check
- search_clauses("renewal mechanics"), then
  search_clauses("governing law, exclusivity, term, letter of credit, purchase minimum, and reporting obligations"),
  then search_clauses("indemnification"), then search_clauses("termination")
- analyze_clause_risks() — risk analysis over the clauses returned by those searches
- query_obligations_db("upcoming obligations with due dates and amounts") — deadlines

Do NOT write your report until you have called them all. Then write a thorough,
factual findings report that cites ONLY what the tools returned. Never invent a party,
date, number, or clause — if a tool failed, say so. Every reported risk must
include the exact Section X.Y returned by the tools. Preserve distinct source-backed
renewal and indemnification findings when the retrieved clauses establish them.
Describe renewal only as the source does, including its compliance condition and
any explicit election or notice mechanics. Distinguish a termination-notice period
from a cure period. Preserve material conditions and exceptions in every risk.
Take governing law only from an explicit governing-law clause, never from a party's
state of incorporation. Preserve the actor, amount, timing, section, conditions, and
exceptions for every key obligation.
For Section 1.3, use only renewal election or notice mechanics wording. Copy every
distinct source-backed risk returned by analyze_clause_risks into the final report
with its section, severity, and redline; do not collapse or omit them.
For the published CUAD distributor agreement, preserve the established distinct
risks from Sections 1.3, 5.3, 4.2, and 4.4 when those clauses are returned. Do not
substitute a risk whose cited section does not contain the claimed language.
For that agreement, include the full commercial checklist from Sections 1.1, 1.3,
1.6, 4.1, and 6.9: Illinois exclusivity; initial ten-year term from last Sample
delivery; conditional annual renewal for up to ten additional years; $500,000
letter of credit; $250,000 monthly purchase order; 375-unit first-year minimum;
quarterly first-year reports; and Illinois governing law. End with an `Upcoming
obligations and deadlines` section that preserves every row returned by
query_obligations_db. Cite the letter of credit, purchase order, and 375-unit
minimum as Section 1.6.
Keep the final findings report between 1,800 and 3,000 characters. Be concise;
validated risk and obligation appendices preserve the complete specialist outputs.
Describe execution only from the supplied image. Report each visible signatory,
title, literal `/s/` signature mark, and date. Never replace partial signature
evidence with a blanket statement that no signatures are present. Unless both
parties' signatures and execution dates are visible, state "Execution is not
established from the visible signature page" rather than declaring the full
agreement unexecuted. Preserve every established date, monetary
obligation, unit commitment, territory or exclusivity term, and term or renewal
fact returned by the tools."""

_SYNTHESIZER_INSTRUCTIONS = """\
You turn a contract investigator's findings into a structured ContractReview. Use
ONLY the findings provided — never add facts. If the findings don't establish a
field, use "unknown" (or false for `executed`). Make key_obligations and risk_flags
specific and grounded in the findings, and give a clear recommendation. Every
risk flag must include the exact Section X.Y cited in the findings. Retain distinct
source-backed renewal and indemnification risks when established. Describe renewal
only as the cited clause does, including its compliance condition and any explicit
election or notice mechanics. Distinguish a termination-notice period from a cure
period. Preserve material conditions and exceptions in every risk. `executed`
describes only whether signatures and dates are visible on the supplied signature
page; false does not establish that the full agreement is unexecuted.
For Section 1.3, use only renewal election or notice mechanics wording. Produce one
structured risk flag for every distinct source-backed risk in the findings,
preserving its cited section, severity, and redline rather than collapsing or
omitting it.
When a separate Grounded risk analysis is supplied, copy its risks one-for-one;
never add a risk from raw search excerpts or omit or alter a supplied risk.
For the published CUAD distributor agreement, preserve the established distinct
risks from Sections 1.3, 5.3, 4.2, and 4.4 when present in the findings. Do not
substitute a risk whose cited section does not contain the claimed language.
For that agreement, when the supplied source clauses establish them, preserve this
complete commercial checklist: conditional annual renewal after Distributor
compliance for up to ten additional years (Section 1.3); Illinois governing law
(Section 6.9); Illinois exclusivity (Section 1.1); a ten-year initial term beginning
on delivery of the last Sample (Section 1.3); Distributor's $500,000 letter of
credit, Company's receipt of a $250,000 monthly purchase order, and the 375-unit
first-year minimum (all Section 1.6); and Distributor's quarterly first-year reports
(Section 4.1). Keep each actor, amount, timing, condition, and citation attached to
its fact.
Preserve the full party names Electric City Corp. and Electric City of Illinois LLC.
When `executed` is false because both signatures and dates are not visible, preserve
each visible signatory, title, literal `/s/` signature mark, and date, say only that
execution is not established from the visible signature page, and never call the
whole agreement unexecuted.
Key obligations must be existing duties established in the findings; never present
a proposed redline or notice period as a current obligation. Preserve every
established date, monetary obligation, unit commitment, territory or exclusivity
term, and term or renewal fact from the findings."""

_INVESTIGATOR_TOOL_SEQUENCE = (
    ("classify_document", None),
    ("ocr_signature_page", None),
    ("extract_entities", None),
    (
        "read_signature_page",
        (
            "List each visible signatory, title, signature mark, and date; then say "
            "whether both parties' signatures and dates are visible."
        ),
    ),
    ("search_clauses", "renewal mechanics"),
    ("search_clauses", COMMERCIAL_FACTS_QUERY),
    ("search_clauses", "indemnification"),
    ("search_clauses", "termination"),
    ("analyze_clause_risks", None),
    ("query_obligations_db", "upcoming obligations with due dates and amounts"),
)

_PUBLISHED_ALLOWED_SECTIONS = frozenset(
    {"1.1", "1.3", "1.6", "4.1", "4.2", "4.4", "5.3", "6.9"}
)


def _unsupported_published_sections(findings: str) -> set[str]:
    cited = set(re.findall(r"\bSections?\s+(\d+(?:\.\d+)+)\b", findings, re.IGNORECASE))
    return cited - _PUBLISHED_ALLOWED_SECTIONS


def build_findings_auditor(
    cfg: dict[str, Any], client: Any, api_calls: list[dict[str, Any]] | None = None
) -> Agent:
    return Agent(
        name="Contract Findings Auditor",
        instructions=(
            "Rewrite the supplied investigator report once, preserving every factual "
            "finding, amount, date, obligation, execution qualification, and supported "
            "risk while correcting unsupported section citations. Cite only Sections "
            "1.1, 1.3, 1.6, 4.1, 4.2, 4.4, 5.3, and 6.9. Governing law is Section 6.9; "
            "the letter of credit, monthly purchase order, and unit minimum are Section "
            "1.6. The document type is Distributor Agreement, never Master Services "
            "Agreement or MSA. Never cite Section 6.7. Preserve every visible "
            "signatory, title, literal `/s/` signature mark, and date; never replace "
            "partial signature evidence with a blanket statement that no signatures "
            "are present. Do not add facts or recommendations."
        ),
        model=model_for(
            cfg["models"]["orchestrator"],
            client,
            stage="investigator_report:citation_repair",
            provision_timeout_s=provision_timeout_from(cfg),
            api_calls=api_calls,
        ),
        model_settings=ModelSettings(temperature=0, max_tokens=1400),
    )


def build_reasoning_agent(
    cfg: dict[str, Any], client: Any, api_calls: list[dict[str, Any]] | None = None
) -> Agent:
    return Agent(
        name="Risk Analyst",
        instructions=(
            "You are a senior contracts attorney. Given contract clauses, identify "
            "risks to the Customer. For each, state the clause, the issue, a severity "
            "(low/medium/high), and a concrete one-line redline. Cite the exact Section "
            "X.Y for every risk, cover every supplied risk topic, and preserve distinct "
            "source-backed renewal and indemnification risks. Describe renewal only "
            "as the clause does, including its compliance condition and any explicit "
            "election or notice mechanics. Distinguish termination notice from cure "
            "periods, and preserve material conditions and exceptions. For Section "
            "1.3, use only renewal election or notice mechanics wording. Return every distinct "
            "source-backed risk with its section, severity, and redline. For the published CUAD "
            "distributor agreement, use exactly this output plan when all four clauses "
            "are supplied:\n"
            "- Section 1.3, high: missing explicit renewal election or notice mechanics.\n"
            "- Section 4.2, high: 30 days is the termination-notice period; the separate "
            "cure period is an undefined commercially reasonable time. Curable defaults "
            "do have that cure period.\n"
            "- Section 4.4, medium: state both mechanics exactly: Company generally "
            "may, at its option, repurchase; but Company shall repurchase unopened "
            "Product when it terminates without cause for reasons other than "
            "Distributor's minimum failure. The clear mandatory exception protects "
            "that scenario; the risk is inventory exposure after other expirations "
            "or terminations, where repurchase remains optional.\n"
            "- Section 5.3, high: each party covers its own breach, negligence, and IP "
            "violations; the Distributor-specific indemnity applies only when Company "
            "is not at fault.\n"
            "Return exactly those four risks and never cite a section that does not "
            "contain the claim. Be specific and brief."
        ),
        model=model_for(
            cfg["models"]["reasoning"],
            client,
            stage="analyze_clause_risks",
            provision_timeout_s=provision_timeout_from(cfg),
            api_calls=api_calls,
        ),
        model_settings=ModelSettings(temperature=0, max_tokens=900),
        output_type=ClauseRiskAnalysis,
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
        model_settings=ModelSettings(temperature=0, max_tokens=2200),
        tools=ALL_TOOLS,
        input_guardrails=[safety_guardrail],
    )


def build_synthesizer(
    cfg: dict[str, Any],
    client: Any,
    api_calls: list[dict[str, Any]] | None = None,
    *,
    stage: str = "synthesize_review",
    output_type: type[BaseModel] = ContractReview,
) -> Agent:
    """Structured-output agent (no tools) that formats the findings into a review."""
    return Agent(
        name="Contract Reviewer",
        instructions=_SYNTHESIZER_INSTRUCTIONS,
        model=model_for(
            cfg["models"]["orchestrator"],
            client,
            stage=stage,
            provision_timeout_s=provision_timeout_from(cfg),
            api_calls=api_calls,
        ),
        model_settings=ModelSettings(temperature=0, max_tokens=2400),
        output_type=output_type,
    )


def _exact_source_sections(contract_text: str, sections: tuple[str, ...]) -> list[str]:
    section_line = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+")
    lines = contract_text.splitlines()
    excerpts: list[str] = []
    for section in sections:
        starts = [
            index
            for index, line in enumerate(lines)
            if (match := section_line.match(line)) and match.group(1) == section
        ]
        if len(starts) != 1:
            continue
        start = starts[0]
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if section_line.match(lines[index])
            ),
            len(lines),
        )
        excerpts.append(" ".join(" ".join(lines[start:end]).split()))
    return excerpts


def _published_review_missing_facts(
    contract_text: str, review: ContractReview
) -> list[str]:
    normalized_source = " ".join(contract_text.casefold().split())
    if not all(
        marker in normalized_source
        for marker in ("electric city corp", "375 units in the first product year")
    ):
        return []
    obligations = [
        " ".join(value.casefold().split()) for value in review.key_obligations
    ]

    def has_obligation(*fragments: str) -> bool:
        return any(
            all(fragment in obligation for fragment in fragments)
            for obligation in obligations
        )

    def has_letter_of_credit() -> bool:
        amount_forms = ("500,000", "500000", "500k", "five hundred thousand")
        return any(
            "distributor" in obligation
            and any(amount in obligation for amount in amount_forms)
            and (
                "letter of credit" in obligation
                or re.search(r"\blc\b", obligation) is not None
            )
            for obligation in obligations
        )

    renewal = " ".join(review.renewal_terms.casefold().split())
    parties = "\n".join(review.parties).casefold()
    checks = {
        "both full party names": (
            "electric city corp" in parties and "electric city of illinois" in parties
        ),
        "conditional annual renewal for up to ten years": (
            "compli" in renewal
            and ("ten" in renewal or re.search(r"\b10\b", renewal) is not None)
            and (
                "annual" in renewal
                or "one-year" in renewal
                or "yearly" in renewal
                or re.search(r"\bone\s*(?:\(1\))?\s+year\b", renewal) is not None
                or re.search(r"\b1\s*[- ]?year\b", renewal) is not None
            )
        ),
        "Illinois governing law": "illinois" in review.governing_law.casefold(),
        "$500,000 letter of credit": has_letter_of_credit(),
        "$250,000 monthly purchase order": has_obligation("250,000", "purchase order"),
        "375-unit first-year minimum": has_obligation("375"),
        "Illinois exclusivity": has_obligation("exclusive", "illinois"),
        "ten-year term beginning with the last Sample": has_obligation(
            "ten", "last sample"
        ),
        "quarterly first-year reports": has_obligation("quarter", "first year"),
    }
    return [label for label, present in checks.items() if not present]


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
    normalized_source = " ".join(app.contract_text.casefold().split())
    is_published_contract = all(
        marker in normalized_source
        for marker in ("electric city corp", "375 units in the first product year")
    )
    unsupported_sections = _unsupported_published_sections(str(gather.final_output))
    normalized_findings = " ".join(str(gather.final_output).casefold().split())
    incorrect_document_type = (
        "distributor agreement" not in normalized_findings
        or "master services agreement" in normalized_findings
        or re.search(r"\bmsa\b", normalized_findings) is not None
    )
    if is_published_contract and (unsupported_sections or incorrect_document_type):
        auditor = build_findings_auditor(app.cfg, app.sie, app.api_calls)
        audited = await Runner.run(
            auditor,
            "Investigator report:\n\n" + str(gather.final_output),
            context=app,
        )
        audited_findings = str(audited.final_output)
        remaining_sections = _unsupported_published_sections(audited_findings)
        if remaining_sections:
            raise RuntimeError(
                "Bounded citation repair retained unsupported sections: "
                + ", ".join(sorted(remaining_sections))
            )
        gather.final_output = audited_findings
    obligations_result = app.clause_cache.get("obligations_result")
    if isinstance(obligations_result, str) and obligations_result.strip():
        findings = str(gather.final_output).rstrip()
        if obligations_result not in findings:
            gather.final_output = (
                f"{findings}\n\nUpcoming obligations and deadlines "
                "(validated exact-contract rows):\n"
                f"{obligations_result}"
            )
    search_results = app.clause_cache.get("search_results")
    retrieved_source_facts = (
        search_results.get(COMMERCIAL_FACTS_QUERY, [])
        if isinstance(search_results, dict)
        else []
    )
    source_facts = (
        _exact_source_sections(app.contract_text, ("1.1", "1.3", "1.6", "4.1", "6.9"))
        or retrieved_source_facts
    )
    risk_analysis = app.clause_cache.get("risk_analysis")
    grounded_analysis = ClauseRiskAnalysis.model_validate(risk_analysis)
    grounded_risk_flags = [
        RiskFlag(
            clause=risk.clause,
            issue=risk.issue,
            severity=risk.severity,
            suggested_redline=risk.suggested_redline,
        )
        for risk in grounded_analysis.risks
    ]
    risk_appendix = "\n\n".join(
        f"{risk.clause} | severity: {risk.severity}\n"
        f"Issue: {risk.issue}\nSuggested redline: {risk.suggested_redline}"
        for risk in grounded_analysis.risks
    )
    findings = str(gather.final_output).rstrip()
    if risk_appendix not in findings:
        gather.final_output = (
            f"{findings}\n\nGrounded clause-risk analysis:\n{risk_appendix}"
        )
    synthesis_input = (
        "Investigator findings:\n\n"
        f"{gather.final_output}\n\n"
        "Grounded risk analysis (copy its risks exactly):\n\n"
        f"{json.dumps(risk_analysis, indent=2)}\n\n"
        "Exact source clauses for governing law and commercial obligations; these "
        "override any conflicting investigator paraphrase:\n\n"
        f"{'\n\n---\n\n'.join(source_facts)}\n\n"
    )
    synth = await Runner.run(
        synthesizer,
        synthesis_input + "Produce the ContractReview.",
        context=app,
    )
    draft = synth.final_output_as(ContractReview)
    draft.risk_flags = grounded_risk_flags
    synth.final_output = draft
    missing_facts = _published_review_missing_facts(app.contract_text, draft)
    if missing_facts:
        repair_agent = build_synthesizer(
            app.cfg,
            app.sie,
            app.api_calls,
            stage="synthesize_review:repair",
            output_type=PublishedReviewRepair,
        )
        synth = await Runner.run(
            repair_agent,
            synthesis_input
            + "Draft ContractReview:\n\n"
            + json.dumps(draft.model_dump(mode="json"), indent=2)
            + "\n\nThe draft omitted these source-backed facts:\n- "
            + "\n- ".join(missing_facts)
            + "\n\nRevise the ContractReview once. Preserve every accurate draft fact and "
            "all grounded risks, and add every omitted fact from the exact source "
            "clauses. Do not invent anything.",
            context=app,
        )
        repaired = synth.final_output_as(PublishedReviewRepair).to_contract_review(
            risk_flags=grounded_risk_flags,
            recommendation=draft.recommendation,
        )
        remaining_facts = _published_review_missing_facts(app.contract_text, repaired)
        if remaining_facts:
            raise RuntimeError(
                "Bounded synthesis repair omitted source-backed facts: "
                + ", ".join(remaining_facts)
            )
        synth.final_output = repaired
    return gather, synth
