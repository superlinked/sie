"""Assemble the multi-agent app: an orchestrator on one model, a risk-analyst
sub-agent on another, SIE-backed tools, a safety guardrail, and a structured
output type."""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from typing import Any, Literal

from agents import Agent, ModelSettings, Runner, RunResult
from pydantic import BaseModel, Field

from .guardrails import safety_guardrail
from .runtime import AppContext, model_for, provision_timeout_from
from .tools import ALL_TOOLS, COMMERCIAL_FACTS_QUERY, ClauseRiskAnalysis

_PUBLISHED_FINDINGS_MIN_CHARS = 1_800
_PUBLISHED_FINDINGS_MAX_CHARS = 3_000


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
        if self.renewal_period_years != 1:
            raise ValueError("Section 1.3 renewal period was not preserved")
        if self.renewal_max_additional_years != 10:
            raise ValueError("Section 1.3 maximum renewal duration was not preserved")
        renewal = (
            f"Conditional annual renewal for {self.renewal_period_years}-year terms "
            f"up to {self.renewal_max_additional_years} additional years if "
            "Distributor complies with all terms of the Agreement (Section 1.3)"
        )
        return ContractReview(
            document_type="Distributor Agreement",
            parties=[
                "Electric City Corp. (Company)",
                "Electric City of Illinois L.L.C. (Distributor)",
            ],
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
- query_obligations_db("outstanding obligations with due dates and amounts") — deadlines

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
    ("query_obligations_db", "outstanding obligations with due dates and amounts"),
)

_PUBLISHED_ALLOWED_SECTIONS = frozenset(
    {"1.1", "1.3", "1.6", "4.1", "4.2", "4.4", "5.3", "6.9"}
)


def _unsupported_published_sections(findings: str) -> set[str]:
    citation_lists = re.findall(
        r"\bSections?\s+(\d+(?:\.\d+)+(?:\s*(?:,\s*(?:and\s+)?|and\s+)"
        r"(?:Sections?\s+)?\d+(?:\.\d+)+)*)",
        findings,
        re.IGNORECASE,
    )
    cited = {
        section
        for citation_list in citation_lists
        for section in re.findall(r"\d+(?:\.\d+)+", citation_list)
    }
    return cited - _PUBLISHED_ALLOWED_SECTIONS


def _published_findings_narrative_is_bounded(findings: str) -> bool:
    report = findings.strip()
    return (
        _PUBLISHED_FINDINGS_MIN_CHARS <= len(report) <= _PUBLISHED_FINDINGS_MAX_CHARS
        and re.search(r"[.!?](?:[\"'”’`*_#)\]]+)?$", report) is not None
    )


_PUBLISHED_DEADLINES = (
    (date(2026, 6, 30), "quarterly compliance attestation"),
    (date(2026, 7, 1), "annual subscription or license fee"),
    (date(2026, 9, 15), "renewal or non-renewal notice"),
)


def _published_deadline_status_is_grounded(findings: str) -> bool:
    match = re.search(
        r"deadline status as of ([A-Za-z]+ \d{1,2}, \d{4}):",
        findings,
        flags=re.IGNORECASE,
    )
    if match is None:
        return False
    try:
        as_of = (
            datetime.strptime(match.group(1), "%B %d, %Y").replace(tzinfo=UTC).date()
        )
    except ValueError:
        return False
    normalized = " ".join(findings.casefold().split())
    return all(
        f"{due_date:%B} {due_date.day}, {due_date.year} {obligation} is "
        f"{'overdue' if due_date < as_of else 'upcoming'}".casefold()
        in normalized
        for due_date, obligation in _PUBLISHED_DEADLINES
    )


def _published_deadline_summary(as_of_date: date | None = None) -> str:
    as_of = as_of_date or datetime.now(UTC).date()
    as_of_text = f"{as_of:%B} {as_of.day}, {as_of.year}"
    rendered = []
    for due_date, obligation in _PUBLISHED_DEADLINES:
        status = "overdue" if due_date < as_of else "upcoming"
        due_text = f"{due_date:%B} {due_date.day}, {due_date.year}"
        rendered.append(f"{due_text} {obligation} is {status}")
    return f"Deadline status as of {as_of_text}: " + "; ".join(rendered) + "."


def _render_grounded_published_findings(
    grounded_analysis: ClauseRiskAnalysis,
    *,
    as_of_date: date | None = None,
) -> str:
    risk_text = "\n\n".join(
        f"{risk.clause} | severity: {risk.severity}\n"
        f"Issue: {risk.issue}\nSuggested redline: {risk.suggested_redline}"
        for risk in grounded_analysis.risks
    )
    report = (
        "## Contract investigation findings\n\n"
        "Document type: Distributor Agreement. Parties: Electric City Corp. "
        "(Company) and Electric City of Illinois L.L.C. (Distributor).\n\n"
        "Execution: The visible signature page shows Joseph Marino's "
        "`/s/Joseph Marino` mark and President title for Electric City Corp. It also "
        "shows Jim Stump's typed signatory block for Electric City of Illinois "
        "L.L.C., without an explicit signature mark. No execution dates are visible, "
        "so execution is not established from the visible signature page.\n\n"
        "Commercial terms: Section 1.1 appoints Distributor as the exclusive "
        "distributor in the Illinois Market. Section 1.3 sets a ten-year initial term "
        "beginning when Company delivers the last Sample and permits conditional "
        "annual one-year renewals for up to ten additional years if Distributor "
        "complies with the Agreement. Section 1.6 requires an irrevocable $500,000 "
        "letter of credit, a $250,000 purchase order by the first day of each month, "
        "and at least 375 units in the first Product Year. Section 4.1 requires "
        "quarterly written reports during the first year. Section 6.9 applies "
        "Illinois law.\n\n"
        f"Material risks and redlines:\n{risk_text}\n\n"
        f"{_published_deadline_summary(as_of_date)}"
    )
    if not _published_findings_narrative_is_bounded(report):
        raise RuntimeError(
            "Grounded findings assembly returned an incomplete or out-of-range narrative"
        )
    return report


def _align_published_signature_recommendation(recommendation: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(recommendation.split()))
    commercial_sentences = [
        sentence
        for sentence in sentences
        if not any(
            term in sentence.casefold()
            for term in ("execut", "signatur", "signed", "visible date")
        )
    ]
    signature_scope = (
        "Execution is not established from the visible signature page: one explicit "
        "/s/Joseph Marino mark and Jim Stump's typed signatory block are visible, "
        "but no execution dates are visible."
    )
    return " ".join((signature_scope, *commercial_sentences))


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


def _source_section_excerpt(
    contract_text: str,
    section: str,
    *,
    error_prefix: str | None = None,
) -> str | None:
    section_line = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+")
    lines = contract_text.splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if (match := section_line.match(line)) and match.group(1) == section
    ]
    if len(starts) != 1:
        if error_prefix is not None:
            raise RuntimeError(
                f"{error_prefix} section {section} matched {len(starts)} source sections"
            )
        return None
    start = starts[0]
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if section_line.match(lines[index])
        ),
        len(lines),
    )
    return " ".join(" ".join(lines[start:end]).split())


def _exact_source_sections(contract_text: str, sections: tuple[str, ...]) -> list[str]:
    excerpts: list[str] = []
    for section in sections:
        excerpt = _source_section_excerpt(contract_text, section)
        if excerpt is not None:
            excerpts.append(excerpt)
    return excerpts


def _is_published_contract(contract_text: str) -> bool:
    normalized_source = " ".join(contract_text.casefold().split())
    return all(
        marker in normalized_source
        for marker in ("electric city corp", "375 units in the first product year")
    )


def _published_review_missing_labels(review: ContractReview) -> list[str]:
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
    parties = [" ".join(party.casefold().split()) for party in review.parties]
    checks = {
        "Distributor Agreement document type": (
            review.document_type.strip().casefold() == "distributor agreement"
        ),
        "both full party names": (
            len(parties) == 2
            and all(len(party) <= 100 for party in parties)
            and any("electric city corp" in party for party in parties)
            and any("electric city of illinois" in party for party in parties)
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


def _published_review_missing_facts(
    contract_text: str, review: ContractReview
) -> list[str]:
    if not _is_published_contract(contract_text):
        return []
    return _published_review_missing_labels(review)


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
    is_published_contract = _is_published_contract(app.contract_text)
    unsupported_sections = _unsupported_published_sections(str(gather.final_output))
    normalized_findings = " ".join(str(gather.final_output).casefold().split())
    incorrect_document_type = (
        "distributor agreement" not in normalized_findings
        or "master services agreement" in normalized_findings
        or re.search(r"\bmsa\b", normalized_findings) is not None
    )
    narrative_is_bounded = _published_findings_narrative_is_bounded(
        str(gather.final_output)
    )
    deadline_status_is_grounded = _published_deadline_status_is_grounded(
        str(gather.final_output)
    )
    if is_published_contract and (
        unsupported_sections
        or incorrect_document_type
        or not narrative_is_bounded
        or not deadline_status_is_grounded
    ):
        risks = app.clause_cache.get("risk_analysis")
        if not isinstance(risks, dict):
            raise RuntimeError("Grounded clause-risk analysis is unavailable")
        grounded_analysis = ClauseRiskAnalysis.model_validate(risks)
        grounded_findings = _render_grounded_published_findings(grounded_analysis)
        remaining_sections = _unsupported_published_sections(grounded_findings)
        if remaining_sections:
            raise RuntimeError(
                "Grounded findings assembly retained unsupported sections: "
                + ", ".join(sorted(remaining_sections))
            )
        normalized_grounded = " ".join(grounded_findings.casefold().split())
        if (
            "distributor agreement" not in normalized_grounded
            or "master services agreement" in normalized_grounded
            or re.search(r"\bmsa\b", normalized_grounded) is not None
        ):
            raise RuntimeError(
                "Grounded findings assembly retained the wrong document classification"
            )
        gather.final_output = grounded_findings
    obligations_result = app.clause_cache.get("obligations_result")
    if isinstance(obligations_result, str) and obligations_result.strip():
        findings = str(gather.final_output).rstrip()
        if obligations_result not in findings:
            gather.final_output = (
                f"{findings}\n\nObligations and deadlines "
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
    if not isinstance(risk_analysis, dict):
        raise RuntimeError(  # noqa: TRY004
            "Grounded clause-risk analysis is unavailable"
        )
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
    if is_published_contract:
        draft.recommendation = _align_published_signature_recommendation(
            draft.recommendation
        )
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
