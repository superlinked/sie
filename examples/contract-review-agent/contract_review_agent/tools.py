"""The agent's tools — each one a different model from the SIE catalog.

Every tool takes the shared :class:`AppContext` (injected by the Agents SDK via
``RunContextWrapper``, never shown to the model) and records to the ledger the
model it used, the latency, how much data it sent, and the throughput — so a
normal end-to-end run doubles as per-model observability.
"""

from __future__ import annotations

import base64
import re
import sqlite3
import time
from pathlib import Path
from typing import Literal

import numpy as np
from agents import RunContextWrapper, Runner, function_tool
from pydantic import BaseModel, Field
from sie_sdk import Item

from .data.make_sample import SCHEMA_DDL, TODAY
from .runtime import AppContext, GenResult, instruct_once, prompt_once, record_api_call

COMMERCIAL_FACTS_QUERY = (
    "governing law, exclusivity, term, letter of credit, purchase minimum, "
    "and reporting obligations"
)


class ClauseRiskFinding(BaseModel):
    clause: str = Field(pattern=r"^Section \d+(?:\.\d+)+$")
    issue: str = Field(
        min_length=1,
        max_length=400,
        description=(
            "Current source-grounded clause mechanics and resulting risk; do not "
            "describe a proposed redline as existing contract language"
        ),
    )
    severity: Literal["low", "medium", "high"]
    suggested_redline: str = Field(
        min_length=1,
        max_length=400,
        description="A proposed change, kept distinct from current clause mechanics",
    )


class ClauseRiskAnalysis(BaseModel):
    risks: list[ClauseRiskFinding] = Field(min_length=1, max_length=8)


def _tok(n: int | None) -> str:
    return f"{n:,} tok" if n else "—"


def _tps(res: GenResult) -> str:
    return f"{res.tokens_per_s:.0f} tok/s" if res.tokens_per_s else "—"


def _validated_clause_risk_output(value: object, source: str) -> str:
    output = str(value).strip()
    if len(output) < 256:
        raise RuntimeError("Clause-risk analysis returned an incomplete response")
    cited_sections = set(re.findall(r"\bSection\s+(\d+(?:\.\d+)+)\b", output))
    source_sections = set(re.findall(r"\b(\d+(?:\.\d+)+)\b", source))
    if not cited_sections or not cited_sections <= source_sections:
        unexpected = sorted(cited_sections - source_sections)
        raise RuntimeError(
            "Clause-risk analysis must cite only exact sections from retrieved "
            f"clauses; unexpected sections: {unexpected}"
        )
    return output


def _format_clause_risk_analysis(analysis: ClauseRiskAnalysis) -> str:
    return "\n\n".join(
        f"{risk.clause} | severity: {risk.severity}\n"
        f"Issue: {risk.issue}\nSuggested redline: {risk.suggested_redline}"
        for risk in analysis.risks
    )


def _split_clauses(text: str, target: int = 900) -> list[str]:
    """Chunk a contract into ~``target``-char passages for retrieval.

    Works on real contract text (plain, no markdown) as well as the synthetic
    markdown body: split on blank-line paragraphs (falling back to lines), greedily
    pack to ``target`` chars, then hard-split anything still oversized.
    """
    text = text.replace("\r\n", "\n")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) < 3:  # e.g. single-spaced contract text
        blocks = [ln.strip() for ln in text.split("\n") if ln.strip()]

    chunks: list[str] = []
    current = ""
    for block in blocks:
        if current and len(current) + len(block) + 2 > target:
            chunks.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        chunks.append(current)

    out: list[str] = []
    for chunk in chunks:
        while len(chunk) > target * 1.6:
            out.append(chunk[:target])
            chunk = chunk[target:]
        out.append(chunk)
    return out


async def _clause_index(
    app: AppContext, embed_model: str
) -> tuple[list[str], np.ndarray]:
    """Embed the contract's clauses once and cache (clauses, matrix)."""
    if "matrix" in app.clause_cache:
        return app.clause_cache["clauses"], app.clause_cache["matrix"]
    clauses = _split_clauses(app.contract_text)
    t0 = time.monotonic()
    results = await app.sie.encode(
        embed_model,
        [Item(id=str(i), text=c) for i, c in enumerate(clauses)],
        output_types=["dense"],
        wait_for_capacity=True,
        provision_timeout_s=app.provision_timeout_s,
    )
    record_api_call(
        app,
        "encode",
        embed_model,
        results,
        stage="search_clauses:index",
    )
    dt = time.monotonic() - t0
    matrix = np.vstack([np.asarray(r["dense"], dtype=np.float32) for r in results])
    app.ledger.record(
        "Embed clauses (index)",
        embed_model,
        "encode",
        latency_s=dt,
        sent=f"{len(clauses)} clauses",
        got=f"{matrix.shape[1]}-dim",
        throughput=f"{len(clauses) / dt:.1f} items/s" if dt > 0 else "—",
    )
    app.clause_cache["clauses"] = clauses
    app.clause_cache["matrix"] = matrix
    return clauses, matrix


# ──────────────────────────────────────────────────────────────────────────
# Generative-LLM tools
# ──────────────────────────────────────────────────────────────────────────
@function_tool(failure_error_function=None)
async def classify_document(ctx: RunContextWrapper[AppContext]) -> str:
    """Classify the contract under review as MSA, NDA, SOW, Distributor
    Agreement, or Other, with a one-line reason. A fast first-pass triage over
    the loaded contract."""
    app = ctx.context
    model = app.cfg["models"]["triage"]
    res = await instruct_once(
        app,
        model,
        [
            {
                "role": "system",
                "content": (
                    "You label legal documents. Reply with exactly one type "
                    "(MSA, NDA, SOW, Distributor Agreement, or Other) followed by "
                    "a short reason. A document titled Distributor Agreement is a "
                    "Distributor Agreement, not an MSA."
                ),
            },
            {"role": "user", "content": app.contract_text[:1500]},
        ],
        stage="classify_document",
        max_tokens=60,
    )
    app.ledger.record(
        "Triage: classify document",
        model,
        "generate",
        warmup_s=res.provision_s,
        latency_s=res.gen_s,
        sent=_tok(res.prompt_tokens),
        got=_tok(res.completion_tokens),
        throughput=_tps(res),
    )
    result = res.text.strip()
    app.clause_cache["classification_result"] = result
    return result


@function_tool(failure_error_function=None)
async def read_signature_page(ctx: RunContextWrapper[AppContext], question: str) -> str:
    """Ask a vision model a question about the scanned signature-page image
    (e.g. 'Are both parties' signatures present and dated?'). Use this for
    visual checks that OCR text alone cannot answer."""
    app = ctx.context
    model = app.cfg["models"]["vision"]
    data = base64.b64encode(Path(app.scan_path).read_bytes()).decode()
    res = await instruct_once(
        app,
        model,
        [
            {
                "role": "system",
                "content": (
                    "You are a meticulous contracts paralegal. Answer only from what "
                    "is visible in the image. Describe only the visible page or image; "
                    "never infer where the full contract ends."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{data}"},
                    },
                ],
            },
        ],
        stage="read_signature_page",
        max_tokens=220,
    )
    app.ledger.record(
        "Read signature page (vision)",
        model,
        "generate + image",
        warmup_s=res.provision_s,
        latency_s=res.gen_s,
        sent=f"{Path(app.scan_path).stat().st_size // 1024} KB img + {_tok(res.prompt_tokens)}",
        got=_tok(res.completion_tokens),
        throughput=_tps(res),
    )
    return res.text.strip()


@function_tool(failure_error_function=None)
async def analyze_clause_risks(ctx: RunContextWrapper[AppContext]) -> str:
    """Analyze the clauses returned by the required retrieval steps."""
    app = ctx.context
    search_results = app.clause_cache.get("search_results")
    if not isinstance(search_results, dict) or not search_results:
        raise RuntimeError("Clause-risk analysis requires saved search results")
    clause_topics: dict[str, list[str]] = {}
    for query, clauses in search_results.items():
        if (
            not isinstance(query, str)
            or not query.strip()
            or not isinstance(clauses, list)
            or any(
                not isinstance(clause, str) or not clause.strip() for clause in clauses
            )
        ):
            raise RuntimeError(
                "Clause-risk search results must map string queries to lists of strings"
            )
        for clause in clauses:
            clause_topics.setdefault(clause, []).append(query)
    if not clause_topics:
        raise RuntimeError("Clause-risk analysis found no saved clauses")
    clauses = "\n\n===\n\n".join(
        f"Topics: {', '.join(topics)}\n\n{clause}"
        for clause, topics in clause_topics.items()
    )
    model = app.cfg["models"]["reasoning"]
    t0 = time.monotonic()
    result = await Runner.run(
        app.reasoning_agent,
        "Analyze the following contract clauses for risk to the Customer. For each "
        "risk, give: the clause, the issue, a severity (low/medium/high), and a "
        f"one-line suggested redline.\n\n{clauses}",
        context=app,
    )
    dt = time.monotonic() - t0
    usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    out_tok = getattr(usage, "output_tokens", None)
    app.ledger.record(
        "Clause-risk analysis (specialist sub-agent)",
        model,
        "generate",
        latency_s=dt,
        sent=_tok(getattr(usage, "input_tokens", None)),
        got=_tok(out_tok),
        throughput=f"{out_tok / dt:.0f} tok/s" if out_tok and dt > 0 else "—",
    )
    analysis = result.final_output_as(ClauseRiskAnalysis)
    app.clause_cache["risk_analysis"] = analysis.model_dump(mode="json")
    return _validated_clause_risk_output(
        _format_clause_risk_analysis(analysis), clauses
    )


# ──────────────────────────────────────────────────────────────────────────
# Retrieval / extraction tools (encode · score · extract)
# ──────────────────────────────────────────────────────────────────────────
@function_tool(failure_error_function=None)
async def ocr_signature_page(ctx: RunContextWrapper[AppContext]) -> str:
    """OCR the executed signature page (a scanned image) into markdown text.
    Use this to recover who signed, their titles, and the execution date —
    details that exist only on the scan, not in the contract body."""
    app = ctx.context
    model = app.cfg["models"]["ocr"]
    t0 = time.monotonic()
    res = await app.sie.extract(
        model,
        Item(images=[app.scan_path]),
        wait_for_capacity=True,
        provision_timeout_s=app.provision_timeout_s,
    )
    record_api_call(app, "extract", model, res, stage="ocr_signature_page")
    dt = time.monotonic() - t0
    entities = res.get("entities") or []
    markdown = entities[0]["text"] if entities else "(no text recognized)"
    app.ledger.record(
        "OCR signature page → markdown",
        model,
        "extract",
        latency_s=dt,
        sent=f"{Path(app.scan_path).stat().st_size // 1024} KB img",
        got=f"{len(markdown):,} chars md",
        throughput=f"{len(markdown) / dt:.0f} chars/s" if dt > 0 else "—",
    )
    return markdown


@function_tool(failure_error_function=None)
async def extract_entities(ctx: RunContextWrapper[AppContext]) -> str:
    """Extract structured entities (parties, dates, monetary amounts, governing
    law, notice periods) from the loaded contract using zero-shot NER."""
    app = ctx.context
    model = app.cfg["models"]["entities"]
    labels = [
        "party",
        "effective date",
        "renewal date",
        "termination notice period",
        "monetary amount",
        "governing law",
        "term length",
    ]
    payload = app.contract_text[:6000]
    t0 = time.monotonic()
    res = await app.sie.extract(
        model,
        Item(text=payload),
        labels=labels,
        wait_for_capacity=True,
        provision_timeout_s=app.provision_timeout_s,
    )
    record_api_call(app, "extract", model, res, stage="extract_entities")
    dt = time.monotonic() - t0
    entities = res.get("entities") or []
    app.ledger.record(
        "Extract entities (zero-shot NER)",
        model,
        "extract",
        latency_s=dt,
        sent=f"{len(payload):,} chars",
        got=f"{len(entities)} entities",
        throughput=f"{len(entities) / dt:.1f} ent/s" if dt > 0 else "—",
    )
    lines = [
        f"- {e['label']}: {e['text']} (score {e.get('score', 0):.2f})" for e in entities
    ]
    return "\n".join(lines) if lines else "(no entities found)"


@function_tool(failure_error_function=None)
async def search_clauses(ctx: RunContextWrapper[AppContext], query: str) -> str:
    """Find the clauses most relevant to a topic (e.g. 'renewal mechanics',
    'limitation of liability', 'indemnification'). Dense-embedding retrieval
    followed by a cross-encoder rerank; returns the top clauses verbatim."""
    app = ctx.context
    embed_model = app.cfg["models"]["embed"]
    rerank_model = app.cfg["models"]["rerank"]
    k_cand = int(app.cfg["search"]["top_k_candidates"])
    k_res = int(app.cfg["search"]["top_k_results"])

    clauses, matrix = await _clause_index(app, embed_model)
    q = await app.sie.encode(
        embed_model,
        Item(text=query),
        output_types=["dense"],
        is_query=True,
        wait_for_capacity=True,
        provision_timeout_s=app.provision_timeout_s,
    )
    record_api_call(
        app,
        "encode",
        embed_model,
        q,
        stage="search_clauses:encode",
    )
    qv = np.asarray(q["dense"], dtype=np.float32)
    denom = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(qv) + 1e-9) + 1e-9
    sims = (matrix @ qv) / denom
    candidate_idx = np.argsort(-sims)[: min(k_cand, len(clauses))]
    candidates = [clauses[i] for i in candidate_idx]

    t0 = time.monotonic()
    scored = await app.sie.score(
        rerank_model,
        Item(text=query),
        [Item(id=str(i), text=c) for i, c in enumerate(candidates)],
        wait_for_capacity=True,
        provision_timeout_s=app.provision_timeout_s,
    )
    record_api_call(
        app,
        "score",
        rerank_model,
        scored,
        stage="search_clauses:score",
    )
    dt = time.monotonic() - t0
    app.ledger.record(
        "Rerank candidate clauses",
        rerank_model,
        "score",
        latency_s=dt,
        sent=f"{len(candidates)} docs",
        got=f"top {k_res}",
        throughput=f"{len(candidates) / dt:.1f} docs/s" if dt > 0 else "—",
    )
    ranked = sorted(scored["scores"], key=lambda s: s["rank"])[:k_res]
    top = [candidates[int(s["item_id"])] for s in ranked]
    search_results = app.clause_cache.setdefault("search_results", {})
    if not isinstance(search_results, dict):
        raise TypeError("Clause search cache has an invalid shape")
    search_results[query] = top
    return "\n\n---\n\n".join(top) if top else "(no relevant clauses found)"


# ──────────────────────────────────────────────────────────────────────────
# Text-to-SQL tool (role-labelled instruction or raw specialist prompt)
# ──────────────────────────────────────────────────────────────────────────
_SQLCODER_PROMPT = """### Task
Generate a SQLite SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- Use SQLite syntax only. Today's date is {today}.
- Dates are ISO-8601 text; use SQLite date functions (e.g. date('now'), date(due_date)).

### Database Schema
The query will run on a database with this schema:
{schema}

### Answer
Given the database schema, here is the SQLite query that answers [QUESTION]{question}[/QUESTION]:
"""

# Chat-mode SQL prompt for chat/instruct models (e.g. Qwen3.5-4B).
_SQL_CHAT_SYSTEM = (
    "You are a text-to-SQL engine for SQLite. Given the schema and a question, "
    "reply with ONE SQLite SELECT statement and nothing else — no prose, no "
    "explanation, no markdown fences. Today's date is {today}. Dates are stored "
    "as ISO-8601 text; use SQLite date functions (e.g. date('now'), date(due_date))."
)

_SQL_CHAT_USER = """Database schema:
{schema}

Question: {question}

SQLite query:"""


def _clean_sql(raw: str) -> str:
    sql = raw.strip()
    for fence in ("```sql", "```sqlite", "```"):
        if sql.startswith(fence):
            sql = sql[len(fence) :].strip()
    sql = sql.split("```")[0]
    if "[SQL]" in sql:
        sql = sql.split("[SQL]", 1)[1]
    return sql.strip().rstrip(";").strip()


def _run_select(
    db_path: str, sql: str, params: dict[str, str] | None = None
) -> tuple[list[str], list[tuple]]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(sql, params or {})
        cols = [d[0] for d in cur.description] if cur.description else []
        return cols, cur.fetchmany(50)
    finally:
        conn.close()


def _open_obligation_count(db_path: str, counterparty: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM obligations "
            "WHERE counterparty = ? AND status = 'open'",
            (counterparty,),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


@function_tool(failure_error_function=None)
async def query_obligations_db(
    ctx: RunContextWrapper[AppContext], question: str
) -> str:
    """Answer a question about tracked contract obligations and deadlines by
    generating SQL (a text-to-SQL specialist model) and running it against the
    obligations database. Good for 'which obligations are due soon?' or 'total
    outstanding payments by counterparty'."""
    app = ctx.context
    model = app.cfg["models"]["sql"]
    mode = (app.cfg.get("sql") or {}).get("mode", "instruct")
    scoped_question = question
    if app.obligation_counterparty:
        scoped_question += (
            "\nReturn every row whose counterparty equals the named SQLite parameter "
            ":counterparty and whose status exactly equals 'open'. Use :counterparty "
            "in the WHERE clause; do not copy or invent a literal counterparty value. "
            "Include counterparty, status, obligation, due_date, and amount_usd in "
            "SELECT. Order by due_date. Do not use LIMIT or a date window."
        )
    if mode == "prompt":
        prompt = _SQLCODER_PROMPT.format(
            question=scoped_question,
            schema=SCHEMA_DDL,
            today=TODAY,
        )
        res = await prompt_once(
            app,
            model,
            prompt,
            stage="query_obligations_db",
            max_tokens=256,
            stop=[";", "```", "\n\n\n"],
        )
    elif mode == "instruct":
        res = await instruct_once(
            app,
            model,
            [
                {
                    "role": "system",
                    "content": _SQL_CHAT_SYSTEM.format(today=TODAY),
                },
                {
                    "role": "user",
                    "content": _SQL_CHAT_USER.format(
                        schema=SCHEMA_DDL,
                        question=scoped_question,
                    ),
                },
            ],
            stage="query_obligations_db",
            max_tokens=256,
        )
    else:
        raise ValueError("sql.mode must be 'instruct' or 'prompt'")
    app.ledger.record(
        "Text-to-SQL",
        model,
        "generate",
        warmup_s=res.provision_s,
        latency_s=res.gen_s,
        sent=_tok(res.prompt_tokens),
        got=_tok(res.completion_tokens),
        throughput=_tps(res),
    )

    sql = _clean_sql(res.text)
    if not sql.lower().startswith("select"):
        return f"Generated query was not a SELECT, so it was not run:\n{sql}"
    sql_params: dict[str, str] | None = None
    if app.obligation_counterparty:
        if ":counterparty" not in sql:
            return (
                "Generated query omitted the required bound :counterparty parameter."
                f"\nSQL: {sql}"
            )
        sql_params = {"counterparty": app.obligation_counterparty}
    try:
        cols, rows = _run_select(app.db_path, sql, sql_params)
    except sqlite3.Error as exc:
        return f"SQL error: {exc}\nQuery was:\n{sql}"
    if app.obligation_counterparty:
        required_scope_columns = {"counterparty", "status"}
        missing_scope_columns = required_scope_columns - set(cols)
        if missing_scope_columns:
            return (
                "Generated query omitted columns required for exact contract "
                f"scoping: {', '.join(sorted(missing_scope_columns))}.\nSQL: {sql}"
            )
        counterparty_index = cols.index("counterparty")
        status_index = cols.index("status")
        rows = [
            row
            for row in rows
            if row[counterparty_index] == app.obligation_counterparty
            and row[status_index] == "open"
        ]
        expected_rows = _open_obligation_count(
            app.db_path, app.obligation_counterparty
        )
        if len(rows) != expected_rows:
            return (
                "Generated query returned incomplete contract scope: expected "
                f"{expected_rows} open rows, got {len(rows)}.\nSQL: {sql}"
            )
    if not rows:
        return f"Query ran but returned no rows.\nSQL: {sql}"
    header = " | ".join(cols)
    body = "\n".join(
        " | ".join("" if v is None else str(v) for v in row) for row in rows
    )
    result = f"SQL: {sql}\n\n{header}\n{body}"
    app.clause_cache["obligations_result"] = result
    return result


ALL_TOOLS = [
    classify_document,
    ocr_signature_page,
    extract_entities,
    search_clauses,
    read_signature_page,
    query_obligations_db,
    analyze_clause_risks,
]
