"""Fetch real contracts from CUAD to build the corpus the agent reviews.

CUAD (Contract Understanding Atticus Dataset) is 510 real commercial contracts
filed with the SEC, released by The Atticus Project under CC BY 4.0 — the
dataset built specifically for contract review. We download the dataset's small
(~18 MB) archive once, parse the SQuAD-format JSON, write a curated handful of
full contracts to ``data/generated/cuad/``, render one page to an image for the
OCR/vision step, and seed an obligations database that references them.

    Dan Hendrycks, Collin Burns, Anya Chen, Spencer Ball. "CUAD: An
    Expert-Annotated NLP Dataset for Legal Contract Review." arXiv:2103.06268.
    Dataset: https://www.atticusprojectai.org/cuad — CC BY 4.0.

Run: ``uv run fetch-contracts`` (offline alternative: ``uv run make-sample``).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path

import httpx

from .make_sample import SCHEMA_DDL
from .paths import CUAD_DIR, DB_PATH, GENERATED_DIR, MANIFEST_PATH
from .render import render_text_page

CUAD_ZIP_URL = "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data.zip"
CITATION = (
    "CUAD (Contract Understanding Atticus Dataset), The Atticus Project, "
    "CC BY 4.0. Hendrycks et al., arXiv:2103.06268."
)

_TYPE_KEYWORDS = [
    ("LICENSE", "License"),
    ("RESELLER", "Reseller"),
    ("DISTRIBUTOR", "Distribution"),
    ("HOSTING", "Hosting"),
    ("MAINTENANCE", "Maintenance"),
    ("SUPPLY", "Supply"),
    ("SERVICE", "Services"),
    ("CONSULTING", "Consulting"),
    ("DEVELOPMENT", "Development"),
    ("MARKETING", "Marketing"),
]


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return (s[:60] or "contract").rstrip("-")


def _infer_type(title: str) -> str:
    upper = title.upper()
    for needle, label in _TYPE_KEYWORDS:
        if needle in upper:
            return label
    return "Agreement"


def _short_name(title: str) -> str:
    # CUAD titles are long EDGAR filenames; keep a readable counterparty-ish label.
    return re.sub(r"[_\-]+", " ", title).strip()[:48]


def _download_archive() -> Path:
    """Download CUAD's data.zip once, cached under data/generated/.cache/."""
    dest = GENERATED_DIR / ".cache" / "cuad-data.zip"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CUAD archive (~18 MB) from {CUAD_ZIP_URL} ...", file=sys.stderr)
    with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
        try:
            with httpx.stream("GET", CUAD_ZIP_URL, follow_redirects=True, timeout=180) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes():
                    tmp.write(chunk)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.rename(dest)
    return dest


def fetch(n: int, match: str | None) -> list[dict]:
    """Return up to ``n`` full contracts from CUAD as {title, text, ...} dicts."""
    archive = _download_archive()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        json_name = next((m for m in names if m.lower().endswith("cuadv1.json")), None) or next(
            m for m in names if m.lower().endswith(".json")
        )
        squad = json.loads(z.read(json_name))

    contracts: list[dict] = []
    for entry in squad["data"]:
        title = (entry.get("title") or "").strip()
        paragraphs = entry.get("paragraphs") or []
        if not title or not paragraphs:
            continue
        if match and match.lower() not in title.lower():
            continue
        text = (paragraphs[0].get("context") or "").strip()
        if not text:
            continue
        contracts.append(
            {
                "title": title,
                "slug": _slug(title),
                "type": _infer_type(title),
                "counterparty": _short_name(title),
                "text": text,
                "char_len": len(text),
            }
        )
        if len(contracts) >= n:
            break
    return contracts


# Synthetic obligations attached to the real contracts, so the text-to-SQL tool
# has deadlines/payments to query about the contracts actually in the corpus.
_OBLIGATION_PLANS = [
    ("Send renewal / non-renewal notice before term end", "us", "2026-09-15", None, "open"),
    ("Annual subscription / license fee", "us", "2026-07-01", 120000.0, "open"),
    ("Quarterly compliance attestation", "counterparty", "2026-06-30", None, "open"),
    ("Prior-period true-up payment", "us", "2026-04-30", 45000.0, "done"),
    ("Audit / records inspection response", "counterparty", "2026-06-10", None, "overdue"),
]


def _signature_page_text(contract_text: str) -> str:
    normalized = contract_text.casefold()
    starts = [
        normalized.rfind(marker)
        for marker in ("in witness whereof", "signature page", "signatures")
    ]
    start = max(starts)
    if start >= 0:
        return contract_text[start:]
    wrapped_lines: list[str] = []
    for raw_line in contract_text.splitlines():
        wrapped_lines.extend(textwrap.wrap(raw_line, width=92) or [""])
    return "\n".join(wrapped_lines[-46:])


def build_obligations(contracts: list[dict]) -> int:
    DB_PATH.unlink(missing_ok=True)
    rows = []
    for i, c in enumerate(contracts):
        for j in range(3):
            obligation, owner, due, amount, status = _OBLIGATION_PLANS[(i + j) % len(_OBLIGATION_PLANS)]
            rows.append((c["counterparty"], c["type"], obligation, owner, due, amount, status))
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(SCHEMA_DDL)
        conn.executemany(
            "INSERT INTO obligations "
            "(counterparty, contract_type, obligation, owner, due_date, amount_usd, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real contracts from CUAD (CC BY 4.0).")
    parser.add_argument("--contracts", type=int, default=4, help="how many contracts to pull")
    parser.add_argument("--match", default=None, help="only titles containing this substring")
    args = parser.parse_args()

    CUAD_DIR.mkdir(parents=True, exist_ok=True)
    contracts = fetch(args.contracts, args.match)
    if not contracts:
        raise SystemExit("No contracts matched. Try a different --match or run `uv run make-sample`.")

    for c in contracts:
        (CUAD_DIR / f"{c['slug']}.txt").write_text(c["text"])

    primary = contracts[0]
    scan_path = CUAD_DIR / f"{primary['slug']}-page.png"
    render_text_page(
        _signature_page_text(primary["text"]),
        scan_path,
        title=f"{primary['title']} — signature page",
    )

    n_obligations = build_obligations(contracts)

    manifest = {
        "source": "CUAD v1",
        "license": "CC BY 4.0",
        "citation": CITATION,
        "primary": primary["slug"],
        "scan_path": str(scan_path.relative_to(GENERATED_DIR)),
        "db_path": str(DB_PATH.relative_to(GENERATED_DIR)),
        "contracts": [
            {k: c[k] for k in ("title", "slug", "type", "counterparty", "char_len")} for c in contracts
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {len(contracts)} contracts to {CUAD_DIR}:")
    for c in contracts:
        flag = "  (primary)" if c["slug"] == primary["slug"] else ""
        print(f"  {c['type']:12s} {c['slug']}.txt  ({c['char_len']:,} chars){flag}")
    print(f"Rendered scan page : {scan_path.name}")
    print(f"Obligations DB     : {DB_PATH.name} ({n_obligations} rows)")
    print(f"Source: {CITATION}")


if __name__ == "__main__":
    main()
