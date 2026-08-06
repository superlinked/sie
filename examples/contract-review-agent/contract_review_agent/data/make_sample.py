"""Generate the self-contained sample corpus for the demo.

Produces, under ``data/generated/``:
  * ``acme-msa.md``, ``mutual-nda.md``, ``acme-sow.md`` — contracts to review/index
  * ``acme-msa-signature.png`` — the executed signature page, as a "scan" image
  * ``obligations.db`` — a SQLite database the text-to-SQL tool queries

Everything here is fictional. The MSA is seeded with genuinely risk-bearing
clauses (auto-renewal, uncapped indemnity, vendor-friendly termination) so the
risk-analysis agent has real findings to surface.
"""

from __future__ import annotations

import sqlite3
import textwrap

from PIL import Image, ImageDraw, ImageFont

from contract_review_agent.data.paths import GENERATED_DIR

# Treat this as "today" for the seeded obligation due dates.
TODAY = "2026-06-22"

# Single source of truth for the obligations schema: used both to create the
# table and to prompt the text-to-SQL model (the inline comments help it).
SCHEMA_DDL = """\
CREATE TABLE obligations (
    id            INTEGER PRIMARY KEY,
    counterparty  TEXT NOT NULL,   -- legal name of the other party
    contract_type TEXT NOT NULL,   -- one of: MSA, NDA, SOW
    obligation    TEXT NOT NULL,   -- description of what is owed/required
    owner         TEXT NOT NULL,   -- who must act: 'us' or 'counterparty'
    due_date      TEXT NOT NULL,   -- ISO-8601 date (YYYY-MM-DD)
    amount_usd    REAL,            -- dollar amount if a payment, else NULL
    status        TEXT NOT NULL    -- one of: open, done, overdue
)"""

ACME_MSA = """\
# Master Services Agreement

This Master Services Agreement (the "Agreement") is entered into between
Northwind Analytics, Inc. ("Customer") and Acme Cloud Services, LLC ("Provider").

## 1. Definitions
"Services" means the cloud data-processing services described in one or more
Order Forms. "Order Form" means an ordering document executed by both parties
that references this Agreement.

## 2. Term and Renewal
The initial term of this Agreement is twelve (12) months from the Effective
Date. Thereafter, this Agreement automatically renews for successive twelve (12)
month terms unless Customer provides written notice of non-renewal at least
ninety (90) days before the end of the then-current term. Fees for each renewal
term may increase by up to ten percent (10%) over the prior term.

## 3. Fees and Payment
Customer shall pay all fees within thirty (30) days of the invoice date.
Late amounts accrue interest at 1.5% per month. All fees are non-refundable,
including fees for partially used subscription periods.

## 4. Termination
Provider may terminate this Agreement or any Order Form for convenience upon
thirty (30) days written notice. Customer may terminate only for Provider's
uncured material breach, after a sixty (60) day cure period. Upon termination,
Customer shall pay all fees due through the end of the then-current term.

## 5. Limitation of Liability
EXCEPT FOR CUSTOMER'S PAYMENT OBLIGATIONS AND CUSTOMER'S INDEMNIFICATION
OBLIGATIONS, EACH PARTY'S TOTAL AGGREGATE LIABILITY SHALL NOT EXCEED THE FEES
PAID BY CUSTOMER IN THE THREE (3) MONTHS PRECEDING THE CLAIM. PROVIDER SHALL NOT
BE LIABLE FOR ANY INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES.

## 6. Indemnification
Customer shall defend, indemnify, and hold harmless Provider from any and all
claims arising out of Customer's use of the Services, without limitation. This
indemnity is uncapped and survives termination of this Agreement.

## 7. Confidentiality
Each party shall protect the other's Confidential Information using at least a
reasonable standard of care, and shall not disclose it except to personnel with
a need to know who are bound by confidentiality obligations.

## 8. Data Protection
Provider shall process Customer personal data only on Customer's documented
instructions and shall maintain appropriate technical and organizational
security measures. Provider may engage sub-processors, provided Provider remains
liable for their performance.

## 9. Service Levels
Provider targets 99.9% monthly availability. If availability falls below the
target, Customer's sole and exclusive remedy is a service credit equal to 5% of
the monthly fee per 0.1% below target, capped at 25% of the monthly fee.

## 10. Governing Law
This Agreement is governed by the laws of the State of Delaware, without regard
to its conflict-of-laws principles. The parties consent to the exclusive
jurisdiction of the state and federal courts located in Wilmington, Delaware.

## 11. Assignment
Customer may not assign this Agreement without Provider's prior written consent.
Provider may assign this Agreement freely, including to an acquirer of all or
substantially all of its assets.

## 12. Entire Agreement
This Agreement, together with all Order Forms, constitutes the entire agreement
between the parties and supersedes all prior agreements on its subject matter.
"""

MUTUAL_NDA = """\
# Mutual Non-Disclosure Agreement

This Mutual Non-Disclosure Agreement is entered into between Northwind
Analytics, Inc. and Globex Corporation (each a "Party").

## 1. Purpose
The Parties wish to explore a potential business relationship and may disclose
confidential information to one another for that purpose.

## 2. Confidential Information
"Confidential Information" means any non-public information disclosed by a Party,
whether orally, in writing, or by inspection of tangible objects, that is
designated confidential or that reasonably should be understood to be
confidential.

## 3. Obligations
The receiving Party shall use Confidential Information solely for the Purpose,
shall protect it with the same degree of care it uses for its own confidential
information (but no less than reasonable care), and shall not disclose it to
third parties without the disclosing Party's prior written consent.

## 4. Term
This Agreement remains in effect for two (2) years from the Effective Date.
Confidentiality obligations survive for three (3) years after disclosure.

## 5. Return of Materials
Upon written request, the receiving Party shall promptly return or destroy all
Confidential Information in its possession.

## 6. Governing Law
This Agreement is governed by the laws of the State of New York.
"""

ACME_SOW = """\
# Statement of Work No. 1

This Statement of Work ("SOW") is issued under and incorporates the Master
Services Agreement between Northwind Analytics, Inc. and Acme Cloud Services,
LLC.

## 1. Scope
Provider will deliver a managed data-ingestion pipeline, including connectors,
transformation jobs, and a monitoring dashboard, as further described in
Exhibit A.

## 2. Deliverables and Milestones
Milestone 1 (design) is due 30 days after the SOW Effective Date. Milestone 2
(implementation) is due 90 days after. Milestone 3 (acceptance) is due 120 days
after.

## 3. Fees
Fees for this SOW total USD 240,000, invoiced 25% per milestone and 25% on final
acceptance. Travel expenses are billed at cost with prior written approval.

## 4. Acceptance
Customer has fifteen (15) business days after delivery of each milestone to
accept or reject deliverables in writing. Deliverables are deemed accepted if
Customer does not respond within that period.

## 5. Personnel
Provider shall assign a named technical lead for the duration of this SOW and
shall not reassign that person without Customer's consent, not to be
unreasonably withheld.

## 6. Governing Law
This SOW is governed by the Master Services Agreement, including its governing
law provision.
"""

CONTRACTS = {
    "acme-msa": ACME_MSA,
    "mutual-nda": MUTUAL_NDA,
    "acme-sow": ACME_SOW,
}

# The executed signature page — rendered to an image so the OCR and vision
# models have a real "scan" to read (these executed details are intentionally
# NOT in the template body above; the agent must read them off the scan).
SIGNATURE_PAGE = """\
EXECUTION PAGE

MASTER SERVICES AGREEMENT

Effective Date: March 1, 2026

IN WITNESS WHEREOF, the parties have executed this Agreement as of the
Effective Date.

CUSTOMER:                          PROVIDER:
Northwind Analytics, Inc.          Acme Cloud Services, LLC

By: /s/ Dana Whitfield             By: /s/ Marcus Reyes
Name: Dana Whitfield               Name: Marcus Reyes
Title: Chief Operating Officer     Title: VP, Customer Success
Date: March 1, 2026                Date: March 1, 2026

Governing Law: State of Delaware
Notices to Customer: legal@northwind-analytics.example
"""

# (counterparty, contract_type, obligation, owner, due_date, amount_usd, status)
OBLIGATIONS = [
    ("Acme Cloud Services, LLC", "MSA", "Send non-renewal notice (90 days before term end) to avoid auto-renewal", "us", "2026-11-30", None, "open"),
    ("Acme Cloud Services, LLC", "MSA", "Annual subscription fee (renewal term)", "us", "2026-03-01", 180000.0, "done"),
    ("Acme Cloud Services, LLC", "SOW", "Milestone 1 (design) payment", "us", "2026-04-15", 60000.0, "done"),
    ("Acme Cloud Services, LLC", "SOW", "Milestone 2 (implementation) payment", "us", "2026-07-14", 60000.0, "open"),
    ("Acme Cloud Services, LLC", "SOW", "Milestone 3 (acceptance) payment", "us", "2026-08-13", 60000.0, "open"),
    ("Acme Cloud Services, LLC", "MSA", "Quarterly security attestation delivery", "counterparty", "2026-06-30", None, "open"),
    ("Globex Corporation", "NDA", "Return or destroy confidential materials on request", "us", "2026-05-10", None, "overdue"),
    ("Globex Corporation", "NDA", "Confidentiality survival period ends", "us", "2027-09-01", None, "open"),
    ("Initech LLC", "MSA", "Send non-renewal notice (60 days before term end)", "us", "2026-07-02", None, "open"),
    ("Initech LLC", "MSA", "Annual subscription fee", "us", "2026-09-01", 95000.0, "open"),
    ("Umbrella Health, Inc.", "MSA", "Data processing audit response", "counterparty", "2026-06-18", None, "overdue"),
    ("Umbrella Health, Inc.", "MSA", "Annual subscription fee", "us", "2026-12-01", 220000.0, "open"),
]


def _load_font(size: int) -> ImageFont.ImageFont:
    """A scalable font with no external file dependency (Pillow >= 10.1)."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow without sized default
        return ImageFont.load_default()


def render_signature_page(out_path) -> None:
    width, height, margin = 1000, 1300, 70
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(22)
    y = margin
    for raw_line in SIGNATURE_PAGE.splitlines():
        # Preserve the two-column layout by not wrapping; just render each line.
        draw.text((margin, y), raw_line, fill="black", font=font)
        y += 34
    img.save(out_path)


def build_obligations_db(out_path) -> None:
    out_path.unlink(missing_ok=True)
    conn = sqlite3.connect(out_path)
    try:
        conn.execute(SCHEMA_DDL)
        conn.executemany(
            "INSERT INTO obligations "
            "(counterparty, contract_type, obligation, owner, due_date, amount_usd, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            OBLIGATIONS,
        )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    for name, body in CONTRACTS.items():
        (GENERATED_DIR / f"{name}.md").write_text(body)

    sig_path = GENERATED_DIR / "acme-msa-signature.png"
    render_signature_page(sig_path)

    db_path = GENERATED_DIR / "obligations.db"
    build_obligations_db(db_path)

    print(
        textwrap.dedent(
            f"""\
            Sample corpus written to {GENERATED_DIR}:
              contracts : {", ".join(f"{n}.md" for n in CONTRACTS)}
              scan      : {sig_path.name}
              database  : {db_path.name} ({len(OBLIGATIONS)} obligations)
            """
        )
    )


if __name__ == "__main__":
    main()
