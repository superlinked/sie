from __future__ import annotations

import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from rich.console import Console

from insurance_claims.config import FIXTURES_DIR, PACKET_DIR, load_claim, load_config, source_by_slug

console = Console()
FIELD_PREFIX = "CBP303[0].FEMAFormTemplate[0]."


def _money(value: Decimal | float) -> str:
    return f"{Decimal(str(value)):,.2f}"


def line_total(row: list[Any]) -> Decimal:
    return Decimal(str(row[1])) * Decimal(str(row[3]))


def reconciliation(claim: dict[str, Any]) -> dict[str, Decimal]:
    building_attachment = sum((line_total(row) for row in claim["estimate"]["building"]), Decimal())
    contents_attachment = sum((line_total(row) for row in claim["estimate"]["contents"]), Decimal())
    proof = claim["proof_of_loss"]
    claimed_total = Decimal(str(proof["building_net_claimed"])) + Decimal(str(proof["contents_net_claimed"]))
    attachment_total = (
        building_attachment
        - Decimal(str(proof["building_deductible"]))
        + contents_attachment
        - Decimal(str(proof["contents_deductible"]))
    )
    return {
        "building_attachment_acv": building_attachment,
        "contents_attachment_acv": contents_attachment,
        "claimed_total": claimed_total,
        "attachment_total": attachment_total,
        "difference": claimed_total - attachment_total,
    }


def _field(name: str) -> str:
    return f"{FIELD_PREFIX}{name}"


def _fill_proof_of_loss(template: Path, destination: Path, claim: dict[str, Any]) -> None:
    property_data = claim["property"]
    contact = claim["contact"]
    proof = claim["proof_of_loss"]
    values = {
        _field("CheckBox1[0]"): "/1",
        _field("TextField1[0]"): claim["insured_name"],
        _field("TextField2[0]"): claim["policy_number"],
        _field("Table1[0].Row1[0].Cell1[0]"): property_data["street"],
        _field("Table1[0].Row2[0].Cell1[0]"): property_data["city"],
        _field("Table1[0].Row2[0].Cell2[0]"): property_data["state"],
        _field("Table1[0].Row2[0].Cell3[0]"): property_data["zip"],
        _field("TextField3[0]"): claim["date_of_loss"],
        _field("CheckBox4[0]"): "/1",
        _field("TextField5[0]"): claim["mailing_address"],
        _field("TextField7[0]"): property_data["city"],
        _field("DropDownList1[0]"): property_data["state"],
        _field("TextField8[0]"): property_data["zip"],
        _field("TextField9[0]"): "Sample Flood Insurance Co.",
        _field("TextField10[0]"): contact["phone"],
        _field("TextField12[0]"): contact["email"],
        _field("TextField13[0]"): contact["phone"],
        _field("CheckBox5[0]"): "/1",
        _field("CheckBox7[0]"): "/1",
        _field("CheckBox12[0]"): "/1",
        _field("TextField15[0]"): _money(claim["coverage"]["building"]),
        _field("TextField16[0]"): _money(claim["coverage"]["contents"]),
        _field("TextField17[0]"): _money(proof["building_rcv"]),
        _field("TextField18[0]"): _money(proof["contents_rcv"]),
        _field("TextField19[0]"): _money(proof["building_acv"]),
        _field("TextField20[0]"): _money(proof["contents_acv"]),
        _field("TextField21[0]"): _money(proof["building_depreciation"]),
        _field("TextField22[0]"): _money(proof["building_deductible"]),
        _field("TextField23[0]"): _money(proof["contents_deductible"]),
        _field("TextField24[0]"): _money(proof["building_net_claimed"]),
        _field("TextField25[0]"): _money(proof["contents_net_claimed"]),
        _field("CheckBox13[0]"): "/1",
    }
    reader = PdfReader(template)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    writer.update_page_form_field_values(writer.pages[0], values, auto_regenerate=False)
    writer.set_need_appearances_writer(True)
    with destination.open("wb") as output:
        writer.write(output)


def _estimate_rows(rows: list[list[Any]]) -> list[list[str]]:
    result = [["Description", "Qty.", "Unit", "Unit price", "Line total"]]
    for description, quantity, unit, price in rows:
        result.append(
            [
                str(description),
                f"{quantity:g}" if isinstance(quantity, float) else str(quantity),
                str(unit),
                f"${_money(price)}",
                f"${_money(line_total([description, quantity, unit, price]))}",
            ]
        )
    return result


def _create_estimate(destination: Path, claim: dict[str, Any]) -> None:
    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8, leading=10)
    right = ParagraphStyle("right", parent=styles["BodyText"], alignment=TA_RIGHT, fontSize=9)
    document = SimpleDocTemplate(
        str(destination),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        title=f"Repair estimate for {claim['claim_number']}",
    )
    story = [
        Paragraph("SYNTHETIC SAMPLE CLAIM", styles["Heading4"]),
        Paragraph("Bayou Restoration Co. (fictional)", styles["Title"]),
        Paragraph(
            f"Estimate BR-1042 · Claim {claim['claim_number']}<br/>"
            f"{claim['property']['street']}, {claim['property']['city']}, "
            f"{claim['property']['state']} {claim['property']['zip']}",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        Paragraph("Building repair estimate", styles["Heading2"]),
    ]
    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7E7E7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#222222")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("LEADING", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAAAAA")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )
    building_table = Table(
        _estimate_rows(claim["estimate"]["building"]),
        colWidths=[3.05 * inch, 0.45 * inch, 0.65 * inch, 0.8 * inch, 0.85 * inch],
        repeatRows=1,
    )
    building_table.setStyle(table_style)
    story.extend([building_table, Spacer(1, 12), Paragraph("Contents inventory", styles["Heading2"])])
    contents_table = Table(
        _estimate_rows(claim["estimate"]["contents"]),
        colWidths=[3.05 * inch, 0.45 * inch, 0.65 * inch, 0.8 * inch, 0.85 * inch],
        repeatRows=1,
    )
    contents_table.setStyle(table_style)
    totals = reconciliation(claim)
    proof = claim["proof_of_loss"]
    summary = Table(
        [
            ["Building attachment ACV", f"${_money(totals['building_attachment_acv'])}"],
            ["Contents attachment ACV", f"${_money(totals['contents_attachment_acv'])}"],
            ["Less deductibles", f"-${_money(proof['building_deductible'] + proof['contents_deductible'])}"],
            ["Attachment net total", f"${_money(totals['attachment_total'])}"],
        ],
        colWidths=[4.85 * inch, 1.0 * inch],
        hAlign="RIGHT",
    )
    summary.setStyle(
        TableStyle(
            [
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("LINEABOVE", (0, -1), (-1, -1), 0.8, colors.black),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.extend(
        [
            contents_table,
            Spacer(1, 14),
            summary,
            Spacer(1, 14),
            Paragraph(
                "This estimate is an explicitly fictional fixture for software evaluation. "
                "It is not an estimate, offer, policy interpretation, or claim decision.",
                small,
            ),
            Paragraph(f"Prepared for evaluation on claim {claim['claim_number']}.", right),
        ]
    )
    document.build(story)


def prepare_claim() -> Path:
    config = load_config()
    claim = load_claim()
    template = source_by_slug(config, "nfip-proof-of-loss").path
    photo = source_by_slug(config, "flooded-house-interior").path
    if not template.exists() or not photo.exists():
        raise FileNotFoundError("Missing source files. Run `uv run fetch-claim-sources` first.")
    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    proof_path = PACKET_DIR / "filled-proof-of-loss.pdf"
    estimate_path = PACKET_DIR / "repair-estimate-and-inventory.pdf"
    photo_path = PACKET_DIR / "damage-photo.jpg"
    note_path = PACKET_DIR / "claim-note.txt"
    _fill_proof_of_loss(template, proof_path, claim)
    _create_estimate(estimate_path, claim)
    shutil.copyfile(photo, photo_path)
    shutil.copyfile(FIXTURES_DIR / "claim-note.txt", note_path)
    totals = reconciliation(claim)
    manifest = {
        "fictional": True,
        "claim_number": claim["claim_number"],
        "files": {
            "proof_of_loss": str(proof_path.relative_to(PACKET_DIR)),
            "estimate": str(estimate_path.relative_to(PACKET_DIR)),
            "damage_photo": str(photo_path.relative_to(PACKET_DIR)),
            "claim_note": str(note_path.relative_to(PACKET_DIR)),
            "policy": str(source_by_slug(config, "sfip-dwelling-policy").path),
        },
        "expected_evidence": {key: float(value) for key, value in totals.items()},
        "known_issues": ["missing_signature", "amount_mismatch"],
    }
    manifest_path = PACKET_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    console.print(f"Claim packet: {manifest_path}")
    console.print(
        f"Form claims ${_money(totals['claimed_total'])}; attachments support "
        f"${_money(totals['attachment_total'])}; difference ${_money(totals['difference'])}."
    )
    return manifest_path


def main() -> None:
    prepare_claim()


if __name__ == "__main__":
    main()
