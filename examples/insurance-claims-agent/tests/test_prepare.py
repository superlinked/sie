from __future__ import annotations

from decimal import Decimal

from insurance_claims.config import load_claim
from insurance_claims.prepare import reconciliation


def test_claim_contains_deliberate_400_dollar_mismatch() -> None:
    totals = reconciliation(load_claim())

    assert totals["building_attachment_acv"] == Decimal("67780.00")
    assert totals["contents_attachment_acv"] == Decimal(15880)
    assert totals["claimed_total"] == Decimal(81060)
    assert totals["attachment_total"] == Decimal("80660.00")
    assert totals["difference"] == Decimal("400.00")
