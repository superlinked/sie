from decimal import Decimal

from invoice_totals.amounts import parse_amount, total


def test_plain_and_formatted_amounts():
    assert parse_amount("$1,234.50") == Decimal("1234.50")
    assert parse_amount("1234.5") == Decimal("1234.5")
    assert parse_amount("  12.00 ") == Decimal("12.00")


def test_credit_in_parentheses_is_negative():
    assert parse_amount("(12.00)") == Decimal("-12.00")
    assert parse_amount("($1,000.00)") == Decimal("-1000.00")


def test_total_nets_credits():
    assert total(["$100.00", "(12.50)", "2.50"]) == Decimal("90.00")
