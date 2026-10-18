"""Parse the amount column of an exported invoice into Decimal values."""

from decimal import Decimal


def parse_amount(text: str) -> Decimal:
    """Turn one amount cell into a Decimal.

    Accepts "$1,234.50", "1234.5" and "  12.00 ". A credit is written with
    parentheses, "(12.00)", and is a negative amount.
    """
    cleaned = text.strip().replace("$", "").replace(",", "")
    return Decimal(cleaned)


def total(cells: list[str]) -> Decimal:
    """Net total of the amount column, credits subtracted."""
    return sum((parse_amount(cell) for cell in cells), Decimal(0))
