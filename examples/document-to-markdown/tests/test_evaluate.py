from __future__ import annotations

from document_to_markdown.evaluate import _evaluate_markdown, _table_count


def test_table_count_finds_markdown_tables() -> None:
    markdown = """
| Metric | Q4 |
| --- | ---: |
| Revenue | $39,331 |

| Product | Revenue |
| :--- | ---: |
| Data Center | $35,580 |
"""

    assert _table_count(markdown) == 2


def test_evaluation_checks_content_order_and_tables() -> None:
    markdown = """
# Q4 Fiscal 2025 Summary

Revenue reached $39,331.

## Fiscal 2025 Summary

| Metric | Value |
| --- | ---: |
| Data Center | $35,580 |
"""

    checks = _evaluate_markdown(
        markdown,
        required=("Revenue", "$39,331", "Data Center"),
        ordered=("Q4 Fiscal 2025 Summary", "Fiscal 2025 Summary", "Data Center"),
        tables=1,
    )

    assert all(check.passed for check in checks)


def test_evaluation_surfaces_wrong_reading_order() -> None:
    markdown = "Second section\n\nFirst section"

    checks = _evaluate_markdown(
        markdown,
        required=(),
        ordered=("First section", "Second section"),
        tables=0,
    )

    assert next(check for check in checks if check.name == "reading-order").passed is False
