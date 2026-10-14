# ruff: noqa: RUF001
# Intentional Unicode in the source-qualified fixtures.
"""Synthetic outputs qualified against LightOnOCR's pinned published recipe.

Expected constants were produced with lightonai/LightOnOCR at
36755d461be079737860a5f03ae0c803501269e9: omit_margins, _join,
fix_escaped_dollars, then pp3. No document text or model inference is needed.
"""

import pytest
from sie_server.adapters.lighton_ocr._markdown import lightonocr3_markdown


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            "discard this prefix\n![HEADER+]( 0, 0, 1000, 100 )\nheader\n"
            "![text](0,100,1000,800)\nbody\nsecond line\n"
            "![aside_text+](0,0,100,100)\naside\n"
            "![footnote+](0,800,1000,900)\nretained footnote\n"
            "![footer](0,900,1000,980)\nfooter\n![page_number](0,980,1000,1000)\n42",
            "body\nsecond line\n\nretained footnote",
            id="block-omission-and-retained-footnote",
        ),
        pytest.param(r"We use \$x^2\$ here. Price \$5.", r"We use $x^2$ here. Price \$5.", id="math-and-currency"),
        pytest.param(r"$x\$\$x\$", "$x$$x$", id="successive-dollar-repair-stages"),
        pytest.param(
            "```tex\n\\$x^2\\$\n```\n\nInline `\\$y^2\\$` stays.\n\nMath \\$z^2\\$.",
            "```tex\n\\$x^2\\$\n```\n\nInline `\\$y^2\\$` stays.\n\nMath $z^2$.",
            id="fenced-and-inline-code",
        ),
        pytest.param(
            "| $x$ | $x^2$ |\n| --- | --- |\n| \\$y\\$ | \\$5 |",
            "| x | $x^2$ |\n| --- | --- |\n| y | \\$5 |",
            id="table-math",
        ),
        pytest.param(
            "preamble\n![text](0,0,1000,800)\nRésumé — 東京\nδεύτερη γραμμή\n![footnote](0,800,1000,1000)\n注釈 αβγ",
            "Résumé — 東京\nδεύτερη γραμμή\n\n注釈 αβγ",
            id="multiline-unicode",
        ),
        pytest.param("  first line\n\nsecond line\n  ", "first line\n\nsecond line", id="no-marker-fallback"),
        pytest.param(
            "Label .............\nName: _____________",
            "Label .............\nName: _____________",
            id="pp3-preserves-dots-and-form-blanks",
        ),
        pytest.param(
            r"We consider the variables \$x\$, \$y\$ and \$z\$ in this sentence.",
            "We consider the variables $x, y and z$ in this sentence.",
            id="pp3-linked-math",
        ),
        pytest.param(" \n\t ", "", id="blank"),
        pytest.param(
            "![header](0,0,1000,500)header\n![footer+](0,500,1000,1000)footer",
            "",
            id="all-margin",
        ),
    ],
)
def test_published_markdown_recipe(raw: str, expected: str) -> None:
    assert lightonocr3_markdown(raw) == expected


@pytest.mark.parametrize(
    "label",
    ["header", "footer", "page_number", "page_header", "page_footer", "header_image", "footer_image", "aside_text"],
)
def test_all_published_margin_labels_are_omitted(label: str) -> None:
    raw = f"![{label.upper()}++](0,0,1000,100)omit\n![text](0,100,1000,1000)body"
    assert lightonocr3_markdown(raw) == "body"


@pytest.mark.parametrize(
    "raw",
    ["![text1](0,0,100,100)body", "![text](-1,0,100,100)body", "![text](0.5,0,100,100)body"],
)
def test_marker_grammar_keeps_unrecognized_markers(raw: str) -> None:
    assert lightonocr3_markdown(raw) == raw


def test_empty_retained_blocks_do_not_add_separators() -> None:
    assert lightonocr3_markdown("![text](0,0,1,1)\n \n![footnote](1,1,2,2)note") == "note"
