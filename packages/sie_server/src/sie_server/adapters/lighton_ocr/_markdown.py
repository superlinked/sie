"""Pinned LightOnOCR-3 grounding-to-Markdown conversion; standard library only.

Adapted from lightonai/LightOnOCR at 36755d461be079737860a5f03ae0c803501269e9:
https://github.com/lightonai/LightOnOCR/tree/36755d461be079737860a5f03ae0c803501269e9

Sources: benchmarks/ocr_bench/providers/loocr_grounding.py (block omission),
benchmarks/ocr_bench/runner.py (page join), and lightonocr/postprocess/
fix_escaped_dollars.py and postprocess_ocr_markdown.py (dollar repair and PP3).
The exact recipe is page join after margin omission, then dollar repair, then PP3.

Local adaptations combine the required definitions in one module, omit CLI and
PP1/PP2 entrypoints, and call the PP3 branch directly without package imports.
Formatting and typing follow this repository; conversion semantics are preserved.
The upstream Apache-2.0 license is reproduced below.
"""

# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF001
# Unicode mathematical characters are part of the pinned conversion rules.
#
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright [yyyy] [name of copyright owner]
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from typing import Any

_BLOCK_MARKER_RE = re.compile(r"!\[([a-zA-Z_+]+)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]*")


MARGIN_LABELS = frozenset(
    {
        "header",
        "footer",
        "page_number",
        "page_header",
        "page_footer",
        "header_image",
        "footer_image",
        "aside_text",
    }
)


def _base_label(raw_label: str) -> str:
    return raw_label.strip().lower().rstrip("+")


class LoocrGroundingProvider:
    @staticmethod
    def parse_blocks(content: str) -> list[dict[str, Any]]:
        """The raw grounding string as ordered ``{label, bbox, text}`` blocks."""
        matches = list(_BLOCK_MARKER_RE.finditer(content))
        ends = [m.start() for m in matches[1:]] + [len(content)] if matches else []
        return [
            {
                "label": m.group(1),
                "bbox": [int(m.group(i)) for i in range(2, 6)],
                "text": content[m.end() : end].strip(),
            }
            for m, end in zip(matches, ends, strict=True)
        ]

    @classmethod
    def omit_margins(cls, content: str) -> str:
        """Raw grounding string -> markdown without margin blocks (``MARGIN_LABELS``) or markers.

        Block texts are joined by blank lines and text before the first marker is dropped; an output
        with no marker is kept as is.
        """
        return cls.omit_labels(content, MARGIN_LABELS)

    @classmethod
    def omit_labels(cls, content: str, labels: frozenset[str]) -> str:
        """Raw grounding string -> markdown without the blocks whose label (``+`` stripped) is in ``labels``."""
        blocks = cls.parse_blocks(content)
        if not blocks:
            return content.strip()
        return "\n\n".join(b["text"] for b in blocks if b["text"] and _base_label(b["label"]) not in labels)


def _join(texts: list[str]) -> str:
    return "\n\n".join(t.strip() for t in texts if t and t.strip())


ATOM = re.compile(
    r"\\(?:[A-Za-z]+|[^\r\n])|(?:\d+(?:\.\d*)?|\.\d+)|[^\W\d_]\w*|[-−–—+*/=<>:;,.'!|_^%&()\[\]{}]", re.UNICODE
)


def math_body(body: str) -> bool:
    r"""Conservative lexical evidence, not proof of mathematical correctness.

    No vocabulary of document/model-specific symbols. Alphabetic runs are
    identifiers; adjacent prose words are not accepted as implicit products.
    Command arguments can contain text (e.g. \\text{a long description}).
    """
    body = body.strip()
    if not body or "<td" in body.lower() or "</" in body:
        return False
    depth = 0
    for lexeme in re.findall(r"\\.|[{}]", body):
        if lexeme == "{":
            depth += 1
        elif lexeme == "}":
            depth -= 1
        if depth < 0:
            return False
    if depth:
        return False
    # Mask balanced command arguments for lexical checks; preserve the source.
    clean = []
    pos = 0
    while pos < len(body):
        m = re.match(r"\\[A-Za-z]+", body[pos:])
        if m:
            clean.append(m[0])
            pos += len(m[0])
            while pos < len(body) and body[pos] == "{":
                depth = 1
                pos += 1
                while pos < len(body) and depth:
                    if body[pos] == "\\":
                        pos += 2
                        continue
                    depth += (body[pos] == "{") - (body[pos] == "}")
                    pos += 1
                if depth:
                    return False
                clean.append("{}")
        else:
            clean.append(body[pos])
            pos += 1
    clean = "".join(clean)
    previous = None
    pos = 0
    for m in ATOM.finditer(clean):
        if clean[pos : m.start()].strip():
            return False
        word = bool(re.fullmatch(r"[^\W\d_]\w*", m[0], re.UNICODE))
        if word and previous and (len(m[0]) > 1 or len(previous) > 1):
            return False
        previous = m[0] if word else None
        pos = m.end()
    return not clean[pos:].strip()


def plausible_math(body: str) -> bool:
    stripped = body.strip()
    if not math_body(body) or re.fullmatch(r"[^\W\d_]{2,}", stripped):
        return False
    if body != stripped and not re.search(r"[\w\\]", stripped):
        return False
    if len(stripped) > 1 and stripped[0] in ",;":
        return False
    # Ignore escaped delimiters and command text arguments for bracket balance.
    # Mixed interval endpoints [a,b) are allowed; missing endpoints abstain.
    simple = re.sub(r"\\text\{[^{}]*\}", "", stripped)
    level = 0
    for lexeme in re.findall(r"\\.|[()\[\]]", simple):
        if lexeme in ("(", "["):
            level += 1
        elif lexeme in (")", "]"):
            level -= 1
        if level < 0:
            return False
    return level == 0


PROTECTED = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=fence)[ \t]*(?:\n|$)|\Z)"
    r"|(?P<ticks>`+)[^`\n]*(?P=ticks)"
    r"|^(?: {4}|\t)[^\n]*(?:\n|$)"
    r"|<!--.*?(?:-->|\Z)"
    r"|<(?:pre|code|script|style)\b[^>]*>.*?(?:</(?:pre|code|script|style)\s*>|\Z)"
    r'|<[/!?A-Za-z](?:"[^"]*"|\x27[^\x27]*\x27|[^\x27">])*>'
    r"|(?:https?://|mailto:|www\.)[^\s<>]+"
    r"|^[ \t]{0,3}\[[^\]\n]+\]:[^\n]*"
    r"|(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]|\\\(.*?\\\)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)


TOKEN = re.compile(r"(?<![\\$])\\?\$(?!\$)")


INLINE = re.compile(r"(?<![\\$])\$(?!\$)((?:\\.|[^$\\\n])*)(?<!\\)\$(?!\$)")


NUMERIC_START = re.compile(r"[+\-−]?\s*(?:\d|[.,]\d)")


def _ranges(text: str) -> list[tuple[int, int]]:
    spans = [(m.start(), m.end()) for m in PROTECTED.finditer(text)]
    # Protect balanced inline-link destinations, including relative URLs.
    for m in re.finditer(r"\]\(", text):
        pos = m.end()
        depth = 1
        while pos < len(text) and text[pos] != "\n" and depth:
            if text[pos] == "\\":
                pos += 2
                continue
            if text[pos] == "(":
                depth += 1
            elif text[pos] == ")":
                depth -= 1
            pos += 1
        if depth == 0:
            spans.append((m.start(), pos))
    # A literal dollar inside a complete valid-looking math span is not a broken
    # closing delimiter. Protect the complete span before attempting recovery.
    for m in INLINE.finditer(text):
        if plausible_math(m[1]):
            spans.append((m.start(), m.end()))
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def repair_dollar_delimiters(text: str, *, audit: list[dict[str, Any]] | None = None) -> tuple[str, Counter[str]]:
    """Return repaired text and counters; optional audit collects exact edits.

    Paired escaped numeric math is eligible; numeric opening-only spans abstain
    because they are indistinguishable locally from prices before math footnotes.
    Bare multi-letter words abstain. No scorer, model or reference access.
    """
    counts = Counter()
    edits = []

    def scan(start: int, end: int) -> None:
        part = text[start:end]
        tokens = list(TOKEN.finditer(part))
        i = 0
        while i + 1 < len(tokens):
            left, right = tokens[i : i + 2]
            body = part[left.end() : right.start()]
            el = left[0].startswith("\\")
            er = right[0].startswith("\\")
            if not el and not er:
                i += 2 if plausible_math(body) else 1
                continue
            numeric = bool(NUMERIC_START.match(body.strip()))
            after = part[right.end() :]
            before = part[: left.start()]
            env = re.fullmatch(r"\s*\\begin\{([A-Za-z*]+)\}.*\\end\{\1\}\s*", body, re.DOTALL)
            # Currency code glued to a dollar, e.g. US$, S$, HK$.
            code_currency = el and bool(re.search(r"(?<!\w)[A-Z]{1,3}$", before))
            ambiguous = (
                (("\n" in body or "\r" in body) and not env)
                or bool(re.search(r"\n[ \t]*\n", body))
                or code_currency
                or (el and not er and numeric)
                or (numeric and bool(NUMERIC_START.match(after.lstrip(" \t"))))
                or bool(re.fullmatch(r"[^\W\d_]{2,}", body.strip()))
                or not plausible_math(body)
            )
            if ambiguous:
                counts["skipped_ambiguous_pairs"] += 1
                i += 1
                continue
            a = start + left.start()
            b = start + right.end()
            replacement = "$" + body + "$"
            edits.append((a, b, replacement))
            counts["repaired_pairs"] += 1
            counts["removed_backslashes"] += el + er
            if audit is not None:
                audit.append(
                    {
                        "start": a,
                        "end": b,
                        "before": text[a:b],
                        "after": replacement,
                        "context": text[max(0, a - 90) : b + 90],
                    }
                )
            i += 2

    previous = 0
    for a, b in _ranges(text):
        scan(previous, a)
        previous = b
    scan(previous, len(text))
    for a, b, replacement in reversed(edits):
        text = text[:a] + replacement + text[b:]
    return text, counts


def fix_escaped_dollars(text: str) -> str:
    """Convert one OCR Markdown string; no other formatting transformations."""
    return repair_dollar_delimiters(text)[0]


PROSE_DOLLARS = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)")


OPAQUE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=fence)[ \t]*(?:\n|$)|\Z)"
    r"|(?P<ticks>`+)[^`\n]*(?P=ticks)"
    r"|(?<!\\)\$\$.*?(?<!\\)\$\$"
    r"|\\\[.*?\\\]|\\\(.*?\\\)"
    r"|(?<![\\$])\$(?!\$)(?:\\.|[^$\n\\])*?(?<!\\)\$(?!\$)",
    re.MULTILINE | re.DOTALL,
)


def normalize_table_math(text: str) -> tuple[str, Counter[str]]:
    """Render only simple inline math as text inside HTML/Markdown cells.

    No fractions, scripts, arbitrary commands, display equations or currency
    amounts are rewritten. This preserves mathematical content and table shape.
    """
    counts: Counter[str] = Counter()
    symbols = {
        "diamond": "◊",
        "times": "×",
        "pm": "±",
        "leq": "≤",
        "geq": "≥",
        "neq": "≠",
        "alpha": "α",
        "beta": "β",
        "gamma": "γ",
        "mu": "μ",
        "sigma": "σ",
        "Delta": "Δ",
    }
    inline = re.compile(r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)|\\\(([^\n]+?)\\\)")

    def cell(content: str) -> str:
        changed = False

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            inner = match.group(1) if match.group(1) is not None else match.group(2)
            rendered = re.sub(r"\\([A-Za-z]+)", lambda m: symbols.get(m[1], m[0]), inner).strip()
            if (
                not re.fullmatch(r"[A-Za-z0-9\s.,()+\-=/<>αβγμσΔ◊×±≤≥≠]+", rendered)
                or not re.search(r"[A-Za-zαβγμσΔ◊×±≤≥≠]", rendered)
                or any(len(word) > 1 for word in re.findall(r"[A-Za-z]+", rendered))
            ):
                return match.group(0)
            changed = True
            counts["normalized_table_math_spans"] += 1
            return rendered

        result = inline.sub(replace, content)
        if changed:
            result = re.sub(r"\(\s+", "(", result)
            result = re.sub(r"\s+\)", ")", result)
        return result

    def transform(part: str) -> str:
        part = re.sub(
            r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]\s*>)",
            lambda m: m[1] + cell(m[2]) + m[3],
            part,
            flags=re.IGNORECASE | re.DOTALL,
        )
        lines = part.splitlines(keepends=True)
        separator = re.compile(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*")
        index = 1
        while index < len(lines):
            if separator.fullmatch(lines[index]) and "|" in lines[index - 1]:
                lines[index - 1] = cell(lines[index - 1])
                index += 1
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    lines[index] = cell(lines[index])
                    index += 1
            else:
                index += 1
        return "".join(lines)

    # Apply to text outside code; existing math is handled by the cell parser.
    code = re.compile(
        r"^[ \t]*(?P<f>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=f)[ \t]*(?:\n|$)|\Z)|(?P<t>`+)[^`\n]*(?P=t)",
        re.MULTILINE | re.DOTALL,
    )
    parts, start = [], 0
    for match in code.finditer(text):
        parts.extend((transform(text[start : match.start()]), match[0]))
        start = match.end()
    parts.append(transform(text[start:]))
    return "".join(parts), counts


CODE = re.compile(
    r"^[ \t]*(?P<f>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=f)[ \t]*(?:\n|$)|\Z)"
    r"|(?P<t>`+)[^`\n]*(?P=t)",
    re.MULTILINE | re.DOTALL,
)


def outside_matches(text: str, protected: re.Pattern[str], transform: Callable[[str], str]) -> str:
    parts, start = [], 0
    for match in protected.finditer(text):
        parts.extend((transform(text[start : match.start()]), match[0]))
        start = match.end()
    parts.append(transform(text[start:]))
    return "".join(parts)


def transform_cells(text: str, cell: Callable[[str], str]) -> str:
    """Visit HTML cells and cells in Markdown tables with separator rows."""

    def transform(part: str) -> str:
        part = re.sub(
            r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]\s*>)",
            lambda m: m[1] + cell(m[2]) + m[3],
            part,
            flags=re.IGNORECASE | re.DOTALL,
        )
        lines = part.splitlines(keepends=True)
        separator = re.compile(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*")

        def row(line: str) -> str:
            # Escaped pipes and math containing pipes are ambiguous: preserve them.
            if "\\|" in line or "$" in line:
                return line
            return "|".join(cell(value) for value in line.split("|"))

        index = 1
        while index < len(lines):
            if separator.fullmatch(lines[index]) and "|" in lines[index - 1]:
                lines[index - 1] = row(lines[index - 1])
                index += 1
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    lines[index] = row(lines[index])
                    index += 1
            else:
                index += 1
        return "".join(lines)

    return outside_matches(text, CODE, transform)


def normalize_table_spacing(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()

    def cell(value: str) -> str:
        body = value.strip()
        if re.fullmatch(r"\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?", body):
            new = re.sub(r"\s*:\s*", ":", body)
        elif re.fullmatch(r"[+-]?\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:\.\d+)?", body):
            new = re.sub(r"[ \u00a0\u202f]", "", body)
        elif re.fullmatch(r"[μµ]\s+(?:m|g|l|L|s|A|V|W|F|M)", body):
            new = re.sub(r"\s+", "", body)
        else:
            return value
        if new != body:
            counts["normalized_table_spacing_cells"] += 1
        return value.replace(body, new, 1)

    return transform_cells(text, cell), counts


def remove_table_leaders(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()

    def cell(value: str) -> str:
        if not re.search(r"[^\W\d_]", value) or re.search(r"[<>$\\]", value):
            return value
        new, number = re.subn(r"[ \t]*(?:\.{3,}|…{2,})[ \t]*(?=\s*$)", "", value)
        counts["removed_table_leaders"] += number
        return new

    return transform_cells(text, cell), counts


def shorten_form_blanks(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()

    def transform(part: str) -> str:
        # Only standalone blanks on lines containing a field label.
        def line(value: str) -> str:
            if not re.search(r"[^\W\d_]", value) or value.lstrip().startswith(("<", "|")):
                return value
            new, number = re.subn(r"(?<!\S)_{10,}(?!\S)", "________", value)
            counts["shortened_form_blanks"] += number
            return new

        return "".join(line(value) for value in part.splitlines(keepends=True))

    return outside_matches(text, OPAQUE, transform), counts


def join_adjacent_math(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    inline = re.compile(r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)")

    def transform(part: str) -> str:
        matches = list(inline.finditer(part))
        if not matches:
            return part
        parts, start, index = [], 0, 0

        def eligible(body: str) -> bool:
            # Require an equation/relation in each span; currency and prose
            # cannot join. Math itself is copied verbatim.
            return bool(re.search(r"[=<>]|\\(?:leq|geq|neq)\b", body))

        while index < len(matches):
            first = last = matches[index]
            body = first[1]
            while index + 1 < len(matches) and eligible(last[1]):
                following = matches[index + 1]
                gap = part[last.end() : following.start()]
                if not re.fullmatch(r"[ \t]*[,;][ \t]*", gap) or not eligible(following[1]):
                    break
                body += gap + following[1]
                last = following
                index += 1
                counts["joined_adjacent_math_spans"] += 1
            parts.extend((part[start : first.start()], "$" + body + "$"))
            start = last.end()
            index += 1
        parts.append(part[start:])
        return "".join(parts)

    protected = re.compile(
        CODE.pattern + r"|(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]|\\\(.*?\\\)", re.MULTILINE | re.DOTALL
    )
    return outside_matches(text, protected, transform), counts


def normalize_prose_math(text: str) -> tuple[str, Counter[str]]:
    """Render simple inline notation on prose lines; preserve actual formulas."""
    counts: Counter[str] = Counter()
    symbols = {
        "alpha": "α",
        "beta": "β",
        "gamma": "γ",
        "delta": "δ",
        "Delta": "Δ",
        "mu": "μ",
        "sigma": "σ",
        "theta": "θ",
        "pi": "π",
        "omega": "ω",
        "tanh": "tanh",
    }
    protected = re.compile(
        CODE.pattern + r"|<table\b.*?</table>"
        r"|(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )

    def transform(part: str) -> str:
        lines = []
        for line in part.splitlines(keepends=True):
            # Mixed dollar errors are deliberately preserved, as requested.
            prose = PROSE_DOLLARS.sub("", line)
            if (
                "\\$" in line
                or "|" in line
                or line.lstrip().startswith(("<", "!["))
                or len(re.findall(r"\b[A-Za-z]{2,}\b", prose)) < 3
            ):
                lines.append(line)
                continue

            def replace(match: re.Match[str]) -> str:
                body = match[1].strip()
                # A prose sentence can contain a complete equation. Retain its
                # delimiters so formula consumers can still extract it.
                if any(char in body for char in "=<>"):
                    return match[0]
                rendered = re.sub(r"\\([A-Za-z]+)", lambda m: symbols.get(m[1], m[0]), body)
                if (
                    not re.fullmatch(r"[A-Za-zαβγδΔμσθπω0-9 .,+/()=<>-]+", rendered)
                    or not re.search(r"[A-Za-zαβγδΔμσθπω]", rendered)
                    or any(len(w) > 1 and w != "tanh" for w in re.findall(r"[A-Za-z]+", rendered))
                ):
                    return match[0]
                counts["normalized_prose_math_spans"] += 1
                return rendered

            lines.append(PROSE_DOLLARS.sub(replace, line))
        return "".join(lines)

    return outside_matches(text, protected, transform), counts


def normalize_prose_math_lists(text: str) -> tuple[str, Counter[str]]:
    """Keep linked math spans wrapped; normalize isolated prose notation as PP1."""
    counts = Counter()
    protected = re.compile(
        CODE.pattern + r"|<table\b.*?</table>|(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )

    def transform(part: str) -> str:
        lines = []
        for line in part.splitlines(keepends=True):
            spans = list(PROSE_DOLLARS.finditer(line))
            keep = set()
            for left, right in zip(spans, spans[1:]):  # noqa: RUF007 - pinned adjacent-span recipe
                gap = line[left.end() : right.start()]
                if re.fullmatch(r"[ \t]*(?:[,;][ \t]*(?:(?:and|or)[ \t]+)?|(?:and|or)[ \t]+)", gap):
                    keep.update((left.start(), right.start()))
            # Use original normalization on the complete line to preserve its
            # eligibility checks, then reconstruct from each span's decision.
            eligible = normalize_prose_math(line)[0]
            if eligible == line or not keep:
                value, c = normalize_prose_math(line)
                counts.update(c)
                lines.append(value)
                continue
            prose = PROSE_DOLLARS.sub("", line)

            def replace(m: re.Match[str]) -> str:
                if m.start() in keep:
                    counts["preserved_linked_math_spans"] += 1
                    return m[0]
                # Original renderer with the same surrounding prose eligibility.
                prefix = " ".join(prose.splitlines())
                probe = prefix + m[0]
                value, c = normalize_prose_math(probe)
                counts.update(c)
                return value[len(prefix) :]

            lines.append(PROSE_DOLLARS.sub(replace, line))
        return "".join(lines)

    return outside_matches(text, protected, transform), counts


def repair_latex_syntax(text: str) -> tuple[str, Counter[str]]:
    """Expand known operators, escape text ampersands, reconnect split fences.

    Never change dollar escaping or invent missing delimiters/content.
    """
    counts: Counter[str] = Counter()
    math = re.compile(
        r"(?<!\\)\$\$(.*?)\$\$|\\\[(.*?)\\\]"
        r"|(?<![\\$])\$(?!\$)([^$\n]*?)(?<!\\)\$(?!\$)|\\\((.*?)\\\)",
        re.DOTALL,
    )
    display = re.compile(r"(?<!\\)\$\$(.*?)\$\$", re.DOTALL)

    def transform(part: str) -> str:
        def repair(match: re.Match[str]) -> str:
            original = match[0]
            # Leave every expression containing escaped dollars untouched.
            if "\\$" in original:
                return original
            new, n = re.subn(r"\\esssup(?![A-Za-z])", lambda _m: r"\operatorname*{ess\,sup}", original)
            counts["expanded_esssup"] += n

            def textext(m: re.Match[str]) -> str:
                body, n = re.subn(r"(?<!\\)&", r"\\&", m[1])
                counts["escaped_text_ampersands"] += n
                return r"\text{" + body + "}"

            return re.sub(r"\\text\{([^{}]*)\}", textext, new)

        part = math.sub(repair, part)
        matches = list(display.finditer(part))
        parts, start, i = [], 0, 0

        def balance(body: str) -> int:
            level = 0
            for lexeme in re.findall(r"\\(left|right)\b", body):
                level += 1 if lexeme == "left" else -1
                if level < 0:
                    return -1
            return level

        while i < len(matches):
            first = matches[i]
            body, end, j = first[1], first.end(), i
            # Join only adjacent display blocks until an existing \left is
            # closed by an existing \right, without intervening prose.
            while balance(body) > 0 and j + 1 < len(matches):
                following = matches[j + 1]
                if part[end : following.start()].strip():
                    break
                body += "\n" + following[1]
                end, j = following.end(), j + 1
                if balance(body) == 0:
                    break
            if j > i and balance(body) == 0 and "\\$" not in body:
                parts.extend((part[start : first.start()], "$$" + body + "$$"))
                counts["joined_split_display_blocks"] += j - i
                start, i = end, j + 1
            else:
                i += 1
        parts.append(part[start:])
        return "".join(parts)

    return outside_matches(text, CODE, transform), counts


TRANSFORMS = {
    "table-math": normalize_table_math,
    "table-spacing": normalize_table_spacing,
    "table-leaders": remove_table_leaders,
    "form-blanks": shorten_form_blanks,
    "adjacent-math": join_adjacent_math,
    "prose-math": normalize_prose_math,
    "latex-syntax": repair_latex_syntax,
}


def merge_inline_math(text: str) -> tuple[str, Counter[str]]:
    counts = Counter()
    marker = "$"
    pattern = r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)"
    formula = re.compile(pattern, re.DOTALL)
    protected = CODE.pattern + r"|^.*\\\$.*$"
    protected += r"|(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]|\\\(.*?\\\)"
    # Avoid DOTALL on the escaped-dollar line alternative.
    protected = re.compile(protected.replace(r"^.*\\\$.*$", r"^[^\n]*\\\$[^\n]*$"), re.MULTILINE | re.DOTALL)

    def transform(part: str) -> str:
        matches = list(formula.finditer(part))
        pieces, start, i = [], 0, 0

        def eligible(m: re.Match[str]) -> bool:
            return not re.fullmatch(r"-?\d{1,3}(?:,\d{3})*", m[1].strip())

        while i < len(matches):
            first = last = matches[i]
            body = first[1]
            while eligible(last) and i + 1 < len(matches) and eligible(matches[i + 1]):
                following = matches[i + 1]
                gap = part[last.end() : following.start()]
                if not re.fullmatch(r"(?:\s|,|and)*", gap, re.IGNORECASE):
                    break
                body += (gap if gap.strip() else " ") + following[1]
                last = following
                counts["merged_inline_pairs"] += 1
                i += 1
            pieces.extend([part[start : first.start()], marker + body + marker])
            start = last.end()
            i += 1
        pieces.append(part[start:])
        return "".join(pieces)

    return outside_matches(text, protected, transform), counts


TRANSFORMS["infinity-inline"] = merge_inline_math


PP1_STEPS = (
    "table-math",
    "table-spacing",
    "table-leaders",
    "form-blanks",
    "adjacent-math",
    "latex-syntax",
    "prose-math",
    "infinity-inline",
)


PP3_STEPS = tuple(name for name in PP1_STEPS if name not in ("table-leaders", "form-blanks"))


def _pp3(text: str) -> str:
    """Apply the pinned cleanup(profile="pp3") branch."""
    text, _ = repair_dollar_delimiters(text)
    for name in PP3_STEPS:
        transform = normalize_prose_math_lists if name == "prose-math" else TRANSFORMS[name]
        text, _ = transform(text)
    return text


def lightonocr3_markdown(raw_response: str) -> str:
    """Convert one generated page using LightOnOCR's published Markdown recipe.

    Margin/aside blocks and text before the first grounding marker are omitted.
    Footnotes remain; a page containing only omitted blocks becomes empty.
    """
    markdown = _join([LoocrGroundingProvider.omit_margins(raw_response)])
    # The pinned runner applies POSTPROCESS before FOLDER_POSTPROCESS (PP3).
    # Preserve both stages even though PP3 also calls dollar repair.
    markdown = fix_escaped_dollars(markdown)
    return _pp3(markdown)
