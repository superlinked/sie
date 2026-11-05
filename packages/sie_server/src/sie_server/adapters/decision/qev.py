"""The qev-labels-v2 text and slot contract used by Elephant and Mammoth."""

from __future__ import annotations

import json
import string
from typing import Any

from sie_server.adapters.decision.base import check_token_budget
from sie_server.adapters.decision.questions import Question

SYSTEM = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
_SPLIT_CODES = frozenset(
    [
        "BQ",
        "BZ",
        "CJ",
        "CQ",
        "CZ",
        "DQ",
        "DZ",
        "EJ",
        "EY",
        "FJ",
        "FQ",
        "FV",
        "FZ",
        "GJ",
        "GK",
        "GQ",
        "GZ",
        "HJ",
        "IY",
        "JF",
        "JG",
        "JH",
        "JL",
        "JN",
        "JQ",
    ]
)
SLOTS = (
    *string.ascii_uppercase,
    *[a + b for a in string.ascii_uppercase for b in string.ascii_uppercase if a + b not in _SPLIT_CODES][:230],
)


def entry(value: Any) -> str:
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def suffix(question: Question) -> str:
    source = question.source
    instructions = entry(source.get("instructions"))
    criteria = source.get("criteria")
    if question.kind == "noul":
        instructions = instructions or "Is the statement true, or is the answer to the question yes?"
        criteria = criteria or {}
        names = ["yes", "no"]
        descriptions = [
            entry(criteria["true"])
            if criteria.get("true") is not None
            else "the statement is true / the answer is yes",
            entry(criteria["false"])
            if criteria.get("false") is not None
            else "the statement is false / the answer is no",
        ]
    elif question.kind == "choice":
        instructions = instructions or "Which option applies to the state?"
        names = list(question.keys)
        descriptions = [entry(value) for value in source["criteria"].values()]
    else:
        instructions = instructions or "Which level describes the state?"
        names = [f"level {i}" for i in range(len(question.keys))]
        descriptions = [entry(value) for value in source["criteria"]]
    options = "\n".join(
        f"{code}. {name}: {description}" if description else f"{code}. {name}"
        for code, name, description in zip(SLOTS, names, descriptions)
    )
    header = "Rate the state:" if question.kind == "score" else "Question:"
    unit = "letter" if len(names) <= 26 else "label"
    return (
        f"{header} {instructions}\n\nOptions:\n{options}\n\nAnswer with one {unit}: {', '.join(SLOTS[: len(names)])}."
    )


def prompts(
    tokenizer: Any, state: Any, questions: tuple[Question, ...], *, max_seq_length: int = 16384
) -> list[tuple[list[int], int]]:
    body = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)
    prefix = f"<state>\n{body}\n</state>\n\n"

    def render(ending: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prefix + ending}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    full = []
    total = 0
    for question in questions:
        ids = tokenizer.encode(render(suffix(question)), add_special_tokens=False)
        total = check_token_budget(len(ids), total, max_seq_length)
        full.append(ids)
    text = render("QEV_PREFIX_MARK")
    prefix_ids = tokenizer.encode(text[: text.index("QEV_PREFIX_MARK")], add_special_tokens=False)
    size = len(prefix_ids)
    while size and any(ids[:size] != prefix_ids[:size] for ids in full):
        size -= 1
    if not size or any(size >= len(ids) for ids in full):
        raise ValueError("qev prompt needs a shared prefix and question suffix")
    return [(ids, size) for ids in full]
