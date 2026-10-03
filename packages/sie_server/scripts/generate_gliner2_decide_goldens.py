#!/usr/bin/env python3
"""Generate reference outputs for the GLiNER2.5-Decide adapter parity tests.

Companion to ``packages/sie_server/tests/adapters/test_gliner2_decide_parity.py``.

The script runs the ``gliner2`` package directly, with no SIE code involved:
``AutoExtractor`` loads the pinned checkpoint and ``gliner2.classification``
scores it in float32 on the CPU. Each case records the SIE ``extract`` request
under test next to the ``ClassificationSchema`` that request must translate to,
and stores every label's probability (``ClassificationScores.probability``) and
a hash of the encoder row the package builds for each text.

A long document is read only as far as the model window: the words whose tokens
fit after the task prompt (``window`` minus the prompt, as the package's
processor builds and tokenizes both). The script computes that word count with
the package's own processor, passes it as ``ClassificationConfig(max_len=...)``,
and records it.

Run it with the stack the models are served on (the transformers5 bundle); the
output records that stack under ``generated_with``:

    uv run --no-project --python 3.12 --with gliner2==2.0.0 --with transformers==5.17.0 \\
        --with torch==2.9.1 python packages/sie_server/scripts/generate_gliner2_decide_goldens.py \\
        --model fastino/GLiNER2.5-Decide --revision 7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6 --window 512 \\
        --out packages/sie_server/tests/adapters/goldens/gliner2_decide/fastino__GLiNER2.5-Decide.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from gliner2 import AutoExtractor
from gliner2.classification import ClassificationConfig, ClassificationSchema, Classifier
from huggingface_hub import snapshot_download

CHECKPOINT_FILES = ["config.json", "encoder_config/*", "tokenizer*", "special_tokens_map.json", "model.safetensors"]

TEXTS = [
    "My subscription renewed on April 15 for 5,400 yen after the service was already down. "
    "Can I get that charge refunded?",
    "Guest in room 1408 says the AC has been out since yesterday and they want to move tonight or leave. "
    "They also asked for the incidentals hold to be released.",
    "Das Paket kam beschädigt an, bitte schicken Sie Ersatz. Ich brauche es bis Freitag, sonst storniere ich "
    "die Bestellung!",
    "请帮我取消订单，我不想要了。退款什么时候到账？",
    "The treaty was signed in Paris in 1992. It entered into force the following year, after the last signatory "
    "ratified it.",
]
_REPORT = [
    "Quarterly operations report for the northern warehouse, week {n}.",
    "The team shipped {orders} orders and handled {returns} returns; late deliveries fell to {late} percent.",
    "Carrier contract {n} renewed without changes, and the forklift inspection passed on schedule.",
    "Payroll ran on time for {staff} staff members and no overtime disputes were filed.",
]
LONG_TEXT = " ".join(
    sentence.format(n=n, orders=12000 + 37 * n, returns=300 + n, late=max(1, 9 - n % 8), staff=40 + n % 5)
    for n in range(60)
    for sentence in _REPORT
) + (
    " In the last week the payroll system failed during the month-end close and salaries for 40 staff are "
    "delayed; finance needs a fix before the Friday 5pm cutoff"
)

INTENTS = {
    "refund_request": "the customer wants money back",
    "cancel_order": "the customer wants to cancel an order or subscription",
    "room_change": "a guest wants another room",
    "maintenance": None,
    "other": "",
}
QUESTIONS = {
    "intent": {"type": "choice", "instructions": "What does the customer want?", "criteria": INTENTS},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "within a week", "today", "blocking or past a deadline"],
    },
    "needs_human": {
        "type": "noul",
        "instructions": "Must a person act on this?",
        "criteria": {"true": "automation cannot resolve it"},
    },
    "rating": {
        "type": "score",
        "instructions": "Rate the severity from 0 to 10.",
        "criteria": [str(i) for i in range(11)],
    },
}
PASSAGE_QUESTION = {"answer": {"type": "noul", "instructions": "Did the treaty enter into force in 1992?"}}
GROUPS = {
    "intent": ["refund_request", "cancel_order", "room_change", "maintenance", "other"],
    "priority": ["low", "normal", "high", "urgent"],
    "needs_human": ["yes", "no"],
}
MULTI_GROUPS = {"topics": ["billing", "hvac", "shipping", "payroll", "account"], "channel": ["email", "chat", "phone"]}
LABELS = ["billing", "technical", "hospitality", "logistics", "other"]


def _single(name: str, labels: Any, instruction: str | None = None) -> dict[str, Any]:
    return {"name": name, "kind": "single", "labels": labels, "instruction": instruction}


def _multi(name: str, labels: Any, instruction: str | None = None) -> dict[str, Any]:
    return {"name": name, "kind": "multi", "labels": labels, "instruction": instruction}


def _described(criteria: dict[str, str | None]) -> dict[str, str | None]:
    return {label: description or None for label, description in criteria.items()}


# Each case: the SIE request, the documents it reads, and the gliner2 tasks it must become.
CASES: list[dict[str, Any]] = [
    {
        "name": "questions",
        "request": {"output_schema": QUESTIONS},
        "documents": "texts",
        "tasks": [
            _single("intent", _described(INTENTS), "What does the customer want?"),
            _single(
                "urgency",
                {"0": "not urgent", "1": "within a week", "2": "today", "3": "blocking or past a deadline"},
                "How urgent is this request?",
            ),
            _single("needs_human", {"yes": "automation cannot resolve it", "no": None}, "Must a person act on this?"),
            _single("rating", [str(i) for i in range(11)], "Rate the severity from 0 to 10."),
        ],
    },
    {
        "name": "question_over_passage",
        "request": {"output_schema": PASSAGE_QUESTION},
        "documents": "texts",
        "tasks": [_single("answer", ["yes", "no"], "Did the treaty enter into force in 1992?")],
    },
    {
        "name": "label_groups",
        "request": {"instruction": "Triage the message.", "options": {"label_groups": GROUPS}},
        "documents": "texts",
        "tasks": [_single(name, labels, "Triage the message.") for name, labels in GROUPS.items()],
    },
    {
        "name": "label_groups_multi_label",
        "request": {"options": {"label_groups": MULTI_GROUPS, "classification_type": "multi-label"}},
        "documents": "texts",
        "tasks": [_multi(name, labels) for name, labels in MULTI_GROUPS.items()],
    },
    {
        "name": "labels",
        "request": {"labels": LABELS, "instruction": "Which team should handle this?"},
        "documents": "texts",
        "tasks": [_single("label", LABELS, "Which team should handle this?")],
    },
    {
        "name": "long_document",
        "request": {"output_schema": QUESTIONS},
        "documents": "long",
        "tasks": None,  # same tasks as "questions"
    },
]


def build_schema(tasks: list[dict[str, Any]]) -> ClassificationSchema:
    schema = ClassificationSchema()
    for task in tasks:
        add = schema.single if task["kind"] == "single" else schema.multi
        add(task["name"], task["labels"], instruction=task["instruction"])
    return schema


def readable_words(processor: Any, model_schema: dict[str, Any], text: str, window: int) -> int | None:
    """Words of ``text`` the model reads after the task prompt, or None when all of them fit."""
    prefix = processor.transform_and_format(".", model_schema).text_word_first_positions[0]
    room = window - prefix
    if not text.endswith((".", "!", "?")):
        text += "."  # as the processor's collate does
    used = 0
    for count, (word, _, _) in enumerate(processor.word_splitter(text, lower=True)):
        tokens = len(processor.tokenizer.tokenize(word))
        if used + tokens > room:
            return count
        used += tokens
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--window", type=int, required=True, help="the model config's max_sequence_length")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    path = snapshot_download(args.model, revision=args.revision, allow_patterns=CHECKPOINT_FILES)
    model = AutoExtractor.from_pretrained(path, map_location="cpu")
    model.eval()
    classifier = Classifier(model)
    processor = model.processor

    cases = []
    for case in CASES:
        tasks = case["tasks"] if case["tasks"] is not None else CASES[0]["tasks"]
        compiled = classifier.compile_schema(build_schema(tasks))
        model_schema = compiled.build()
        documents = TEXTS if case["documents"] == "texts" else [LONG_TEXT]
        results = []
        for text in documents:
            max_len = readable_words(processor, model_schema, text, args.window)
            batch = processor.collate_fn_inference([(text, model_schema)], max_len=max_len)
            row = batch.input_ids[0, : batch.original_lengths[0]].tolist()
            assert len(row) <= args.window, "reference row exceeds the window"
            with torch.inference_mode():
                scores = classifier.score(text, compiled, config=ClassificationConfig(max_len=max_len))
            probabilities = {
                task["name"]: {label: scores.probability(task["name"], label) for label in scores.tasks[task["name"]]}
                for task in tasks
            }
            results.append(
                {
                    "max_len": max_len,
                    "row_length": len(row),
                    "row_sha256": hashlib.sha256(json.dumps(row).encode()).hexdigest(),
                    "probabilities": probabilities,
                }
            )
        cases.append(
            {
                "name": case["name"],
                "request": case["request"],
                "documents": case["documents"],
                "reference_call": {"tasks": tasks},
                "results": results,
            }
        )

    golden = {
        "model": args.model,
        "revision": args.revision,
        "generated_with": {
            "gliner2": version("gliner2"),
            "transformers": version("transformers"),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "window": args.window,
        },
        "texts": TEXTS,
        "long_text": LONG_TEXT,
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(golden, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
