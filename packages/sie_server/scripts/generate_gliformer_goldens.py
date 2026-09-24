#!/usr/bin/env python3
"""Generate reference outputs for the GLiFormer adapter parity tests.

Companion to ``packages/sie_server/tests/adapters/test_gliformer_parity.py``.

The script runs the upstream ``gliformer`` package directly, with no SIE code
involved, over a fixed set of texts. Each case records the SIE ``extract``
request under test next to the GLiFormer call that request must translate to,
and stores the package's raw output for that call. The parity test sends the
recorded requests through the adapter and checks the results against these
outputs, with structured values passed through GLiFormer's own output
formatter as ``GLiFormer.structure`` does.

It downloads the pinned checkpoint and runs it in float32 on the CPU, in any
environment with the package installed (inside an activated virtual
environment, the command below runs on that environment's packages). The
output records the stack it ran on under ``generated_with``:

    uv run --no-project --python 3.12 --with gliformer==0.1.2 python \\
        packages/sie_server/scripts/generate_gliformer_goldens.py \\
        --model knowledgator/gliformer-base-v1 \\
        --revision 590f9d3f577ea2f6d685aaec84b2d66b6db86b15 \\
        --out packages/sie_server/tests/adapters/goldens/gliformer/knowledgator__gliformer-base-v1.json
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from gliformer import GLiFormer
from huggingface_hub import snapshot_download

MAX_LENGTH = 2048
EMBEDDING_HEAD_DIMS = 32

TEXTS = [
    "Alice Johnson works at Acme Corp in London. She joined in March 2021 as a senior software engineer.",
    "Apple announced the iPhone 17 in Cupertino on Tuesday; Tim Cook called it the best phone ever made.",
    "The refund request from Maria Lopez for order 4471 was denied because the item was returned after 45 days.",
    "Dr. Maria Rossi prescribed 20 mg of atorvastatin to the patient at St. Mary's Hospital in Boston.",
]
ENTITY_TYPES = ["person", "organization", "location", "date", "product"]
RELATION_TYPES = ["works at", "located in", "chief executive of"]
DECISIONS = ["approved", "denied", "escalated"]
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "customer": {"type": "string"},
        "order number": {"type": "string"},
        "decision": {"type": "string", "enum": DECISIONS},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "role": {"type": "string"}},
            },
        },
    },
}
# The structuring template and classification group ``OUTPUT_SCHEMA`` maps to.
SCHEMA_STRUCTURES = {
    "$root": {"customer": "str", "order number": "str", "reasons": ["str"], "people": [{"name": "str", "role": "str"}]}
}
SCHEMA_CLASSES = {"decision": DECISIONS}
JOINT_RELATIONS = {None: {"entities": ENTITY_TYPES, "relations": RELATION_TYPES}}
LABEL_GROUPS = {
    "sentiment": ["positive", "negative", "neutral"],
    "topic": ["business", "technology", "health", "customer support"],
}
NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "patient": {"type": "object", "properties": {"name": {"type": "string"}, "hospital": {"type": "string"}}},
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"drug": {"type": "string"}, "dose": {"type": "string"}},
                "required": ["drug"],
            },
        },
        "employer": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "location": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        },
    },
}
NESTED_STRUCTURES = {
    "$root": {
        "patient": {"name": "str", "hospital": "str"},
        "medications": [{"drug": "str", "dose": "str"}],
        "employer": {"name": "str", "location": {"city": "str"}},
    }
}
# Entity spans supplied through item metadata for relation extraction.
METADATA_SPANS = [
    [("Alice Johnson", "person"), ("Acme Corp", "organization"), ("London", "location")],
    [("Apple", "organization"), ("Cupertino", "location"), ("Tim Cook", "person")],
    [("Maria Lopez", "person")],
    [("Dr. Maria Rossi", "person"), ("St. Mary's Hospital", "organization"), ("Boston", "location")],
]
ITEM_METADATA = [
    {
        "entities": [
            {"text": span, "label": label, "start": text.index(span), "end": text.index(span) + len(span)}
            for span, label in spans
        ]
    }
    for text, spans in zip(TEXTS, METADATA_SPANS, strict=True)
]

# (name, SIE extract request, GLiFormer.inference keyword arguments)
CASES: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    ("entities", {"labels": ENTITY_TYPES}, {"entities": ENTITY_TYPES}),
    (
        "classification",
        {"labels": ["positive", "negative", "neutral"], "options": {"classification_task": "sentiment"}},
        {"classes": {"sentiment": ["positive", "negative", "neutral"]}},
    ),
    (
        "classification_multi_label",
        {
            "labels": ["business", "technology", "health", "customer support"],
            "options": {"classification_task": "topic", "multi_label": True, "threshold": 0.3},
        },
        {
            "classes": {"topic": ["business", "technology", "health", "customer support"]},
            "multi_label": True,
            "threshold": 0.3,
        },
    ),
    (
        "relations",
        {"labels": ENTITY_TYPES, "options": {"relation_labels": RELATION_TYPES}},
        {"joint_relations": JOINT_RELATIONS},
    ),
    ("output_schema", {"output_schema": OUTPUT_SCHEMA}, {"classes": SCHEMA_CLASSES, "structures": SCHEMA_STRUCTURES}),
    (
        "combined",
        {"labels": ENTITY_TYPES, "output_schema": OUTPUT_SCHEMA, "options": {"relation_labels": RELATION_TYPES}},
        {"classes": SCHEMA_CLASSES, "joint_relations": JOINT_RELATIONS, "structures": SCHEMA_STRUCTURES},
    ),
    (
        "label_groups",
        {"labels": ENTITY_TYPES, "options": {"label_groups": LABEL_GROUPS}},
        {"entities": ENTITY_TYPES, "classes": LABEL_GROUPS},
    ),
    ("nested_objects", {"output_schema": NESTED_SCHEMA}, {"structures": NESTED_STRUCTURES}),
    # Relations between supplied entities: one call per distinct entity-type
    # set, in sorted order, as the adapter batches them.
    ("metadata_entities", {"labels": RELATION_TYPES, "item_metadata": ITEM_METADATA}, {}),
]


def _json_safe(kwargs: dict[str, Any]) -> dict[str, Any]:
    joint = kwargs.get("joint_relations")
    if joint is None:
        return kwargs
    return {**kwargs, "joint_relations": [{"name": name, **group} for name, group in joint.items()]}


def _metadata_reference(model: Any, request: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    """Joint relation calls grouped by each item's entity types."""
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, metadata in enumerate(request["item_metadata"]):
        types = tuple(sorted({entity["label"] for entity in metadata["entities"]}))
        groups.setdefault(types, []).append(index)
    outputs: dict[str, list[Any]] = {"ner": [[] for _ in TEXTS], "joint_relex": [[] for _ in TEXTS]}
    calls = []
    for types, indices in groups.items():
        joint = {None: {"entities": list(types), "relations": request["labels"]}}
        result = model.inference(
            [TEXTS[index] for index in indices], joint_relations=joint, threshold=0.5, batch_size=len(indices)
        )
        for task, per_text in outputs.items():
            for index, value in zip(indices, result.get(task, [[] for _ in indices]), strict=True):
                per_text[index] = value
        calls.append({"texts": indices, **_json_safe({"joint_relations": joint, "threshold": 0.5})})
    return calls, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    torch.manual_seed(0)
    model_dir = snapshot_download(args.model, revision=args.revision, ignore_patterns=["*.gif"])
    model = GLiFormer.from_pretrained(model_dir, load_tokenizer=True, map_location="cpu", max_length=MAX_LENGTH)
    model = model.to(dtype=torch.float32).eval()

    cases = []
    with torch.inference_mode():
        for name, request, reference_kwargs in CASES:
            if "item_metadata" in request:
                calls, outputs = _metadata_reference(model, request)
                cases.append({"name": name, "request": request, "reference_calls": calls, "reference_output": outputs})
                continue
            call = {"threshold": 0.5, **reference_kwargs}
            outputs = model.inference(TEXTS, batch_size=len(TEXTS), **call)
            cases.append(
                {"name": name, "request": request, "reference_call": _json_safe(call), "reference_output": outputs}
            )
        embeddings = model.embed_text(TEXTS, batch_size=len(TEXTS)).float()
    unit = torch.nn.functional.normalize(embeddings, dim=-1)

    golden = {
        "model": args.model,
        "revision": args.revision,
        "generated_with": {
            "gliformer": version("gliformer"),
            "gliner": version("gliner"),
            "transformers": version("transformers"),
            "torch": version("torch"),
            "device": "cpu",
            "dtype": "float32",
            "max_length": MAX_LENGTH,
        },
        "texts": TEXTS,
        "cases": cases,
        "embeddings": {
            "dim": int(unit.shape[1]),
            "normalized_head": [[round(value, 6) for value in row[:EMBEDDING_HEAD_DIMS]] for row in unit.tolist()],
            "cosine": [[round(value, 6) for value in row] for row in (unit @ unit.T).tolist()],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(golden, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
