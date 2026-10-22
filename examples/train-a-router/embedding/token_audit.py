"""Count exact encoder inputs on a remote CPU; never truncate or call a model."""

import argparse
from pathlib import Path

from embedding import recipes as core

TEMPLATE = "Instruct: {instruction}\nQuery: {text}"
MAX_LENGTH = 32768


def audit_rows(rows, tokenizer):
    result = []
    for start in range(0, len(rows), 128):
        batch = rows[start : start + 128]
        encoded = tokenizer(
            [TEMPLATE.format(instruction=core.INSTRUCTION, text=r["request"]) for r in batch], truncation=False
        )["input_ids"]
        core.require(len(encoded) == len(batch), "Tokenizer population changed")
        for row, tokens in zip(batch, encoded, strict=True):
            core.require(len(tokens) > 0, "Empty encoded request")
            result.append({"id": row["id"], "input_tokens": len(tokens)})
    return result


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--tokenizer-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    binding = core.read(args.tokenizer_receipt)
    core.require(
        binding["model"] == core.MODEL and binding["hf_revision"] == core.REVISION, "Tokenizer revision changed"
    )
    for name, expected in binding["files_sha256"].items():
        relative = Path(name)
        core.require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and core.sha(args.tokenizer / relative) == expected,
            "Tokenizer file binding changed",
        )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    plan = core.read(args.source / "source-plan.json")
    for role in core.ROLES:
        source = args.source / f"question-only/{role}-inputs.json"
        core.require(
            core.sha(source) == plan["files_sha256"][source.relative_to(args.source).as_posix()],
            "Question inputs changed",
        )
        counts = audit_rows(core.read(source), tokenizer)
        result = {
            "format": "bbeh-encoder-token-audit-v1",
            "role": role,
            "source_plan_sha256": core.sha(args.source / "source-plan.json"),
            "inputs_sha256": core.sha(source),
            "model": core.MODEL,
            "hf_revision": core.REVISION,
            "instruction": core.INSTRUCTION,
            "template": TEMPLATE,
            "max_length": MAX_LENGTH,
            "truncation": False,
            "tokenizer_receipt_sha256": core.sha(args.tokenizer_receipt),
            "rows": counts,
            "overflow_ids": [r["id"] for r in counts if r["input_tokens"] > MAX_LENGTH],
            "truncated_ids": [],
            "source_sha256": core.sha(__file__),
        }
        core.write(args.output / role / "token-audit.json", result)


if __name__ == "__main__":
    main()
