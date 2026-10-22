"""Acquire a fresh question-only Qwen feature set and singleton fee validation."""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path

from embedding import collect_features, recipes as core, selfhost, token_audit, wire
from reader_io import ChangedFileSync, require, sha


def prepare(source, budget, seconds):
    cap = Decimal(str(budget))
    require(cap.is_finite() and cap > 0 and type(seconds) is int and seconds > 0, "Invalid budget/deadline")
    source = Path(source)
    source_hash = sha(source / "source-plan.json")
    rows, input_hashes = collect_features.source_rows(source, source_hash)
    return {
        "format": "bbeh-public-feature-acquisition-v1", "reproduction": True,
        "source_plan_sha256": source_hash, "n": len(rows), "role_inputs_sha256": input_hashes,
        "model": core.MODEL, "hf_revision": core.REVISION, "instruction": core.INSTRUCTION,
        "source_sha256": {"encode": sha(__file__), "collector": sha(collect_features.__file__),
                          "wire": sha(wire.__file__), "token_audit": sha(token_audit.__file__)},
        "budget_usd": str(cap), "collection_seconds": seconds,
        "singleton_validation_including_warmup": len(collect_features.singleton_rows(rows)),
        "source_only": True, "automatic_redispatch": False,
    }


def execute(source, output, plan):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    require(not Path(output).exists(), "Preserve earlier feature receipts; do not repeat unresolved calls")
    local = plan["format"] == "bbeh-public-selfhost-feature-plan-v1"
    require(local or bool(os.environ.get("SIE_API_KEY")), "Set SIE_API_KEY for the SIE Cloud encoder")
    root = Path(snapshot_download(core.MODEL, revision=core.REVISION, token=False,
                                 allow_patterns=["*.json", "*.txt", "*.model"]))
    binding = {"model": core.MODEL, "hf_revision": core.REVISION,
               "files_sha256": {p.relative_to(root).as_posix(): sha(p) for p in root.rglob("*") if p.is_file()}}
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if local:
        return selfhost.run(source, output, plan, tokenizer, binding)
    return collect_features.run(source, output, plan, tokenizer, binding, ChangedFileSync())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-usd", help="Required for Cloud; self-hosted execution remains unpriced")
    parser.add_argument("--self-hosted", action="store_true")
    parser.add_argument("--sie-base-url", help="Explicit self-hosted server origin")
    parser.add_argument("--collection-seconds", type=int, default=4500)
    parser.add_argument("--execute", action="store_true", help="Dispatch SIE requests; default only prepares the plan")
    args = parser.parse_args()
    if args.self_hosted:
        require(args.sie_base_url is not None and args.budget_usd is None,
                "Self-hosted encoding requires --sie-base-url; a Cloud USD cap does not price local hosting")
        plan = selfhost.prepare(args.source, args.sie_base_url, args.collection_seconds)
    else:
        require(args.sie_base_url is None and args.budget_usd is not None,
                "Cloud encoding requires --budget-usd; use --self-hosted for a custom SIE origin")
        plan = prepare(args.source, args.budget_usd, args.collection_seconds)
    result = execute(args.source, args.output, plan) if args.execute else plan
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
