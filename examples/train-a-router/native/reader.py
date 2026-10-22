"""Apply both exported Laya checkpoints on CPU or CUDA without reading labels."""

import argparse
import json
from pathlib import Path
import time

import bbeh_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-file", type=Path)
    parser.add_argument("--variant", choices=("base", "lora"))
    args = parser.parse_args()
    if args.question_file:
        core = bbeh_native.core
        core.require(args.variant is not None, "Select base or lora")
        plan, _ = core.checked_plan(args.run)
        trained = core.read(args.run / "native/trained.json")
        core.require(trained["plan_sha256"] == core.sha(args.run / "plan.json")
                     and trained["training_ids"] == plan["populations"]["train"] + plan["populations"]["development"],
                     "Native checkpoint belongs to another training plan")
        checkpoint = args.run / "native" / args.variant
        core.require(bbeh_native.checkpoint_files(checkpoint) == trained["checkpoints"][args.variant]["files_sha256"],
                     "Native checkpoint changed")
        item = core.read(args.question_file)
        started = time.perf_counter()
        p = bbeh_native.checkpoint_probabilities(checkpoint, [item], plan["pair"], plan["native"], device=args.device)
        result = {"probabilities": p.tolist(), "expected_usd": core.expected_usd(p, trained["conditional_cost_usd"]).tolist(),
                  "elapsed_seconds_including_load": time.perf_counter() - started, "outcomes_accessed": False}
        core.write(args.output, result)
    else:
        result = bbeh_native.score_native(args.run, args.output, device=args.device)
    print(json.dumps({"output": str(args.output), "outcomes_accessed": result["outcomes_accessed"]}))


if __name__ == "__main__":
    main()
