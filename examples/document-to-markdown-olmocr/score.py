#!/usr/bin/env python3
"""Reproduce the /doc-to-markdown figures from the recorded run. No API key, no network.

    python3 fetch.py
    python3 score.py                     # the recorded run, checked against the page
    uv run --with 'olmocr[bench]==0.4.27' score.py --rescore runs   # your own run.py output

The recorded run: SIE's hosted LightOnOCR-2-1B on all 1,403 pages of Ai2's olmOCR-Bench,
scored with Ai2's unmodified scorer (olmocr 0.4.27). This verifies every file against the
manifest pinned below, recomputes each of the 8 per-file pass rates and Ai2's Overall
(their unweighted mean) from the per-test results, prints them beside the published runs
of AWS Textract and Azure Document Intelligence on the same scorer, and exits non-zero if
any figure superlinked.com/doc-to-markdown publishes does not come out.

`--rescore DIR` instead runs Ai2's scorer on the Markdown in DIR/sie-lightonocr-2-1b (what
run.py writes), over the pages it holds. The default mode is standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"

# The SHA-256 of manifest.json at the dataset revision fetch.py pins. It lives here,
# outside the evidence, because a digest inside a file cannot authenticate that file.
MANIFEST_SHA256 = "dc0cdb0f77d1ca4d3fff6a1b601bffbe2c9421cdc02bf1ccfb77c8a82c8ae6fe"

FILES = (
    "arxiv_math",
    "baseline",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "table_tests",
)
ARMS = {
    "sie-lightonocr-2-1b": "SIE LightOnOCR-2-1B",
    "sie-lightonocr-2-1b-docling-furniture": "SIE LightOnOCR-2-1B, headers and footers removed with docling",
    "sie-docling-markdown": "SIE docling",
}

# What superlinked.com/doc-to-markdown publishes: Overall to one decimal, and $ per 1,000
# pages at each vendor's cheapest public tier (read 2026-09-30).
PAGE = {
    "overall": {"sie-lightonocr-2-1b": "75.3", "azure-di-prebuilt-layout": "48.7", "aws-textract-tables": "40.2"},
    "usd_per_1k": {"sie-lightonocr-2-1b": 2.00, "azure-di-prebuilt-layout": 6.00, "aws-textract-tables": 10.00},
}


def verified() -> dict:
    manifest_bytes = (EVIDENCE / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise SystemExit("evidence/manifest.json is not the recorded run's manifest; run python3 fetch.py")
    manifest = json.loads(manifest_bytes)
    for relative, digest in manifest["files"].items():
        path = EVIDENCE / relative
        if relative.startswith("outputs/") and not path.exists():
            continue  # fetched only with --outputs
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"evidence/{relative} does not match the manifest")
    return manifest


def rates(rows: list[dict]) -> dict[str, tuple[int, int]]:
    out = {}
    for file in FILES:
        mine = [row for row in rows if row["file"] == file]
        out[file] = (sum(row["passed"] for row in mine), len(mine))
    return out


def overall(by_file: dict[str, tuple[int, int]]) -> float:
    return sum(passed / total for passed, total in by_file.values()) / len(by_file)


def recorded() -> int:
    manifest = verified()
    rivals = json.loads((EVIDENCE / "rivals.json").read_text())
    print(f"olmOCR-Bench, {manifest['pages']} pages, {manifest['scorer']}, run {', '.join(manifest['run_dates'])}\n")
    header = f"{'':44}" + "".join(f"{f[:10]:>11}" for f in FILES) + f"{'Overall':>9}"
    print(header)
    problems = []
    results = {}
    for arm, name in ARMS.items():
        rows = [json.loads(line) for line in (EVIDENCE / "per_test" / f"{arm}.jsonl").read_text().splitlines()]
        by_file = rates(rows)
        summary = json.loads((EVIDENCE / "scores" / f"{arm}.json").read_text())
        for file, (passed, total) in by_file.items():
            if (summary["files"][file]["passed"], summary["files"][file]["total"]) != (passed, total):
                problems.append(f"{arm} {file}: per-test {passed}/{total}, scorer {summary['files'][file]}")
        score = overall(by_file)
        if abs(score - summary["overall"]) > 0.0005:
            problems.append(f"{arm}: Overall {score:.4f} per test, {summary['overall']:.4f} from the scorer")
        results[arm] = score
        cells = "".join(f"{100 * p / t:>11.1f}" for p, t in by_file.values())
        print(f"{name[:43]:44}{cells}{100 * score:>9.1f}")
    for row in rivals["systems"]:
        cells = "".join(f"{100 * row['score']['files'][f]:>11.1f}" for f in FILES)
        print(f"{(row['name'] + ' (published)')[:43]:44}{cells}{100 * row['score']['overall']:>9.1f}")
        results[row["id"]] = row["score"]["overall"]

    print("\n$ per 1,000 pages, cheapest public tier:")
    for row in rivals["systems"]:
        print(f"  {row['name']}: ${row['price']['usd_per_1k']:.2f} ({row['price']['tier']})")
        if row["price"]["usd_per_1k"] != PAGE["usd_per_1k"][row["id"]]:
            problems.append(f"{row['id']}: price {row['price']['usd_per_1k']}, page {PAGE['usd_per_1k'][row['id']]}")
    print(f"  SIE LightOnOCR-2-1B: ${PAGE['usd_per_1k']['sie-lightonocr-2-1b']:.2f}")

    for key, published in PAGE["overall"].items():
        if f"{100 * results[key]:.1f}" != published:
            problems.append(f"{key}: Overall {100 * results[key]:.1f} here, {published} on the page")
    if problems:
        print("\nDoes not match superlinked.com/doc-to-markdown:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("\nEvery figure matches superlinked.com/doc-to-markdown.")
    return 0


def rescore(runs: Path) -> int:
    """Ai2's scorer, unmodified, over the pages run.py converted."""
    from huggingface_hub import snapshot_download

    bench = (
        Path(
            snapshot_download(
                "allenai/olmOCR-bench",
                repo_type="dataset",
                revision="54a96a6fb6a2bd3b297e59869491db4d3625b711",
                allow_patterns=["bench_data/*"],
            )
        )
        / "bench_data"
    )
    arm = runs / "sie-lightonocr-2-1b"
    pages = {str(p.relative_to(arm))[: -len("_pg1_repeat1.md")] + ".pdf" for p in arm.rglob("*_pg1_repeat1.md")}
    if not pages:
        raise SystemExit(f"no Markdown under {arm}; run run.py first")
    with tempfile.TemporaryDirectory() as tmp:
        scoring = Path(tmp)
        for pdf in pages:
            (scoring / "pdfs" / pdf).parent.mkdir(parents=True, exist_ok=True)
            (scoring / "pdfs" / pdf).symlink_to(bench / "pdfs" / pdf)
        for jsonl in bench.glob("*.jsonl"):
            kept = [line for line in jsonl.read_text().splitlines() if line and json.loads(line)["pdf"] in pages]
            (scoring / jsonl.name).write_text("\n".join(kept) + ("\n" if kept else ""))
        (scoring / arm.name).symlink_to(arm.resolve())
        return subprocess.run(
            [sys.executable, "-m", "olmocr.bench.benchmark", "--dir", str(scoring)], check=False
        ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rescore", type=Path, metavar="DIR", help="score run.py's output with Ai2's scorer")
    args = parser.parse_args()
    return rescore(args.rescore) if args.rescore else recorded()


if __name__ == "__main__":
    sys.exit(main())
