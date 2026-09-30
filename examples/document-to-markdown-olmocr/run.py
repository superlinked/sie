#!/usr/bin/env python3
"""Convert olmOCR-Bench pages to Markdown with SIE's LightOnOCR-2-1B.

    export SIE_API_KEY=sk-sie-...
    uv run run.py --pages 20        # 20 pages spread over the benchmark's folders, about $0.04
    uv run run.py --all             # all 1,403 pages, about $2.81

Each page is rendered to a PNG, longest side 1,540 px (the model's native size), and
sent as one `extract` call. The Markdown lands in runs/sie-lightonocr-2-1b/ in the file
layout Ai2's scorer reads, so `score.py --rescore runs` scores it the way the recorded
run was scored. The PDFs come from Ai2's dataset at the revision the recorded run used.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
MODEL = "lightonai/LightOnOCR-2-1B"
BENCH_REPO = "allenai/olmOCR-bench"
BENCH_REVISION = "54a96a6fb6a2bd3b297e59869491db4d3625b711"
ARM = "sie-lightonocr-2-1b"

# PDFium is not thread-safe; renders run one at a time, requests in parallel.
_render_lock = threading.Lock()


def bench_dir() -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        BENCH_REPO,
        repo_type="dataset",
        revision=BENCH_REVISION,
        allow_patterns=["bench_data/*"],
    )
    return Path(path) / "bench_data"


def all_pdfs(bench: Path) -> list[str]:
    return sorted(str(p.relative_to(bench / "pdfs")) for p in (bench / "pdfs").rglob("*.pdf"))


def spread(pdfs: list[str], n: int) -> list[str]:
    """n pages, evenly spaced over the sorted list, so every folder is represented."""
    if n >= len(pdfs):
        return pdfs
    return [pdfs[(i * (len(pdfs) - 1)) // max(1, n - 1)] for i in range(n)]


def render_png(pdf: Path, max_dim: int = 1540) -> bytes:
    import pypdfium2 as pdfium

    with _render_lock:
        doc = pdfium.PdfDocument(str(pdf))
        try:
            page = doc[0]
            width, height = page.get_size()
            scale = min(max_dim / width, max_dim / height)
            image = page.render(scale=scale).to_pil().convert("RGB")
        finally:
            doc.close()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pages", type=int, help="convert this many pages, spread over the benchmark")
    group.add_argument("--all", action="store_true", help="convert all 1,403 pages")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--base-url", default=os.environ.get("SIE_BASE_URL", "https://api.superlinked.com"))
    args = parser.parse_args()

    key = os.environ.get("SIE_API_KEY")
    if not key:
        print("Set SIE_API_KEY to an SIE Cloud key.", file=sys.stderr)
        return 2
    from sie_sdk import SIEClient

    client = SIEClient(args.base_url, api_key=key, timeout_s=900)
    bench = bench_dir()
    pdfs = all_pdfs(bench) if args.all else spread(all_pdfs(bench), args.pages)

    def convert(pdf_rel: str) -> tuple[str, float, str | None]:
        target = RUNS / ARM / f"{pdf_rel[:-4]}_pg1_repeat1.md"
        if target.exists():
            return pdf_rel, 0.0, None
        target.parent.mkdir(parents=True, exist_ok=True)
        image = {"data": render_png(bench / "pdfs" / pdf_rel), "format": "png"}
        started = time.perf_counter()
        try:
            result = client.extract(MODEL, {"images": [image]})
            target.write_text(result["entities"][0]["text"])
            return pdf_rel, time.perf_counter() - started, None
        except Exception as error:  # noqa: BLE001 - reported per page
            return pdf_rel, time.perf_counter() - started, f"{type(error).__name__}: {error}"

    failed = 0
    with ThreadPoolExecutor(args.concurrency) as pool:
        for future in as_completed([pool.submit(convert, pdf) for pdf in pdfs]):
            pdf_rel, seconds, error = future.result()
            failed += error is not None
            print(f"  {pdf_rel}: {'FAILED ' + error if error else f'{seconds:.1f} s'}")
    print(f"{len(pdfs) - failed} of {len(pdfs)} pages converted into {RUNS / ARM}")
    if failed:
        print("A failed page has no file; the scorer counts it as failing every test.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
