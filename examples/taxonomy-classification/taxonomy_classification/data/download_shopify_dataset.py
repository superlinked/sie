from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

DATASET_URL = (
    "https://huggingface.co/datasets/Shopify/product-catalogue/resolve/"
    "main/data/train-00000-of-00015.parquet"
)
DEFAULT_OUTPUT = Path("data/train-00000-of-00015.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the fixed Shopify dataset shard used by this example."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output parquet path.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download even when the output file already exists.",
    )
    return parser.parse_args()


def download_file(output: Path) -> int:
    temporary_path: Path | None = None

    try:
        with urlopen(DATASET_URL) as response:
            with NamedTemporaryFile(
                dir=output.parent,
                prefix=f"{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                shutil.copyfileobj(response, temporary_file)
        temporary_path.replace(output)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return output.stat().st_size


def main() -> None:
    args = parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.refresh:
        print(
            json.dumps(
                {
                    "status": "skipped_existing",
                    "source_url": DATASET_URL,
                    "output_path": str(output),
                    "file_size_bytes": output.stat().st_size,
                },
                indent=2,
            )
        )
        return

    downloaded_bytes = download_file(output)
    print(
        json.dumps(
            {
                "status": "downloaded",
                "source_url": DATASET_URL,
                "output_path": str(output),
                "file_size_bytes": downloaded_bytes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
