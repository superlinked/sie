from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

TAXONOMY_CATEGORIES_URL = (
    "https://raw.githubusercontent.com/Shopify/product-taxonomy/"
    "main/dist/en/categories.txt"
)
DEFAULT_OUTPUT = Path("data/shopify-taxonomy-categories.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Shopify taxonomy categories list used by this example."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output categories.txt path.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download even when the output file already exists.",
    )
    return parser.parse_args()


def download_file(output: Path) -> int:
    with urlopen(TAXONOMY_CATEGORIES_URL) as response:
        content = response.read()

    output.write_bytes(content)
    return len(content)


def main() -> None:
    args = parse_args()
    output = args.output

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.refresh:
        print(
            json.dumps(
                {
                    "status": "skipped_existing",
                    "source_url": TAXONOMY_CATEGORIES_URL,
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
                "source_url": TAXONOMY_CATEGORIES_URL,
                "output_path": str(output),
                "file_size_bytes": downloaded_bytes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
