from __future__ import annotations

import json
from pathlib import Path

import httpx

SERVER_URL = "http://localhost:8800"
SAMPLE_PATH = Path(__file__).with_name("sample_pages.json")


def main() -> None:
    pages = json.loads(SAMPLE_PATH.read_text())
    with httpx.Client(timeout=60.0) as client:
        for page in pages:
            response = client.post(f"{SERVER_URL}/ingest", json=page)
            response.raise_for_status()
            print(f"seeded: {page['title']}")


if __name__ == "__main__":
    main()
