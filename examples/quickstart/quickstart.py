# ruff: noqa: INP001

import os

from sie_sdk import SIEClient


def main() -> None:
    with SIEClient(
        os.getenv("SIE_CLUSTER_URL", "http://localhost:8080"),
        api_key=os.getenv("SIE_API_KEY"),
    ) as client:
        result = client.encode(
            os.getenv("SIE_MODEL", "BAAI/bge-m3"),
            {"text": "Embeddings make meaning searchable."},
        )

    print(f"Created a {len(result['dense'])}-dimensional embedding.")


if __name__ == "__main__":
    main()
