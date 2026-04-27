"""CLI utility to check SIE server status and available services.

Usage:
    python cli_sie_status.py              # full status report
    python cli_sie_status.py --health     # health check only
    python cli_sie_status.py --models     # list models only
    python cli_sie_status.py --pools      # list pools only
"""

import argparse
import json
import sys

import httpx
from app.config import settings

TIMEOUT = 15.0


def _client() -> httpx.Client:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.sie_api_key:
        headers["Authorization"] = f"Bearer {settings.sie_api_key}"
    return httpx.Client(headers=headers, timeout=TIMEOUT)


def _get(client: httpx.Client, path: str) -> dict | list | str | None:
    url = f"{settings.sie_api_endpoint.rstrip('/')}{path}"
    try:
        r = client.get(url)
        r.raise_for_status()
        try:
            return r.json()
        except (json.JSONDecodeError, ValueError):
            return r.text.strip() or f"(empty — HTTP {r.status_code})"
    except httpx.ConnectError:
        print(f"  ERROR: cannot connect to {url}")
        return None
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} from {url}")
        return None


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_result(data: dict | list | str) -> None:
    if isinstance(data, str):
        print(f"  {data}")
    else:
        print(json.dumps(data, indent=2))


def check_health(client: httpx.Client) -> None:
    section("Health")
    data = _get(client, "/health")
    if data is None:
        return
    _print_result(data)


def check_liveness(client: httpx.Client) -> None:
    section("Liveness  (/healthz)")
    data = _get(client, "/healthz")
    if data is None:
        return
    _print_result(data)


def check_readiness(client: httpx.Client) -> None:
    section("Readiness (/readyz)")
    data = _get(client, "/readyz")
    if data is None:
        return
    _print_result(data)


def check_models(client: httpx.Client) -> None:
    section("Models    (/v1/models)")
    data = _get(client, "/v1/models")
    if data is None:
        return
    if isinstance(data, str):
        print(f"  {data}")
        return
    models = data.get("models") or data.get("data") or []
    if not models:
        print(json.dumps(data, indent=2))
        return

    loaded = [m for m in models if m.get("loaded")]
    print(f"  {len(models)} model(s) registered, {len(loaded)} loaded\n")
    for m in models:
        name = m.get("name") or m.get("id") or "?"
        workers = m.get("worker_count", 0)
        state = "LOADED" if m.get("loaded") else "not loaded"
        bundles = ", ".join(m.get("bundles", []))
        parts = [f"  - {name}", f"[{state}, {workers} worker(s)]"]
        if bundles:
            parts.append(f"bundle={bundles}")
        print("  ".join(parts))


def check_pools(client: httpx.Client) -> None:
    section("Pools     (/v1/pools)")
    data = _get(client, "/v1/pools")
    if data is None:
        return
    if isinstance(data, dict) and not data:
        print("  No pools configured")
        return
    print(json.dumps(data, indent=2))


def full_report(client: httpx.Client) -> None:
    endpoint = settings.sie_api_endpoint
    print(f"SIE Server: {endpoint}")
    check_liveness(client)
    check_readiness(client)
    check_health(client)
    check_models(client)
    check_pools(client)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SIE server status")
    parser.add_argument("--health", action="store_true", help="Health check only")
    parser.add_argument("--models", action="store_true", help="List models only")
    parser.add_argument("--pools", action="store_true", help="List pools only")
    args = parser.parse_args()

    with _client() as client:
        if not (args.health or args.models or args.pools):
            full_report(client)
            return

        print(f"SIE Server: {settings.sie_api_endpoint}")
        if args.health:
            check_health(client)
        if args.models:
            check_models(client)
        if args.pools:
            check_pools(client)
        print()


if __name__ == "__main__":
    main()
