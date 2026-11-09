"""Command-line entry point of the placement tools."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from sie_placement import keda
from sie_placement.schema import Plan

ADMIN_TOKEN_ENV = "SIE_ADMIN_TOKEN"  # noqa: S105 - an environment variable name
SDK_API_KEY_ENV = "SIE_API_KEY"
DEFAULT_INTERVAL_S = 60.0
EXIT_FAILED = 1
EXIT_REFUSED = 2

_TOKEN = re.compile(r"[\x21-\x7e]+")


class _UsageError(ValueError):
    pass


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {text!r}")
    return value


def _integer_at_least(minimum: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {text!r}")
        return value

    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sie-placement", description="Placement planner tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    tune = commands.add_parser(
        "tune",
        help="keep each plan lane's warm floor inside its envelope through gateway floor-carrier pools",
        description=(
            "Read each plan lane's series from Prometheus, decide one tuner step per lane and, with --apply, "
            "upsert one floor-carrier pool per lane through the gateway pool API. Without --apply nothing is written."
        ),
    )
    tune.add_argument("--plan", required=True, type=Path, help="placement-plan document")
    tune.add_argument("--prometheus-url", required=True, help="Prometheus base URL that serves /api/v1/query")
    tune.add_argument("--gateway-url", required=True, help="gateway base URL")
    tune.add_argument(
        "--series-selector",
        default="",
        help="label matchers added to every series selector, for example 'namespace=\"sie\"'",
    )
    tune.add_argument("--apply", action="store_true", help="write the floor-carrier pools")
    tune.add_argument("--once", action="store_true", help="run one tick and exit")
    tune.add_argument(
        "--interval-s", type=_positive_float, default=DEFAULT_INTERVAL_S, help="seconds between ticks (default 60)"
    )
    tune.add_argument(
        "--max-pool-floor",
        type=_integer_at_least(0),
        default=keda.DEFAULT_MAX_POOL_FLOOR,
        help="the gateway's SIE_GATEWAY_POOL_MAX_MINIMUM_WORKER_COUNT (default 4)",
    )
    tune.add_argument(
        "--max-pools",
        type=_integer_at_least(1),
        default=keda.DEFAULT_MAX_POOLS,
        help="the gateway's SIE_GATEWAY_MAX_POOLS (default 64)",
    )
    tune.add_argument("--api-key-file", type=Path, help="file holding a gateway client token for pool reads")
    tune.add_argument(
        "--admin-token-file",
        type=Path,
        help=f"file holding the gateway admin token for pool writes; read only with --apply (default ${ADMIN_TOKEN_ENV})",
    )
    return parser


def _token(value: str, source: str, label: str) -> str:
    if _TOKEN.fullmatch(value) is None:
        raise _UsageError(f"the {label} in {source} must be one run of printable ASCII characters without spaces")
    return value


def _secret(path: Path | None, env_name: str | None, label: str) -> str | None:
    if path is not None:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as error:
            reason = error.strerror if isinstance(error, OSError) else "it is not UTF-8 text"
            raise _UsageError(f"cannot read the {label} file {str(path)!r}: {reason}") from None
        if not value:
            raise _UsageError(f"the {label} file {str(path)!r} is empty")
        return _token(value, f"the file {str(path)!r}", label)
    if env_name is None:
        return None
    value = os.environ.get(env_name, "").strip()
    return _token(value, f"${env_name}", label) if value else None


def _load_plan(path: Path) -> Plan:
    try:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
    except OSError as error:
        raise _UsageError(f"cannot read the plan {str(path)!r}: {error.strerror}") from None
    except ValueError as error:
        raise _UsageError(f"the plan {str(path)!r} is not JSON: {error}") from None
    except RecursionError:
        raise _UsageError(f"the plan {str(path)!r} is not JSON: it nests deeper than the parser allows") from None
    return Plan.from_dict(document)


def _emit(result: keda.Tick) -> None:
    for row in result.audit_rows():
        print(json.dumps(row), flush=True)
    for error in result.series_errors:
        print(f"sie-placement tune: Prometheus series {error}", file=sys.stderr, flush=True)


def _tune(args: argparse.Namespace) -> int:
    try:
        carriers = keda.carriers(_load_plan(args.plan), max_pool_floor=args.max_pool_floor, max_pools=args.max_pools)
        keda.series_queries(args.series_selector)
        prometheus = keda.PrometheusHttp(args.prometheus_url)
        keda.require_base_url("gateway", args.gateway_url)
        read_token = _secret(args.api_key_file, None, "gateway client token")
        if read_token is None:
            _secret(None, SDK_API_KEY_ENV, "gateway client token")
        admin_token = _secret(args.admin_token_file, ADMIN_TOKEN_ENV, "gateway admin token") if args.apply else None
    except ValueError as error:
        print(f"sie-placement tune: {error}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        from sie_placement import pools  # noqa: PLC0415
    except ImportError:
        print("sie-placement tune needs the sie-placement keda extra, which installs the SIE SDK", file=sys.stderr)
        return EXIT_REFUSED
    if args.apply and admin_token is None:
        print(
            f"sie-placement tune: no gateway admin token in --admin-token-file or ${ADMIN_TOKEN_ENV}; "
            "pool writes carry no credential",
            file=sys.stderr,
        )
    try:
        reader = pools.connect(args.gateway_url, read_token)
        writer = pools.connect(args.gateway_url, admin_token or "") if args.apply else None
    except ValueError as error:
        print(f"sie-placement tune: {error}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        while True:
            started = time.monotonic()
            result = keda.tick(
                carriers, prometheus=prometheus, reader=reader, writer=writer, selector=args.series_selector
            )
            _emit(result)
            if args.once:
                return EXIT_FAILED if result.failed else 0
            time.sleep(max(0.0, args.interval_s - (time.monotonic() - started)))
    except KeyboardInterrupt:
        return 0
    finally:
        reader.close()
        if writer is not None:
            writer.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return _tune(args)
