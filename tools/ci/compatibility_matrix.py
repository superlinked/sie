"""Generate the per-release compatibility matrix and render its docs page.

Every cell is one routed model profile in one published image lane, with a
status and the reason for it:

- ``tested``: a GPU smoke of this lane's exact image digest served the profile.
- ``declared``: the lane's bundle routes the profile and nothing excludes it,
  but no GPU run backs it.
- ``unsupported``: a named rule excludes it.

Lanes, CUDA runtimes, engine pins and profiles come from the repository at the
release revision; resolved versions and digests come from built-image
metadata; the driver floors come from NVIDIA's published tables.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from sie_server.config.model import _GPU_MEMORY_GB_BY_PROFILE_HINT, ModelConfig
from sie_server.core.loader import load_model_configs

ROOT = Path(__file__).resolve().parents[2]
DRIVER_FLOORS = ROOT / "docs/compatibility/nvidia-driver-floors.json"
SCHEMA = 1
REGISTRY = "ghcr.io/superlinked"
RUST_LANE = "cuda12-sm89"
RUST_BUNDLE = "candle"
CUDA_ONLY_ADAPTER_PREFIXES = (
    "sie_server.adapters.sglang",
    "sie_server.adapters.tensorrt_llm",
)
ENGINE_DISTRIBUTIONS = ("sglang", "tensorrt-llm", "ctranslate2")
STATUSES = ("tested", "declared", "unsupported")
_FROM_NVIDIA_CUDA = re.compile(r"^FROM nvidia/cuda:(?P<version>[0-9.]+)-\S+ AS (?P<stage>\S+)$", re.MULTILINE)
_FROM_STAGE = re.compile(r"^FROM (?P<parent>runtime-\S+) AS (?P<stage>runtime-\S+)$", re.MULTILINE)
_CUDA_VERSION_ARG = re.compile(r"^ARG CUDA_VERSION=(?P<version>[0-9.]+)$", re.MULTILINE)
_COMPUTE_CAP_ARG = re.compile(r"^ARG CUDA_COMPUTE_CAP=(?P<cap>[0-9]+)$", re.MULTILINE)


def load_bundles(root: Path) -> dict[str, dict[str, Any]]:
    bundles = {}
    for path in sorted((root / "packages/sie_server/bundles").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        bundles[str(data.get("name", path.stem))] = data
    # A composite declares no deps of its own; its engines are its components'.
    for data in bundles.values():
        if data.get("components") and "deps" not in data:
            deps: dict[str, Any] = {}
            for component in data["components"]:
                deps.update(bundles.get(component, {}).get("deps") or {})
            data["deps"] = deps
    return bundles


def release_targets(root: Path) -> list[tuple[str, str]]:
    data = json.loads((root / ".github/release-matrix.json").read_text(encoding="utf-8"))
    targets = [(platform, bundle) for platform in data["platforms"] for bundle in data["bundles"]]
    targets.extend((item["platform"], item["bundle"]) for item in data.get("include", []))
    targets.extend((item["platform"], item["bundle"]) for item in data.get("composites", []))
    return targets


def cuda_runtimes(root: Path) -> dict[tuple[str, str], str | None]:
    """The CUDA runtime base each server lane's final stage is built ``FROM``."""
    runtimes: dict[tuple[str, str], str | None] = {}
    cuda12 = (root / "packages/sie_server/Dockerfile.cuda12").read_text(encoding="utf-8")
    stages = {match["stage"]: match["version"] for match in _FROM_NVIDIA_CUDA.finditer(cuda12)}
    for match in _FROM_STAGE.finditer(cuda12):
        if match["parent"] in stages:
            stages[match["stage"]] = stages[match["parent"]]
    for stage, version in stages.items():
        if stage.startswith("runtime-"):
            runtimes["cuda12", stage.removeprefix("runtime-")] = version
    cuda13 = (root / "packages/sie_server/Dockerfile.cuda13").read_text(encoding="utf-8")
    cuda13_runtime = {match["stage"]: match["version"] for match in _FROM_NVIDIA_CUDA.finditer(cuda13)}
    runtimes["cuda13", "*"] = cuda13_runtime.get("runtime")
    candle = (root / "packages/sie_server_rust/Dockerfile.candle").read_text(encoding="utf-8")
    match = _CUDA_VERSION_ARG.search(candle)
    runtimes["cuda12", RUST_BUNDLE] = match["version"] if match else None
    return runtimes


def driver_floor(floors: dict[str, Any], cuda_runtime: str | None) -> dict[str, Any] | None:
    if cuda_runtime is None:
        return None
    major = cuda_runtime.split(".", 1)[0]
    for floor in floors["floors"]:
        if floor["cuda_major"] == major:
            return floor
    msg = f"no NVIDIA driver floor recorded for CUDA {major}"
    raise ValueError(msg)


def _exact_pin(spec: object) -> str | None:
    if isinstance(spec, dict):
        spec = spec.get("version")
    if isinstance(spec, str) and spec.startswith("=="):
        return spec.removeprefix("==")
    return None


def bundle_engine(bundle: dict[str, Any]) -> tuple[str, str | None]:
    """The engine a bundle serves with, and its exact pin when the bundle declares one."""
    deps = bundle.get("deps") or {}
    for name in ENGINE_DISTRIBUTIONS:
        if name in deps:
            return name, _exact_pin(deps[name])
    if bundle.get("engine"):
        return str(bundle["engine"]), None
    return "transformers", _exact_pin(deps.get("transformers"))


def gpu_class(profile: str) -> str | None:
    lower = profile.lower()
    for hint, _memory_gb in _GPU_MEMORY_GB_BY_PROFILE_HINT:
        if hint in lower:
            return hint
    return None


def _route_parts(route: str, config: ModelConfig) -> tuple[str, str]:
    source = config.synthetic_profile_variant_source
    if source is not None:
        return source
    return route, "default"


def lane_record(
    *,
    lane: str,
    image: str,
    platform: str,
    bundle_name: str,
    bundle: dict[str, Any],
    cuda_runtime: str | None,
    floors: dict[str, Any],
    metadata: dict[str, Any],
    compute_capability: str | None = None,
) -> dict[str, Any]:
    engine, pinned = bundle_engine(bundle)
    observed = metadata.get("distributions") or {}
    observed_engine = observed.get(engine)
    floor = driver_floor(floors, cuda_runtime)
    return {
        "lane": lane,
        "image": image,
        "digest": metadata.get("digest"),
        "platform": platform,
        "bundle": bundle_name,
        "engine": engine,
        "engine_version": observed_engine or pinned,
        "engine_version_source": "image" if observed_engine else ("bundle-pin" if pinned else None),
        "distributions": dict(sorted(observed.items())),
        "cuda_runtime": cuda_runtime,
        "driver_min": floor["driver_min"] if floor else None,
        "compute_capability": compute_capability,
    }


def lanes(root: Path, version: str, floors: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    bundles = load_bundles(root)
    runtimes = cuda_runtimes(root)
    records = []
    for platform, bundle_name in release_targets(root):
        lane = f"{platform}-{bundle_name}"
        if platform == "cpu":
            runtime = None
        elif platform == "cuda13":
            runtime = runtimes["cuda13", "*"]
        else:
            runtime = runtimes.get((platform, bundle_name), runtimes.get((platform, "default")))
        records.append(
            lane_record(
                lane=lane,
                image=f"{REGISTRY}/sie-server:v{version}-{lane}",
                platform=platform,
                bundle_name=bundle_name,
                bundle=bundles[bundle_name],
                cuda_runtime=runtime,
                floors=floors,
                metadata=metadata.get(lane, {}),
            )
        )
    candle = (root / "packages/sie_server_rust/Dockerfile.candle").read_text(encoding="utf-8")
    cap = _COMPUTE_CAP_ARG.search(candle)
    records.append(
        lane_record(
            lane=RUST_LANE,
            image=f"{REGISTRY}/sie-server-rust:v{version}-{RUST_LANE}",
            platform="cuda12",
            bundle_name=RUST_BUNDLE,
            bundle=bundles[RUST_BUNDLE],
            cuda_runtime=runtimes["cuda12", RUST_BUNDLE],
            floors=floors,
            metadata=metadata.get(RUST_LANE, {}),
            compute_capability=f"{cap['cap'][:-1]}.{cap['cap'][-1]}" if cap else None,
        )
    )
    return records


Runs = dict[tuple[str, str], list[dict[str, Any]]]


def _runs_by_route(tested: dict[str, Any], *, failed: bool) -> Runs:
    runs: Runs = {}
    for run in tested.get("runs", []):
        if (run.get("result") == "failed" and bool(run.get("conclusive"))) if failed else run.get("result") == "passed":
            runs.setdefault((run["lane"], run["route"]), []).append(run)
    return runs


def _evidence(run: dict[str, Any]) -> dict[str, Any]:
    keys = ("gpu", "gpu_count", "driver", "cuda_driver_api", "date", "test", "image")
    return {key: run[key] for key in keys if key in run}


def _lane_run(runs: list[dict[str, Any]] | None, lane: dict[str, Any]) -> dict[str, Any] | None:
    """The first smoke run of this lane's exact recorded image digest, if any."""
    digest = lane["digest"]
    if digest is None:
        return None
    return next((run for run in runs or [] if run.get("image", "").endswith(f"@{digest}")), None)


def cell(
    lane: dict[str, Any] | None,
    route: str,
    config: ModelConfig,
    *,
    bundle: str,
    passed: Runs,
    failed: Runs,
) -> dict[str, Any]:
    model, profile = _route_parts(route, config)
    resolved = config.resolve_profile("default")
    adapter = resolved.adapter_path.partition(":")[0]
    record: dict[str, Any] = {
        "lane": lane["lane"] if lane else None,
        "bundle": bundle,
        "model": model,
        "profile": profile,
        "variant_id": route,
        "gpu_class": gpu_class(profile) if lane is None or lane["platform"] != "cpu" else "cpu",
        "tp_width": int(resolved.loadtime.get("tensor_parallel_size") or 1),
    }
    if lane is None:
        return {**record, "status": "unsupported", "reason": f"no published image serves the {bundle} bundle"}
    if lane["platform"] == "cpu" and adapter.startswith(CUDA_ONLY_ADAPTER_PREFIXES):
        return {**record, "status": "unsupported", "reason": f"the {lane['engine']} engine serves on CUDA only"}
    key = (lane["lane"], route)
    failure = _lane_run(failed.get(key), lane)
    if failure is not None:
        return {
            **record,
            "status": "unsupported",
            "reason": f"GPU smoke failed on {failure['gpu']} on {failure['date']}: "
            f"{failure.get('cause') or failure['detail']}",
            "evidence": _evidence(failure),
        }
    success = _lane_run(passed.get(key), lane)
    if success is not None:
        return {
            **record,
            "status": "tested",
            "reason": f"GPU smoke passed on {success['gpu']} (driver {success['driver']}) on {success['date']}",
            "evidence": _evidence(success),
        }
    reason = "routed to this lane's bundle; no smoke test of this image backs it"
    return {**record, "status": "declared", "reason": reason}


def build(
    root: Path,
    version: str,
    *,
    floors: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    tested: dict[str, Any] | None = None,
    source_revision: str | None = None,
) -> dict[str, Any]:
    lane_records = lanes(root, version, floors, metadata or {})
    bundles = load_bundles(root)
    configs = load_model_configs(root / "packages/sie_server/models")
    passed = _runs_by_route(tested or {}, failed=False)
    failed = _runs_by_route(tested or {}, failed=True)
    cells = []
    for route, config in sorted(configs.items()):
        module = config.resolve_profile("default").adapter_path.partition(":")[0]
        owners = sorted(name for name, data in bundles.items() if module in (data.get("adapters") or []))
        for bundle in owners:
            served = [lane for lane in lane_records if lane["bundle"] == bundle]
            for lane in served or [None]:
                cells.append(cell(lane, route, config, bundle=bundle, passed=passed, failed=failed))
    counts = Counter(entry["status"] for entry in cells)
    return {
        "schema": SCHEMA,
        "release": version,
        "source_revision": source_revision,
        "driver_floors": floors,
        "lanes": lane_records,
        "cells": cells,
        "summary": {status: counts.get(status, 0) for status in STATUSES},
    }


def diff(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Cells whose status changed between two matrices, plus cells that appeared or went away."""

    def index(matrix: dict[str, Any]) -> dict[tuple[str | None, str], str]:
        return {(entry["lane"], entry["variant_id"]): entry["status"] for entry in matrix["cells"]}

    before, after = index(old), index(new)
    lines = []
    for key in sorted(before.keys() | after.keys(), key=lambda item: (item[0] or "", item[1])):
        lane, route = key
        if before.get(key) != after.get(key):
            lines.append(f"{lane or '-'} {route}: {before.get(key, 'absent')} -> {after.get(key, 'absent')}")
    old_lanes = {lane["lane"]: lane for lane in old["lanes"]}
    for lane in new["lanes"]:
        previous = old_lanes.get(lane["lane"], {})
        for field in ("engine_version", "cuda_runtime", "driver_min"):
            if previous.get(field) != lane.get(field):
                lines.append(f"{lane['lane']} {field}: {previous.get(field)} -> {lane.get(field)}")
    return lines


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    def escape(value: str) -> str:
        return value.replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    lines.extend("| " + " | ".join(escape(value) for value in row) + " |" for row in rows)
    return lines


def render_markdown(matrix: dict[str, Any]) -> str:
    release = matrix["release"]
    summary = matrix["summary"]
    lines = [
        f"# Compatibility matrix: SIE v{release}",
        "",
        "<!-- Generated by tools/ci/compatibility_matrix.py from "
        f"docs/compatibility/sie-compatibility-v{release}.json. Do not edit by hand. -->",
        "",
        "Each cell is one model profile in one published image lane. A `tested` cell passed a GPU smoke test "
        "of that lane's exact image digest. A `declared` cell is routed to the lane and nothing excludes it, "
        "but no smoke test of that image backs it. An `unsupported` cell is excluded by the rule its reason names.",
        "",
        f"Cells: {summary['tested']} tested, {summary['declared']} declared, {summary['unsupported']} unsupported.",
        "",
        "## NVIDIA driver floors",
        "",
        matrix["driver_floors"]["note"],
        "",
    ]
    lines.extend(
        _table(
            ["CUDA", "Minimum Linux driver", "Source"],
            [
                [f"{floor['cuda_major']}.x", f">= {floor['driver_min']}", f"[NVIDIA]({floor['source']})"]
                for floor in matrix["driver_floors"]["floors"]
            ],
        )
    )
    lines.extend(["", "## Image lanes", ""])
    lines.extend(
        _table(
            ["Lane", "Image", "CUDA runtime", "Minimum driver", "Engine", "Digest"],
            [
                [
                    f"`{lane['lane']}`",
                    f"`{lane['image']}`",
                    lane["cuda_runtime"] or "none (CPU)",
                    f">= {lane['driver_min']}" if lane["driver_min"] else "n/a",
                    f"{lane['engine']} {lane['engine_version'] or '(unpinned)'}"
                    + (f", compute capability {lane['compute_capability']} only" if lane["compute_capability"] else ""),
                    f"`{lane['digest']}`" if lane["digest"] else "not recorded",
                ]
                for lane in matrix["lanes"]
            ],
        )
    )
    by_lane: dict[str | None, list[dict[str, Any]]] = {}
    for entry in matrix["cells"]:
        by_lane.setdefault(entry["lane"], []).append(entry)
    for lane in [*(lane["lane"] for lane in matrix["lanes"]), None]:
        entries = by_lane.get(lane)
        if not entries:
            continue
        counts = Counter(entry["status"] for entry in entries)
        title = f"`{lane}`" if lane else "Bundles without a published image"
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                ", ".join(f"{counts.get(status, 0)} {status}" for status in STATUSES) + ".",
                "",
            ]
        )
        lines.extend(
            _table(
                ["Model", "Profile", "GPU class", "TP", "Status", "Reason"],
                [
                    [
                        entry["model"],
                        entry["profile"],
                        entry["gpu_class"] or "unspecified",
                        str(entry["tp_width"]),
                        entry["status"],
                        entry["reason"],
                    ]
                    for entry in entries
                ],
            )
        )
    return "\n".join(lines) + "\n"


def _read_json(path: Path | None) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="write the matrix JSON for one release")
    generate.add_argument("--version", required=True)
    generate.add_argument("--root", type=Path, default=ROOT, help="repository checkout at the release revision")
    generate.add_argument("--source-revision")
    generate.add_argument("--driver-floors", type=Path, default=DRIVER_FLOORS)
    generate.add_argument(
        "--evidence", type=Path, help="per-lane image digests and resolved versions, and GPU smoke runs"
    )
    generate.add_argument("--output", type=Path)
    markdown = commands.add_parser("markdown", help="render a matrix JSON as the docs page")
    markdown.add_argument("matrix", type=Path)
    markdown.add_argument("--output", type=Path)
    compare = commands.add_parser("diff", help="list cells and lanes that changed between two matrices")
    compare.add_argument("old", type=Path)
    compare.add_argument("new", type=Path)
    args = parser.parse_args()

    if args.command == "generate":
        evidence = _read_json(args.evidence) or {}
        matrix = build(
            args.root,
            args.version,
            floors=json.loads(args.driver_floors.read_text(encoding="utf-8")),
            metadata=evidence.get("images"),
            tested=evidence,
            source_revision=args.source_revision,
        )
        text = json.dumps(matrix, indent=1) + "\n"
    elif args.command == "markdown":
        text = render_markdown(json.loads(args.matrix.read_text(encoding="utf-8")))
    else:
        changes = diff(
            json.loads(args.old.read_text(encoding="utf-8")), json.loads(args.new.read_text(encoding="utf-8"))
        )
        text = "".join(f"{line}\n" for line in changes)
    if getattr(args, "output", None):
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
