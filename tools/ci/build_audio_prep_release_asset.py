#!/usr/bin/env python3
"""Build and stage the exact native audio wheel used as a release asset."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_build_wheel(project_root: Path) -> ModuleType:
    build_wheel_path = project_root / "packages/sie_audio_prep/build_wheel.py"
    spec = importlib.util.spec_from_file_location("sie_audio_prep_build_wheel", build_wheel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {build_wheel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sie_audio_prep_build_wheel", module)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    build_wheel = load_build_wheel(project_root)
    wheel = build_wheel.build_audio_prep_wheel(project_root, required=True)
    if wheel is None:
        raise RuntimeError("required audio wheel build returned no artifact")
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / wheel.name
    shutil.copyfile(wheel, destination)
    build_wheel._validate_wheel(destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
