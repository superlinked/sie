"""Run the ordinary SIE CLI with the dedicated child stream observer."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from sie_server.adapters.sglang import cuda13, generation
from sie_server.api import openai_local
from sie_server.cli import app

from tools.benchmarks.stream_decode.observer import SOURCE_FILES, SOURCE_SHA256, TRACE_ENV, VERSION, clock_domain, save


@contextlib.contextmanager
def observe(output: Path, cohort: str) -> Iterator[None]:
    if re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cohort) is None:
        raise ValueError("cohort must contain 1-64 letters, digits, underscores or hyphens")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    manifest = {
        "schema": 2,
        "cohort": cohort,
        "clock_domain": clock_domain(),
        "sglang_version": VERSION,
        "tokenizer_manager_sha256": SOURCE_SHA256,
        "engine_sources": SOURCE_FILES,
        "sie_sources": {
            module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (generation, cuda13, openai_local)
        },
        "observer_sha256": hashlib.sha256(source.with_name("observer.py").read_bytes()).hexdigest(),
    }
    save(output / "server.json", manifest)
    adapter = cast("Any", generation.SGLangGenerationAdapter)
    original = adapter._compat_pythonpath_entries
    previous = os.environ.get(TRACE_ENV)

    def paths(self: generation.SGLangGenerationAdapter) -> tuple[str, ...]:
        return (str(source.parent / "_hooks"), str(source.parents[3]), *original(self))

    adapter._compat_pythonpath_entries = paths
    os.environ[TRACE_ENV] = str(output)
    try:
        yield
    finally:
        adapter._compat_pythonpath_entries = original
        if previous is None:
            os.environ.pop(TRACE_ENV, None)
        else:
            os.environ[TRACE_ENV] = previous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-directory", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("sie_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.sie_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if any(arg.split("=")[0] == "--reload" for arg in arguments):
        raise ValueError("benchmark wrapper does not support server reload")
    sys.argv = ["sie-server", *arguments]
    with observe(args.trace_directory, args.cohort):
        app()


if __name__ == "__main__":
    main()
