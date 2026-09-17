"""Every engine flag the SGLang adapters emit must be one the engine declares.

Two failures motivated this. ``--tp`` was emitted for years and is not a
declared option at all: it worked only through argparse prefix matching, so any
future upstream flag sharing that prefix would have turned every SIE launch
into an ambiguous-option error. And a sweep of the pinned engine found a flag
emitted that the engine does not declare in any form.

The list below is the contract. It is deliberately explicit rather than derived
at run time, because the engine is only importable inside a GPU bundle and this
check has to run in an ordinary unit suite. Adding a flag means adding it here,
which is the moment to confirm the engine declares it in **every** pinned
version that loads the adapter emitting it.

Verified against both pinned engines by reading ``sglang/srt/server_args.py``:
``0.5.10.post1`` (the ``sglang`` and ``sglang-embedding`` bundles) from the
published sdist, 364 declared flags, and ``0.5.13`` (the ``sglang-cu130``
bundle) from the installed package inside a GPU container, 417 declared flags.
Neither declares ``--tp``. No flag recorded below is a strict prefix of another
declared flag in either version.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ADAPTER_DIR = Path(__file__).resolve().parents[2] / "src/sie_server/adapters/sglang"

# Flags confirmed present in sglang 0.5.10.post1's declared argument set.
_DECLARED_IN_PINNED_ENGINE = frozenset(
    {
        "--attention-backend",
        "--context-length",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--dtype",
        "--enable-lora",
        "--grammar-backend",
        "--is-embedding",
        "--log-level",
        "--lora-paths",
        "--mamba-scheduler-strategy",
        "--max-loras-per-batch",
        "--mem-fraction-static",
        "--model-path",
        "--nccl-port",
        "--port",
        "--reasoning-parser",
        "--revision",
        "--served-model-name",
        "--speculative-algorithm",
        "--speculative-draft-model-path",
        "--speculative-draft-model-revision",
        "--speculative-eagle-topk",
        "--speculative-num-draft-tokens",
        "--speculative-num-steps",
        "--tensor-parallel-size",
        "--tool-call-parser",
        "--trust-remote-code",
        "--watchdog-timeout",
    }
)

# Emitted only by adapters whose bundle pins the newer engine. Absent from
# 0.5.10.post1 and present in 0.5.13, which is the bundle that loads them.
_NEWER_ENGINE_ONLY = frozenset(
    {
        "--enable-strict-thinking",  # cuda13.py, sglang-cu130 bundle
    }
)

# Nothing currently. A flag belongs here only while it is emitted and known not
# to be declared, which should be never: the previous occupant,
# ``--pooling-method``, was removed once the sweep showed neither pinned engine
# declares it in any form.
_KNOWN_UNDECLARED: frozenset[str] = frozenset()

_FLAG_LITERAL = re.compile(r'"(--[a-z0-9][a-z0-9-]*)"')


def _emitted_flags() -> dict[str, set[str]]:
    """Return every ``--flag`` literal in the adapter sources, by file."""
    found: dict[str, set[str]] = {}
    for source in sorted(_ADAPTER_DIR.rglob("*.py")):
        for match in _FLAG_LITERAL.finditer(source.read_text(encoding="utf-8")):
            found.setdefault(match.group(1), set()).add(source.name)
    return found


def test_no_emitted_flag_is_outside_the_verified_contract() -> None:
    """A new flag must be added to this module, which is where it gets checked."""
    known = _DECLARED_IN_PINNED_ENGINE | _NEWER_ENGINE_ONLY | _KNOWN_UNDECLARED
    emitted = _emitted_flags()
    # "--tp" survives as comment text explaining why it is not emitted.
    unexpected = {flag: sorted(files) for flag, files in emitted.items() if flag not in known and flag != "--tp"}

    assert not unexpected, (
        f"adapter sources carry engine flag literal(s) not recorded in this module: {unexpected}. "
        "Confirm the engine declares each one in every pinned version that loads the emitting "
        "adapter, then add it to the matching set here."
    )


def test_the_width_flag_is_spelled_in_full() -> None:
    """``--tp`` is not a declared option, only an argparse prefix match."""
    emitted = _emitted_flags()

    assert "--tensor-parallel-size" in emitted
    for flag in ("--tp", "--tp-size"):
        files = emitted.get(flag, set())
        for name in files:
            source = (_ADAPTER_DIR / name).read_text(encoding="utf-8")
            for line in source.splitlines():
                if f'"{flag}"' in line:
                    assert line.lstrip().startswith("#"), (
                        f"{name} emits the abbreviated {flag!r}. Emit --tensor-parallel-size: "
                        "the short form is not a declared option and survives only by prefix matching."
                    )


@pytest.mark.parametrize("abbreviation", ["--tp", "--dp", "--pp", "--ep"])
def test_no_parallelism_abbreviation_is_emitted(abbreviation: str) -> None:
    """The whole class, not just the one instance that was found."""
    for flag, files in _emitted_flags().items():
        if flag != abbreviation:
            continue
        for name in sorted(files):
            source = (_ADAPTER_DIR / name).read_text(encoding="utf-8")
            emitting = [
                line
                for line in source.splitlines()
                if f'"{abbreviation}"' in line and not line.lstrip().startswith("#")
            ]
            assert not emitting, f"{name} emits the abbreviated parallelism flag {abbreviation!r}: {emitting}"
