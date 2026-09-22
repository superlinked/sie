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
version that loads the adapter emitting it. Moving a bundle to a new engine
version means confirming every recorded flag again, including the flags model
profiles routed to that bundle pass through ``extra_launch_args``.

Verified against both pinned engines: ``0.5.10.post1`` (the ``sglang`` and
``sglang-embedding`` bundles) by reading ``sglang/srt/server_args.py`` from the
published sdist, 364 declared flags, and ``0.5.20`` (the ``sglang-cu130``
bundle) by building the engine's own parser inside a GPU container, 515
declared flags. Neither declares ``--tp``. In 0.5.20, ``--disable-cuda-graph``,
``--enable-lora`` and ``--log-level`` are strict prefixes of other declared
flags; each is declared itself, and argparse takes an exact match before any
prefix, so none of them is ambiguous.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from sie_server.core.loader import load_model_configs

_PACKAGE_DIR = Path(__file__).resolve().parents[2]
_ADAPTER_DIR = _PACKAGE_DIR / "src/sie_server/adapters/sglang"

# Flags confirmed present in both pinned engines' declared argument sets.
_DECLARED_IN_PINNED_ENGINE = frozenset(
    {
        "--attention-backend",
        "--context-length",
        "--disable-cuda-graph",
        "--dtype",
        "--enable-lora",
        "--grammar-backend",
        "--is-embedding",
        "--log-level",
        "--lora-paths",
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
# 0.5.10.post1 and present in 0.5.20, which is the bundle that loads them.
_NEWER_ENGINE_ONLY = frozenset(
    {
        "--disable-prefill-cuda-graph",  # cuda13.py and gemma.py, sglang-cu130 bundle
        "--enable-strict-thinking",  # cuda13.py, sglang-cu130 bundle
        "--mamba-radix-cache-strategy",  # cuda13.py and gemma.py, sglang-cu130 bundle
    }
)

# Emitted only by adapters whose bundle pins 0.5.10.post1, which declares them.
# 0.5.20 renamed both, and the sglang-cu130 adapters override the spelling.
_OLDER_ENGINE_ONLY = frozenset(
    {
        "--disable-piecewise-cuda-graph",
        "--mamba-scheduler-strategy",
    }
)

# Flags that profiles routed to the ``sglang-cu130`` bundle pass through
# ``extra_launch_args``, each confirmed declared by that bundle's engine (0.5.20).
_CU130_PROFILE_FLAGS = frozenset(
    {
        "--chunked-prefill-size",
        "--cuda-graph-max-bs-decode",
        "--disable-overlap-schedule",
        "--enable-strict-thinking",
        "--kv-cache-dtype",
        "--mamba-radix-cache-strategy",
        "--mamba-ssm-dtype",
        "--max-prefill-tokens",
        "--max-running-requests",
        "--mm-process-config",
        "--page-size",
        "--quantization",
    }
)

# Nothing currently. A flag belongs here only while it is emitted and known not
# to be declared, which should be never: the previous occupant,
# ``--pooling-method``, was removed once the sweep showed neither pinned engine
# declares it in any form.
_KNOWN_UNDECLARED: frozenset[str] = frozenset()

_FLAG_LITERAL = re.compile(r'"(--[a-z0-9][a-z0-9-]*)"')


def _emitted_flags() -> dict[str, set[Path]]:
    """Return every ``--flag`` literal in the adapter sources, by source path."""
    found: dict[str, set[Path]] = {}
    for source in sorted(_ADAPTER_DIR.rglob("*.py")):
        for match in _FLAG_LITERAL.finditer(source.read_text(encoding="utf-8")):
            found.setdefault(match.group(1), set()).add(source)
    return found


def test_no_emitted_flag_is_outside_the_verified_contract() -> None:
    """A new flag must be added to this module, which is where it gets checked."""
    known = _DECLARED_IN_PINNED_ENGINE | _NEWER_ENGINE_ONLY | _OLDER_ENGINE_ONLY | _KNOWN_UNDECLARED
    emitted = _emitted_flags()
    # "--tp" survives as comment text explaining why it is not emitted.
    unexpected = {
        flag: sorted(str(path.relative_to(_ADAPTER_DIR)) for path in paths)
        for flag, paths in emitted.items()
        if flag not in known and flag != "--tp"
    }

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
        for path in sorted(emitted.get(flag, set())):
            for line in path.read_text(encoding="utf-8").splitlines():
                if f'"{flag}"' in line:
                    assert line.lstrip().startswith("#"), (
                        f"{path.relative_to(_ADAPTER_DIR)} emits the abbreviated {flag!r}. Emit "
                        "--tensor-parallel-size: the short form is not a declared option and survives only by "
                        "prefix matching."
                    )


@pytest.mark.parametrize("abbreviation", ["--tp", "--dp", "--pp", "--ep"])
def test_no_parallelism_abbreviation_is_emitted(abbreviation: str) -> None:
    """The whole class, not just the one instance that was found."""
    for path in sorted(_emitted_flags().get(abbreviation, set())):
        emitting = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if f'"{abbreviation}"' in line and not line.lstrip().startswith("#")
        ]
        assert not emitting, (
            f"{path.relative_to(_ADAPTER_DIR)} emits the abbreviated parallelism flag {abbreviation!r}: {emitting}"
        )


def _cu130_profile_flags() -> dict[str, set[str]]:
    """Return every ``--flag`` that sglang-cu130 profiles pass in ``extra_launch_args``, by profile."""
    bundle = yaml.safe_load((_PACKAGE_DIR / "bundles/sglang-cu130.yaml").read_text(encoding="utf-8"))
    modules = set(bundle["adapters"])
    found: dict[str, set[str]] = {}
    for name, config in load_model_configs(_PACKAGE_DIR / "models").items():
        profile = config.resolve_profile("default")
        if (profile.adapter_path or "").partition(":")[0] not in modules:
            continue
        for arg in profile.loadtime.get("extra_launch_args") or []:
            if isinstance(arg, str) and arg.startswith("--"):
                found.setdefault(arg.partition("=")[0], set()).add(name)
    return found


def test_cu130_profile_launch_args_are_declared_by_the_bundle_engine() -> None:
    """A flag the engine renamed fails here rather than when the model loads."""
    flags = _cu130_profile_flags()
    unexpected = {flag: sorted(names) for flag, names in flags.items() if flag not in _CU130_PROFILE_FLAGS}

    assert flags, "no sglang-cu130 profile passes extra_launch_args; the catalog walk found nothing"
    assert not unexpected, (
        f"sglang-cu130 profiles pass engine flag(s) not recorded in this module: {unexpected}. "
        "Confirm the bundle's engine declares each one, then add it to _CU130_PROFILE_FLAGS."
    )
