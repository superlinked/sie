"""Model ids in user-facing examples must exist in the shipped catalog.

An example is a promise that the snippet runs. The registry matches ids
exactly — ``has_model`` is ``name in self._configs``, keyed by each config's
``sie_id`` — and there is no short-name aliasing, so an example that drops the
org prefix (``bge-m3`` for ``BAAI/bge-m3``) does not degrade: it 404s.

This guards the copy-paste path in both SDKs and the docs. It deliberately
scans only calls on a ``client`` receiver, so ``"text".encode("utf-8")`` and
similar non-SIE calls are not mistaken for model ids.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).parents[3]
_MODELS_DIR = _REPO_ROOT / "packages" / "sie_server" / "models"

# Only calls on a `client` receiver: `client.encode("X"`, `await client.score("X"`.
#
# The model id is frequently on the line after the open paren, and that line
# carries the surrounding block's continuation marker — `...` in a Python
# doctest, `*` in a JSDoc comment — which is not whitespace. Allowing for it is
# what makes multi-line examples visible to this guard at all.
_CLIENT_CALL = re.compile(
    r"""
    client \s* \. \s* (?: encode | score | extract | generate ) \s* \( \s*
    (?: \n [ \t]* (?: \* | \.\.\. | >>> | // )? [ \t]* )?
    ["'] ([^"'\n]+) ["']
    """,
    re.VERBOSE,
)

# Files whose examples a user is expected to copy and run.
_SCANNED = (
    "packages/sie_sdk/src",
    "packages/sie_ts_sdk/src",
    "integrations",
    "docs",
    "README.md",
)

_SUFFIXES = {".py", ".ts", ".md"}

# Placeholders that are obviously not catalog ids.
_PLACEHOLDERS = frozenset({"model", "MODEL", "your-model", "<model>"})


def _catalog_ids() -> set[str]:
    ids: set[str] = set()
    for path in _MODELS_DIR.glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("sie_id"), str):
            ids.add(data["sie_id"])
    return ids


def _documented_ids() -> list[tuple[str, str]]:
    """(model_id, "path:line") for every model id in a documented client call."""
    found: list[tuple[str, str]] = []
    for rel in _SCANNED:
        root = _REPO_ROOT / rel
        if not root.exists():
            continue
        paths = [root] if root.is_file() else [p for p in root.rglob("*") if p.suffix in _SUFFIXES]
        for path in paths:
            if "__pycache__" in path.parts or "node_modules" in path.parts:
                continue
            try:
                text = path.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            for match in _CLIENT_CALL.finditer(text):
                model_id = match.group(1)
                if model_id in _PLACEHOLDERS or any(ch in model_id for ch in "{}$<>"):
                    continue
                line = text[: match.start()].count("\n") + 1
                found.append((model_id, f"{path.relative_to(_REPO_ROOT)}:{line}"))
    return found


@pytest.mark.skipif(not _MODELS_DIR.is_dir(), reason="model catalog not present in this checkout")
def test_catalog_is_readable() -> None:
    """Guard the guard: an empty catalog would make the check below vacuous."""
    assert len(_catalog_ids()) > 100


@pytest.mark.skipif(not _MODELS_DIR.is_dir(), reason="model catalog not present in this checkout")
def test_every_documented_model_id_exists() -> None:
    """A documented id that is not in the catalog 404s when copy-pasted."""
    catalog = _catalog_ids()
    documented = _documented_ids()
    assert documented, "scanner found no documented model ids — the pattern has rotted"

    unknown = sorted({f"{model_id!r} at {where}" for model_id, where in documented if model_id not in catalog})
    assert not unknown, "model ids in user-facing examples that are not in the catalog:\n  " + "\n  ".join(unknown)
