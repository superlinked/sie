"""Every SDK error a caller can be handed must be importable from ``sie_sdk``.

The SDK's error docstrings tell callers to catch specific classes ("new code
can catch :class:`RateLimitError` specifically to back off at a higher level"),
but a class that is raised on a public code path and absent from the top-level
package is unreachable advice: ``from sie_sdk import RateLimitError`` fails, so
the caller is forced to reach into ``sie_sdk.client.errors`` or to over-catch a
base class. These tests pin the public surface so an error added to
``client/errors.py`` cannot ship without an export again.

The TypeScript SDK already exports its full error set from ``index.ts``; the
cross-SDK parity test below keeps the two surfaces from drifting apart.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import sie_sdk
import sie_sdk.client
import sie_sdk.client.errors

# tests/ -> sie_sdk/ -> packages/ ; the TS SDK is a sibling package.
_TS_INDEX = Path(__file__).parents[2] / "sie_ts_sdk" / "src" / "index.ts"

# TS-only classes with no Python counterpart by design:
#   SIEStreamError  — the TS client's SSE-abort signal; the Python client
#                     surfaces stream faults as SIEConnectionError.
_TS_ONLY = frozenset({"SIEStreamError"})


def _public_error_classes(module: object) -> set[str]:
    """Names of public exception classes defined directly in ``module``."""
    return {
        name
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, BaseException) and not name.startswith("_") and obj.__module__ == module.__name__
    }


def test_every_error_class_is_exported_from_the_package_root() -> None:
    """No error defined in ``client/errors.py`` may be missing from ``sie_sdk``."""
    missing = sorted(_public_error_classes(sie_sdk.client.errors) - set(sie_sdk.__all__))
    assert not missing, f"error classes unreachable via `from sie_sdk import ...`: {missing}"


def test_every_error_class_is_exported_from_the_client_package() -> None:
    """``sie_sdk.client`` documents itself as re-exporting *all* client errors."""
    missing = sorted(_public_error_classes(sie_sdk.client.errors) - set(sie_sdk.client.__all__))
    assert not missing, f"error classes missing from sie_sdk.client.__all__: {missing}"


# Errors a caller can be handed that are NOT defined in client/errors.py, with
# the module that raises each. They belong in the root package but not in
# `sie_sdk.client`, which re-exports the *client's* errors:
#   GatedModelError      raised by sie_sdk.cache during a gated weight download
#   MalformedChunkError  raised by sie_sdk.jobs.decode_chunk_bytes; the client
#                        catches it internally and warns rather than propagating
_NON_CLIENT_ERRORS = frozenset({"GatedModelError", "MalformedChunkError"})


@pytest.mark.parametrize(
    "name",
    [
        "AccountInactiveError",
        "AccountStateUnavailableError",
        "EstimateUnroutableError",
        "GatedModelError",
        "IncompleteBatchError",
        "InputTooLongError",
        "InsufficientCreditsError",
        "JobFailedError",
        "LoraLoadingError",
        "MalformedChunkError",
        "ModelLoadFailedError",
        "ModelLoadingError",
        "PoolError",
        "ProvisioningError",
        "RateLimitError",
        "RequestError",
        "ResourceExhaustedError",
        "SIEConnectionError",
        "SIEError",
        "ServerError",
        "SpendLimitError",
    ],
)
def test_error_is_importable_and_declared(name: str) -> None:
    """Each caller-catchable error resolves as an attribute and is in ``__all__``.

    Checked against both public namespaces, except for the errors that are not
    the client's to export (see ``_NON_CLIENT_ERRORS``) — putting those in
    ``sie_sdk.client`` would advertise them as client errors, which they are
    not.
    """
    assert name in sie_sdk.__all__, f"{name} missing from sie_sdk.__all__"
    obj = getattr(sie_sdk, name)
    assert isinstance(obj, type), f"{name} is not a class"
    assert issubclass(obj, BaseException), f"{name} is not an exception type"

    if name in _NON_CLIENT_ERRORS:
        assert name not in sie_sdk.client.__all__, (
            f"{name} is not defined in client/errors.py; exporting it from sie_sdk.client "
            "would mislabel it as a client error"
        )
        return
    assert name in sie_sdk.client.__all__, f"{name} missing from sie_sdk.client.__all__"
    assert getattr(sie_sdk.client, name) is obj, f"{name} differs between sie_sdk and sie_sdk.client"


def test_all_entries_resolve() -> None:
    """``__all__`` must not advertise a name the package cannot supply."""
    unresolved = [name for name in sie_sdk.__all__ if not hasattr(sie_sdk, name)]
    assert not unresolved, f"names in __all__ with no attribute: {unresolved}"


def test_all_is_sorted_and_unique() -> None:
    """Keep ``__all__`` in the sorted, duplicate-free form ruff's RUF022 expects."""
    assert sie_sdk.__all__ == sorted(set(sie_sdk.__all__))
    assert sie_sdk.client.__all__ == sorted(set(sie_sdk.client.__all__))


def test_non_client_errors_are_still_reachable_from_the_root() -> None:
    """The classes excluded from `sie_sdk.client` must not be unreachable."""
    for name in sorted(_NON_CLIENT_ERRORS):
        assert name in sie_sdk.__all__
        assert issubclass(getattr(sie_sdk, name), BaseException)


@pytest.mark.skipif(not _TS_INDEX.is_file(), reason="TypeScript SDK not present in this checkout")
def test_python_exports_every_error_the_typescript_sdk_exports() -> None:
    """Both SDKs must offer the same catchable error vocabulary.

    A caller porting error handling between the two SDKs should not discover
    that a class exists in one and not the other.
    """
    block = re.search(r"export \{([^}]*)\} from \"\./errors\.js\";", _TS_INDEX.read_text())
    assert block is not None, "could not locate the error re-export block in index.ts"
    ts_errors = {name.strip() for name in block.group(1).split(",") if name.strip()}

    missing = sorted(ts_errors - _TS_ONLY - set(sie_sdk.__all__))
    assert not missing, f"exported by the TypeScript SDK but not by sie_sdk: {missing}"
