# Regression tests for ``sie_server.core.oom.is_oom_error``.
#
# The OOM detector is the single source of truth that decides whether an
# exception flows down the recovery / 503 path or the legacy 500 path. Two
# classes of bug have been seen historically:
#
# 1. **False positives**: a substring match that fires on unrelated error
#    messages (e.g., a model named ``oom-classifier`` triggering OOM
#    recovery). These pollute dashboards, push the SDK into pointless
#    retries, and obscure the real failure.
# 2. **False negatives on wrapped errors**: when the worker wraps an OOM
#    in ``ResourceExhaustedError`` and the wrapper's message no longer
#    contains the literal phrase, substring-only matchers miss it.
#
# These tests pin both behaviours so a future refactor of the matcher
# can't silently regress them.

from __future__ import annotations

import pytest
from sie_server.core.oom import (
    ResourceExhausted,
    ResourceExhaustedError,
    is_oom_error,
)

# -------------------------------------------------------------------------
# Positive cases — these MUST be classified as OOM
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory. Tried to allocate 2.00 GiB",
        "MPS backend out of memory (MPS allocator)",
        "out of memory",
        "OUT OF MEMORY",  # case-insensitive
        "Cannot allocate memory",
        "cannot allocate memory",
        "Failed to allocate 1024 bytes on device",
        "torch.cuda.OutOfMemoryError: CUDA out of memory.",
    ],
    ids=[
        "cuda-oom-with-amount",
        "mps-oom",
        "generic-out-of-memory",
        "uppercase-out-of-memory",
        "cannot-allocate-titlecase",
        "cannot-allocate-lowercase",
        "failed-to-allocate",
        "torch-typed-oom-message",
    ],
)
def test_real_oom_messages_classify_positively(message: str) -> None:
    assert is_oom_error(RuntimeError(message)) is True


def test_typed_resource_exhausted_classifies_positively() -> None:
    """A ``ResourceExhaustedError`` is authoritative regardless of message.

    The worker's ``BatchExecutor`` raises this after recovery is
    exhausted; substring-only matching is a fallback for un-wrapped
    OOMs that escaped without being routed through recovery.
    """
    marker = ResourceExhausted(
        operation="encode",
        attempts=4,
        original_message="some upstream message",
    )
    err = ResourceExhaustedError(
        # Intentionally use a message that contains NONE of the OOM
        # substrings — the typed signal must still classify it.
        "Recovery exhausted; allocation refused upstream.",
        marker=marker,
    )
    assert is_oom_error(err) is True


# -------------------------------------------------------------------------
# Negative cases — these MUST NOT be classified as OOM
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        # Pre-fix bug: ``\boom\b`` matched "oom" inside hyphenated words
        # because Python regex treats ``-`` as a word boundary. A model
        # name like ``oom-classifier`` would mis-classify any error
        # mentioning it as OOM, sending the SDK into a futile retry loop.
        "Failed to load oom-classifier model",
        "Permission denied: /models/oom-detector/config.yaml",
        # Standalone "oom" as a word should also NOT match — we removed
        # the regex entirely; the three remaining indicators are specific
        # phrases. (If "OOM killed" really needs to match in the future,
        # add it as a full phrase, not a bare token.)
        "Process was OOM killed by the kernel",
        "OOMError raised by upstream",
        # Non-OOM runtime errors that should keep the legacy 500 path.
        "Tensor shape mismatch: got [1,2,3], expected [4,5,6]",
        "Invalid input: text exceeds max length",
        "Adapter not found for model 'foo'",
        # Common false-positive triggers — model names with substrings.
        "Failed to load zoom-detector",
        "Loom-net inference failed",
        "Boomerang model returned unexpected dtype",
    ],
    ids=[
        "oom-classifier-name",
        "oom-detector-path",
        "oom-killed-kernel",
        "OOMError-upstream",
        "tensor-shape-mismatch",
        "invalid-input",
        "adapter-not-found",
        "zoom-detector",
        "loom-net",
        "boomerang",
    ],
)
def test_non_oom_messages_classify_negatively(message: str) -> None:
    """Critical regression: don't classify these as OOM.

    A false positive on any of these strings would push the SDK into the
    OOM auto-retry path for a non-transient failure, hiding the real
    error from the caller and producing 5×N spurious 503s on dashboards.
    """
    assert is_oom_error(RuntimeError(message)) is False, f"Misclassified as OOM: {message!r}"


def test_empty_message_is_not_oom() -> None:
    assert is_oom_error(RuntimeError("")) is False


def test_value_error_with_oom_substring_still_classifies() -> None:
    """The matcher keys on the message, not the exception type.

    PyTorch raises ``RuntimeError`` for OOM, but custom adapters or
    transports might wrap it in ``ValueError``/``OSError``. We still
    want to recognise the underlying signal.
    """
    assert is_oom_error(ValueError("CUDA out of memory")) is True
    assert is_oom_error(OSError("Cannot allocate memory")) is True
