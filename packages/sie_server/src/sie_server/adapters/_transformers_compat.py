"""Load-time security backport for the supported Transformers 4 runtime.

Transformers 5.3 restricts Hub attention kernels to ``kernels-community`` and
ignores serialized private attention settings (GHSA-29pf-2h5f-8g72). The default
SIE bundle still needs Transformers 4.57.6. Install this compatibility guard once,
before loading models, to retain that runtime without retaining the vulnerable
loader. It never wraps model inference or temporarily changes a trust setting.

Upstream fix: huggingface/transformers@a7f8e7ff37d87d1a1a0c8cf607971c607741452f.
"""

from __future__ import annotations

import functools
import re
import threading
from typing import Any

from packaging.version import Version

_INSTALL_LOCK = threading.Lock()
_PATCHED_VERSION = Version("4.57.6")
_UPSTREAM_FIXED_VERSION = Version("5.3.0")
# The native loader splits on ANY pipe, before parsing the repository/revision.
# A pipe in a later component must not redirect an allowlisted-looking string.
_KERNEL_PATTERN = re.compile(r"(?:paged\|)?kernels-community/[^/:|@\s]+(?:@[^/:|@\s]+)?(?::[^/:|@\s]+)?")
_PRIVATE_IMPLEMENTATION_KEYS = frozenset(
    {
        "_attn_implementation_internal",
        "_attn_implementation",
        "_experts_implementation_internal",
        "_experts_implementation",
    }
)
_CONFIG_STRUCTURE_KEYS = frozenset({"__dict__", "__class__", "attribute_map", "sub_configs"})


def _is_community_kernel(attn_implementation: str | None) -> bool:
    """Recognize the upstream allowlisted namespace before any Hub access."""
    return isinstance(attn_implementation, str) and _KERNEL_PATTERN.fullmatch(attn_implementation) is not None


def install_transformers_security_fixes() -> bool:
    """Install permanent loader guards; return whether this call installed them.

    Call during process initialization, before model loading begins. Installation
    is idempotent across threads. All subsequent loads use the same policy,
    including loads with ``trust_remote_code=True``, just as in upstream 5.3.
    Unsupported vulnerable versions fail closed instead of receiving an untested
    private-API patch. Versions with the upstream fix require no patch.
    """
    # Importing this helper must not require Transformers in remote-only runtimes.
    import transformers

    version = Version(transformers.__version__)
    if version >= _UPSTREAM_FIXED_VERSION:
        return False
    if version != _PATCHED_VERSION:
        raise RuntimeError(
            "SIE's Transformers security backport requires transformers==4.57.6; "
            "use that supported runtime or transformers>=5.3.0."
        )

    from transformers import PretrainedConfig, modeling_utils
    from transformers.integrations import hub_kernels

    with _INSTALL_LOCK:
        original_init = PretrainedConfig.__init__
        if getattr(original_init, "_sie_kernel_security_backport", False):
            if (
                hub_kernels.is_kernel is not _is_community_kernel
                or modeling_utils.is_kernel is not _is_community_kernel
            ):
                raise RuntimeError("The installed Transformers kernel security guard was replaced.")
            return False

        @functools.wraps(original_init)
        def guarded_init(self: Any, *args: Any, **kwargs: Any) -> None:
            # The property names are aliases for the private storage fields.
            # Keeping either alias would let model-owned JSON restore the same
            # vulnerable value after the native constructor chooses attention.
            # Native attention propagation follows class-defined sub_configs.
            # Model-owned maps could redirect it to __class__ and mutate the
            # shared configuration class during an otherwise ordinary load.
            forbidden_structure = _CONFIG_STRUCTURE_KEYS.intersection(kwargs)
            if forbidden_structure:
                raise ValueError("Model configuration cannot override Python object or config attribute structure.")
            clean_kwargs = {key: value for key, value in kwargs.items() if key not in _PRIVATE_IMPLEMENTATION_KEYS}
            original_init(self, *args, **clean_kwargs)

        guarded_init.__dict__["_sie_kernel_security_backport"] = True
        # Both functions resolve their globals in their original modules. The
        # native kernel loader therefore also sees the restricted predicate.
        # Apply this policy before exposing the constructor guard to callers.
        hub_kernels.is_kernel = _is_community_kernel  # ty: ignore[invalid-assignment]
        modeling_utils.is_kernel = _is_community_kernel  # ty: ignore[invalid-assignment]
        PretrainedConfig.__init__ = guarded_init  # ty: ignore[invalid-assignment]
        return True
