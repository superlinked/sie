"""Dependency-only bundle requirement normalization.

This module deliberately stays free of the serving/runtime dependency graph so
release tooling can hash bundle requirements without installing Torch or CUDA.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from hashlib import sha256
from typing import cast

_CUDA_ONLY_PACKAGES = frozenset({"flash-attn", "xformers"})


def resolve_bundle_requirements(
    bundle_deps: Mapping[str, object],
    *,
    exclude_cuda: bool = False,
) -> list[str]:
    """Convert a bundle ``deps`` mapping into canonical PEP 508 strings."""
    requirements: list[str] = []
    for package, constraint in bundle_deps.items():
        normalized = re.sub(r"[-_.]+", "-", package.lower())
        if exclude_cuda and normalized in _CUDA_ONLY_PACKAGES:
            continue

        if isinstance(constraint, Mapping):
            fields = cast("Mapping[str, object]", constraint)
            url = fields.get("url", "")
            marker = fields.get("marker", "")
            version = fields.get("version", "")
            if url:
                dependency = f"{package} @ {url}"
                if marker:
                    dependency += f" ; {marker}"
                requirements.append(dependency)
            elif version:
                dependency = f"{package}{version}"
                if marker:
                    dependency += f" ; {marker}"
                requirements.append(dependency)
            continue

        requirements.append(f"{package}{constraint}" if constraint else package)
    return requirements


def normalized_bundle_requirements(bundle_deps: Mapping[str, object]) -> list[str]:
    """Return the marker-free, sorted requirements used by release pins."""
    return sorted(
        requirement.split(";", maxsplit=1)[0].strip() for requirement in resolve_bundle_requirements(bundle_deps)
    )


def bundle_requirements_sha256(bundle_deps: Mapping[str, object]) -> str:
    """Hash the exact normalized requirement payload baked by worker images."""
    payload = "\n".join(normalized_bundle_requirements(bundle_deps)).encode()
    return sha256(payload).hexdigest()
