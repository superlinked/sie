from __future__ import annotations

from sie_server.bundle_requirements import (
    bundle_requirements_sha256,
    normalized_bundle_requirements,
    resolve_bundle_requirements,
)


def test_resolve_bundle_requirements_preserves_cli_semantics() -> None:
    assert resolve_bundle_requirements(
        {
            "plain": "",
            "bounded": ">=1,<2",
            "wheel": {"url": "https://example.com/wheel.whl", "marker": "sys_platform == 'linux'"},
            "versioned": {"version": "==3", "marker": "sys_platform == 'darwin'"},
        }
    ) == [
        "plain",
        "bounded>=1,<2",
        "wheel @ https://example.com/wheel.whl ; sys_platform == 'linux'",
        "versioned==3 ; sys_platform == 'darwin'",
    ]


def test_resolve_bundle_requirements_can_exclude_normalized_cuda_packages() -> None:
    assert resolve_bundle_requirements(
        {"Flash_Attn": "==1", "xformers": "==2", "portable": "==3"},
        exclude_cuda=True,
    ) == ["portable==3"]


def test_release_pin_normalization_is_sorted_and_marker_free() -> None:
    deps = {
        "z-last": "==2",
        "a-first": {"version": "==1", "marker": "sys_platform == 'linux'"},
    }

    assert normalized_bundle_requirements(deps) == ["a-first==1", "z-last==2"]
    assert bundle_requirements_sha256(deps) == "d2ccdab8db5e0df55e2b47ae7c7a8f4c9b7623bd42fb7c09f028114ed3aba884"
