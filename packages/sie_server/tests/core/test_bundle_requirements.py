from __future__ import annotations

from pathlib import Path

import yaml
from sie_server.bundle_requirements import (
    bundle_requirements_sha256,
    lock_constraint_lines,
    locked_versions_from_uv_lock,
    merge_locked_requirements,
    normalized_bundle_requirements,
    parse_uv_lock,
    resolve_bundle_requirements,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BUNDLES = _REPO_ROOT / "packages" / "sie_server" / "bundles"


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
        {"Flash_Attn": "==1", "xformers": "==2", "FLA_core": {"version": "==3"}, "portable": "==4"},
        exclude_cuda=True,
    ) == ["portable==4"]


def test_release_pin_normalization_is_sorted_and_marker_free() -> None:
    deps = {
        "z-last": "==2",
        "a-first": {"version": "==1", "marker": "sys_platform == 'linux'"},
    }

    assert normalized_bundle_requirements(deps) == ["a-first==1", "z-last==2"]
    assert bundle_requirements_sha256(deps) == "d2ccdab8db5e0df55e2b47ae7c7a8f4c9b7623bd42fb7c09f028114ed3aba884"


def _bundle_requirements(name: str) -> list[str]:
    bundle = yaml.safe_load((_BUNDLES / f"{name}.yaml").read_text())
    return resolve_bundle_requirements(bundle["deps"])


def test_default_bundle_ranges_pin_to_uv_lock_versions() -> None:
    locked = locked_versions_from_uv_lock(_REPO_ROOT / "uv.lock")
    original = _bundle_requirements("default")
    merged = merge_locked_requirements(original, locked)

    assert "gliner>=0.2.26,<1" in original
    assert f"gliner=={locked['gliner']}" in merged
    assert "gliner>=0.2.26,<1" not in merged
    assert f"torch=={locked['torch']}" in merged
    assert f"requests=={locked['requests']}" in merged


def test_exact_pins_and_url_specs_override_the_lock() -> None:
    parsed = parse_uv_lock(_REPO_ROOT / "uv.lock")
    locked = parsed.versions
    original = _bundle_requirements("transformers5")
    merged = merge_locked_requirements(original, locked)
    url_specs = [requirement for requirement in original if " @" in requirement]

    assert url_specs
    assert "gliner2==2.0.0" in merged
    # Exact bundle pin, not the lock's ``0.24.1.*`` prefix constraint.
    torchvision_lines = [requirement for requirement in merged if requirement.startswith("torchvision==")]
    assert torchvision_lines == ["torchvision==0.24.1"]
    for requirement in url_specs:
        assert requirement in merged
    # Ranges that do not contain the locked version stay as written.
    assert "transformers>=5.14,<6" in merged
    assert "sentence-transformers>=5.6,<6" in merged
    constraints = lock_constraint_lines(original, parsed)
    assert "gliner2==2.0.0" not in constraints
    assert f"gliner2=={locked['gliner2']}" not in constraints
    assert not any(line.startswith("torchvision==") for line in constraints)
    assert all(" @" not in line for line in constraints)
    assert f"transformers=={locked['transformers']}" not in constraints
    assert f"sentence-transformers=={locked['sentence-transformers']}" not in constraints
    # The 4.x lock's huggingface-hub and safetensors do not satisfy transformers 5.
    assert f"huggingface-hub=={locked['huggingface-hub']}" not in constraints
    assert f"safetensors=={locked['safetensors']}" not in constraints
    # gliner2==2.0.0 moves off the lock, so its 1.x edges are not pinned.
    # torch stays because the bundle range still contains the locked version.
    assert f"peft=={locked['peft']}" not in constraints
    assert f"torch=={locked['torch']}" in constraints


def test_merge_leaves_url_specs_and_unlocked_packages_unchanged() -> None:
    locked = {"gliner": "0.2.26", "demo": "9.9.9"}
    requirements = [
        "gliner>=0.2.26,<1 ; sys_platform == 'linux'",
        "demo @ git+https://example.com/demo.git@abc123",
        "wheel @ https://example.com/wheel.whl ; sys_platform == 'linux'",
        "absent>=1,<2",
        "pinned==3",
    ]

    assert merge_locked_requirements(requirements, locked) == [
        "gliner==0.2.26 ; sys_platform == 'linux'",
        "demo @ git+https://example.com/demo.git@abc123",
        "wheel @ https://example.com/wheel.whl ; sys_platform == 'linux'",
        "absent>=1,<2",
        "pinned==3",
    ]


def test_local_version_builds_share_one_prefix_constraint() -> None:
    lock = """
version = 1
[[package]]
name = "torch"
version = "2.9.1"

[[package]]
name = "torch"
version = "2.9.1+cu129"
"""
    assert locked_versions_from_uv_lock(lock) == {"torch": "2.9.1.*"}
    assert merge_locked_requirements(["torch>=2.9,<2.10"], {"torch": "2.9.1.*"}) == ["torch==2.9.1.*"]
    constraints = lock_constraint_lines(["torch>=2.9,<2.10"], parse_uv_lock(lock))
    assert "torch==2.9.1.*" in constraints
    assert not any("skipped:" in line for line in constraints)


def test_default_bundle_constraints_pin_transitive_deps_and_name_the_downgrade() -> None:
    parsed = parse_uv_lock(_REPO_ROOT / "uv.lock")
    constraints = lock_constraint_lines(_bundle_requirements("default"), parsed)

    assert constraints[0] == "# Ranged dependencies are pinned to uv.lock. This can install an older"
    assert "older" in constraints[0]
    assert "release than an unconstrained build" in constraints[1]
    assert any(line.startswith("# For example, sentence-transformers and gliner ") for line in constraints)
    assert f"onnxruntime=={parsed.versions['onnxruntime']}" in constraints
    assert f"docling-core=={parsed.versions['docling-core']}" in constraints
    assert f"sentence-transformers=={parsed.versions['sentence-transformers']}" in constraints
    assert f"gliner=={parsed.versions['gliner']}" in constraints
    # Exact bundle pins stay requirements, not constraints.
    assert not any(line.startswith("pillow==") for line in constraints)
    assert not any(line.startswith("gliformer==") for line in constraints)
    assert all("[" not in line for line in constraints)
    assert f"chromadb=={parsed.versions['chromadb']}" not in constraints
    # The public torch pin must not import the cu129 wheel's nvidia stack.
    assert not any(line.startswith("nvidia-cublas-cu12==") for line in constraints)


def test_constraints_pin_transitive_closure_not_the_whole_lock() -> None:
    lock = """
version = 1
[[package]]
name = "direct"
version = "1.2.0"
dependencies = [
    { name = "onnxruntime" },
    { name = "docling-core" },
]

[[package]]
name = "onnxruntime"
version = "1.25.0"

[[package]]
name = "docling-core"
version = "2.79.0"
dependencies = [
    { name = "leaf" },
]

[[package]]
name = "leaf"
version = "8.0.0"

[[package]]
name = "outside"
version = "9.0.0"
"""
    constraints = lock_constraint_lines(["direct>=1,<2"], parse_uv_lock(lock))
    assert "direct==1.2.0" in constraints
    assert "onnxruntime==1.25.0" in constraints
    assert "docling-core==2.79.0" in constraints
    assert "leaf==8.0.0" in constraints
    assert "outside==9.0.0" not in constraints


def test_constraint_lines_strip_extras() -> None:
    lock = """
version = 1
[[package]]
name = "demo"
version = "1.2.3"
dependencies = [
    { name = "base" },
]

[package.optional-dependencies]
extra = [
    { name = "optional-dep" },
]

[[package]]
name = "base"
version = "4.0.0"

[[package]]
name = "optional-dep"
version = "5.0.0"

[[package]]
name = "not-requested"
version = "6.0.0"
"""
    constraints = lock_constraint_lines(
        ["demo[extra]>=1,<2 ; sys_platform == 'linux'"],
        parse_uv_lock(lock),
    )
    assert "demo==1.2.3" in constraints
    assert "base==4.0.0" in constraints
    assert "optional-dep==5.0.0" in constraints
    assert "not-requested==6.0.0" not in constraints
    assert all("[" not in line for line in constraints)
    assert all(";" not in line for line in constraints)


def test_exact_override_neighbors_stay_pinned_only_when_the_pin_matches_the_lock() -> None:
    lock = """
version = 1
[[package]]
name = "kept"
version = "1.0.0"
dependencies = [
    { name = "shared" },
]

[[package]]
name = "matched"
version = "1.0.0"
dependencies = [
    { name = "matched-neighbor" },
    { name = "contested" },
]

[[package]]
name = "pinned-override"
version = "1.0.0"
dependencies = [
    { name = "neighbor" },
    { name = "contested" },
]

[[package]]
name = "moved"
version = "4.0.0"
dependencies = [
    { name = "old-only" },
    { name = "contested" },
]

[[package]]
name = "neighbor"
version = "2.0.0"

[[package]]
name = "matched-neighbor"
version = "2.1.0"

[[package]]
name = "shared"
version = "3.0.0"

[[package]]
name = "old-only"
version = "5.0.0"

[[package]]
name = "contested"
version = "7.0.0"
"""
    constraints = lock_constraint_lines(
        ["kept>=1,<2", "matched==1.0.0", "pinned-override==9.9.9", "moved>=5,<6"],
        parse_uv_lock(lock),
    )
    assert "kept==1.0.0" in constraints
    assert "shared==3.0.0" in constraints
    assert "matched-neighbor==2.1.0" in constraints
    assert "neighbor==2.0.0" not in constraints
    assert not any(line.startswith("matched==") for line in constraints)
    assert not any(line.startswith("pinned-override==") for line in constraints)
    assert not any(line.startswith("moved==") for line in constraints)
    assert "old-only==5.0.0" not in constraints
    assert "contested==7.0.0" not in constraints


def test_untrusted_root_unpins_a_transitive_also_reached_by_a_constrained_root() -> None:
    lock = """
version = 1
[[package]]
name = "sentence-transformers"
version = "5.4.1"
dependencies = [
    { name = "huggingface-hub" },
]

[[package]]
name = "transformers"
version = "4.57.6"
dependencies = [
    { name = "huggingface-hub" },
]

[[package]]
name = "huggingface-hub"
version = "0.36.2"
"""
    constraints = lock_constraint_lines(
        ["sentence-transformers>=5.4.1,<6", "transformers>=5.14,<6"],
        parse_uv_lock(lock),
    )
    assert "sentence-transformers==5.4.1" in constraints
    assert "huggingface-hub==0.36.2" not in constraints
    assert not any(line.startswith("transformers==") for line in constraints)


def test_public_prefix_constraint_ignores_local_build_edges() -> None:
    lock = """
version = 1
[[package]]
name = "torch"
version = "2.9.1"
dependencies = [
    { name = "filelock" },
]

[[package]]
name = "torch"
version = "2.9.1+cu129"
dependencies = [
    { name = "filelock" },
    { name = "nvidia-cublas-cu12" },
]

[[package]]
name = "filelock"
version = "3.0.0"

[[package]]
name = "nvidia-cublas-cu12"
version = "12.9.1.4"

[[package]]
name = "torch-local-only"
version = "2.9.1+cu129"
dependencies = [
    { name = "nvidia-cublas-cu12" },
]

[[package]]
name = "torch-local-only"
version = "2.9.1+cu130"
dependencies = [
    { name = "nvidia-cublas-cu13" },
]

[[package]]
name = "nvidia-cublas-cu13"
version = "13.0.0"
"""
    parsed = parse_uv_lock(lock)
    assert parsed.versions["torch"] == "2.9.1.*"
    assert parsed.versions["torch-local-only"] == "2.9.1.*"
    public = lock_constraint_lines(["torch>=2.9,<2.10"], parsed)
    assert "torch==2.9.1.*" in public
    assert "filelock==3.0.0" in public
    assert "nvidia-cublas-cu12==12.9.1.4" not in public
    local_only = lock_constraint_lines(["torch-local-only>=2.9,<2.10"], parsed)
    assert "torch-local-only==2.9.1.*" in local_only
    assert "nvidia-cublas-cu12==12.9.1.4" not in local_only
    assert "nvidia-cublas-cu13==13.0.0" not in local_only


def test_sole_local_build_keeps_its_own_edges() -> None:
    lock = """
version = 1
[[package]]
name = "torch"
version = "2.9.1+cu129"
dependencies = [
    { name = "nvidia-cublas-cu12" },
]

[[package]]
name = "nvidia-cublas-cu12"
version = "12.9.1.4"
"""
    constraints = lock_constraint_lines(["torch>=2.9,<2.10"], parse_uv_lock(lock))
    assert "torch==2.9.1+cu129" in constraints
    assert "nvidia-cublas-cu12==12.9.1.4" in constraints


def test_moved_exact_pins_do_not_contribute_lock_edges() -> None:
    parsed = parse_uv_lock(_REPO_ROOT / "uv.lock")
    constraints = lock_constraint_lines(_bundle_requirements("tensorrt-llm"), parsed)

    assert f"huggingface-hub=={parsed.versions['huggingface-hub']}" not in constraints
    assert f"setuptools=={parsed.versions['setuptools']}" not in constraints
    assert not any(line.startswith("nvidia-cublas-cu12==") for line in constraints)
    assert not any(line.startswith("torch==") for line in constraints)
    assert not any(line.startswith("transformers==") for line in constraints)


def test_multiple_public_versions_are_skipped_without_raising() -> None:
    lock = """
version = 1
[[package]]
name = "direct"
version = "1.0.0"
dependencies = [
    { name = "split" },
    { name = "stable" },
]

[[package]]
name = "split"
version = "1.0.0"
dependencies = [
    { name = "stable" },
]

[[package]]
name = "split"
version = "2.0.0"

[[package]]
name = "stable"
version = "3.1.0"
"""
    parsed = parse_uv_lock(lock)
    assert locked_versions_from_uv_lock(lock) == {"direct": "1.0.0", "stable": "3.1.0"}
    assert "split" in parsed.ambiguous
    constraints = lock_constraint_lines(["direct>=1,<2"], parsed)
    assert "direct==1.0.0" in constraints
    assert "stable==3.1.0" in constraints
    assert not any(line.startswith("split==") for line in constraints)
    assert "# split skipped: uv.lock has multiple public versions" in constraints
