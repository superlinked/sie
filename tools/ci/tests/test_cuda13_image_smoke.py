from __future__ import annotations

from unittest.mock import Mock

import pytest

from tools.ci import cuda13_image_smoke


def test_exact_bundle_allowlist():
    assert cuda13_image_smoke.ALLOWED_BUNDLES == ("sglang-cu130", "tensorrt-llm")


@pytest.mark.parametrize("bundle", cuda13_image_smoke.ALLOWED_BUNDLES)
def test_commands_are_networkless_offline_and_never_publish(bundle):
    compile(cuda13_image_smoke.validation_script(bundle), f"<{bundle}-image-check>", "exec")
    commands = cuda13_image_smoke.docker_commands(bundle, "sie-cuda13-smoke:local")
    assert len(commands) == 2
    for command in commands:
        assert command[:3] == ["docker", "run", "--rm"]
        assert ["--network", "none"] == command[3:5]
        assert "HF_HUB_OFFLINE=1" in command
        assert "TRANSFORMERS_OFFLINE=1" in command
        assert not {"push", "login"} & set(command)
    assert commands[1][-5:] == ["resolve-deps", "--bundle", bundle, "--models-dir", "/app/models"]


def test_sglang_check_uses_safe_distribution_and_adapter_surfaces():
    script = cuda13_image_smoke.validation_script("sglang-cu130")
    assert 'distribution("sglang")' in script
    assert "import tvm_ffi" in script
    assert "import xgrammar" in script
    assert "from sie_server.adapters.sglang import cuda13, gemma, generation" in script
    assert "import sglang" not in script


def test_sglang_native_wheels_require_exact_cu130_local_variants():
    script = cuda13_image_smoke.validation_script("sglang-cu130")
    assert '(("sgl-deep-gemm", "0.1.2"), ("sglang-kernel", "0.4.3"))' in script
    assert '.partition("+")' in script
    assert '(upstream, "+", "cu130")' in script


def test_tensorrt_check_locates_engine_and_verifies_qualified_sources():
    script = cuda13_image_smoke.validation_script("tensorrt-llm")
    assert 'find_spec("tensorrt_llm")' in script
    assert "import tensorrt_llm" not in script
    assert "HF_WEIGHT_LOADER_SHA256" in script
    assert "MODELING_T5_SHA256" in script
    assert "verify_exact_transformers_5_sources" in script
    assert '"tensorrt-llm": "1.3.0rc24"' in script


def test_unknown_bundle_is_rejected_before_docker(monkeypatch):
    run = Mock()
    monkeypatch.setattr(cuda13_image_smoke.subprocess, "run", run)
    with pytest.raises(ValueError, match="unsupported CUDA 13 bundle"):
        cuda13_image_smoke.docker_commands("unknown", "image:local")
    run.assert_not_called()
