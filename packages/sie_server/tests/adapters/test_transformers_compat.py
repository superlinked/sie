"""Offline adversarial controls for the actual Transformers 4 model loader."""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import torch
import transformers
from sie_server.adapters._transformers_compat import install_transformers_security_fixes
from transformers import (
    AutoModelForCausalLM,
    CLIPConfig,
    LlamaConfig,
    LlamaForCausalLM,
    PretrainedConfig,
    modeling_utils,
)
from transformers.integrations import hub_kernels

pytestmark = pytest.mark.skipif(transformers.__version__ != "4.57.6", reason="Tests the exact supported 4.x runtime")

_ATTACKER_KERNEL = "sie-security-probe.invalid/owned-marker:attention"
# Capture native bindings at collection, before adapter tests install the
# permanent production policy. Restore the preceding test's policy on teardown.
_NATIVE_CONFIG_INIT = PretrainedConfig.__init__
_NATIVE_HUB_IS_KERNEL = hub_kernels.is_kernel
_NATIVE_MODEL_IS_KERNEL = modeling_utils.is_kernel


@pytest.fixture(autouse=True)
def _restore_loader_and_disable_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # Production installs permanently. Tests restore the original bindings so a
    # preceding test cannot make an unguarded negative control pass accidentally.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert not getattr(_NATIVE_CONFIG_INIT, "_sie_kernel_security_backport", False)
    monkeypatch.setattr(PretrainedConfig, "__init__", _NATIVE_CONFIG_INIT)
    monkeypatch.setattr(hub_kernels, "is_kernel", _NATIVE_HUB_IS_KERNEL)
    monkeypatch.setattr(modeling_utils, "is_kernel", _NATIVE_MODEL_IS_KERNEL)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    with torch.random.fork_rng():
        torch.manual_seed(137)
        config = LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            attn_implementation="eager",
        )
        LlamaForCausalLM(config).save_pretrained(tmp_path)
    return tmp_path


@pytest.fixture
def kernel_requests(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def intercept(repo_id: str, **kwargs: Any) -> None:
        calls.append(repo_id)
        raise RuntimeError("Owned offline kernel boundary reached; no fetch or external code executed")

    # Exercise the real native loader even if the optional kernels package is
    # absent. Intercept only its network/code-loading boundary.
    monkeypatch.setattr(hub_kernels, "_kernels_available", True)
    monkeypatch.setattr(hub_kernels, "get_kernel", intercept, raising=False)
    return calls


def _write_config(checkpoint: Path, updates: dict[str, Any]) -> None:
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config.update(updates)
    path.write_text(json.dumps(config))


@pytest.mark.parametrize("field", ["_attn_implementation_internal", "_attn_implementation"])
@pytest.mark.parametrize("trust", [False, None, True])
def test_unguarded_checkpoint_reaches_kernel_boundary(
    checkpoint: Path, kernel_requests: list[str], field: str, trust: bool | None
) -> None:
    _write_config(checkpoint, {field: _ATTACKER_KERNEL})
    with pytest.raises(ValueError, match="Owned offline kernel boundary reached"):
        AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=trust, attn_implementation="eager"
        )
    assert kernel_requests == ["sie-security-probe.invalid/owned-marker"]


@pytest.mark.parametrize("field", ["_attn_implementation_internal", "_attn_implementation"])
@pytest.mark.parametrize("trust", [False, None, True])
def test_checkpoint_cannot_replace_explicit_attention(
    checkpoint: Path, kernel_requests: list[str], field: str, trust: bool | None
) -> None:
    _write_config(checkpoint, {field: _ATTACKER_KERNEL})
    assert install_transformers_security_fixes()
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=trust, attn_implementation="eager"
    )
    assert type(model) is LlamaForCausalLM
    assert model.config._attn_implementation == "eager"
    assert not kernel_requests


@pytest.mark.parametrize(
    "kernel",
    [
        _ATTACKER_KERNEL,
        "paged|sie-security-probe.invalid/owned-marker@main:attention",
        "kernels-community.evil/owned-marker",
        "Kernels-community/owned-marker",
        " kernels-community/owned-marker",
        "kernels-community/../owned-marker",
        "other|kernels-community/owned-marker",
        "kernels-community/name|owned-marker:attention",
        "kernels-community/name@main|owned-marker:attention",
        "kernels-community/name:attention|owned-marker",
        "kernels-community/name@main@other:attention",
        "paged|paged|kernels-community/owned-marker:attention",
        "kernels-community/owned-marker:attention\n",
    ],
)
def test_explicit_foreign_kernel_never_reaches_loader(
    checkpoint: Path, kernel_requests: list[str], kernel: str
) -> None:
    install_transformers_security_fixes()
    with pytest.raises(ValueError, match="not supported"):
        AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False, attn_implementation=kernel
        )
    # The direct native entrypoint and its modeling_utils alias must also be
    # guarded; checking only the caller's predicate would leave a second route.
    hub_kernels.load_and_register_kernel(kernel)
    modeling_utils.load_and_register_kernel(kernel)
    assert not kernel_requests


@pytest.mark.parametrize(
    "kernel",
    [
        "kernels-community/owned-marker:attention",
        "kernels-community/owned-marker@main:attention",
        "paged|kernels-community/owned-marker@main:attention",
    ],
)
def test_upstream_community_namespace_remains_available(kernel_requests: list[str], kernel: str) -> None:
    install_transformers_security_fixes()
    with pytest.raises(ValueError, match="Owned offline kernel boundary reached"):
        hub_kernels.load_and_register_kernel(kernel)
    assert kernel_requests == ["kernels-community/owned-marker"]


@pytest.mark.parametrize("field", ["_attn_implementation_internal", "_attn_implementation"])
def test_nested_native_configs_discard_serialized_private_attention(field: str) -> None:
    data = {
        "text_config": {"hidden_size": 8, "num_attention_heads": 2, field: _ATTACKER_KERNEL},
        "vision_config": {"hidden_size": 8, "num_attention_heads": 2, field: _ATTACKER_KERNEL},
    }
    untouched = copy.deepcopy(data)
    native_subconfigs = dict(CLIPConfig.sub_configs)
    install_transformers_security_fixes()
    config = CLIPConfig(**data)
    assert config.text_config._attn_implementation is None
    assert config.vision_config._attn_implementation is None
    assert data == untouched
    assert CLIPConfig.sub_configs == native_subconfigs


@pytest.mark.parametrize("field", ["__dict__", "__class__", "attribute_map", "sub_configs"])
def test_checkpoint_cannot_replace_config_structure(checkpoint: Path, kernel_requests: list[str], field: str) -> None:
    attack = (
        {"__class__": "owned-json-marker"}
        if field == "sub_configs"
        else {"_attn_implementation_internal": _ATTACKER_KERNEL}
    )
    _write_config(checkpoint, {field: attack})
    native_attention_property = LlamaConfig._attn_implementation
    install_transformers_security_fixes()
    with pytest.raises(ValueError, match="cannot override"):
        AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    assert not kernel_requests
    assert LlamaConfig._attn_implementation is native_attention_property


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_native_forward_weights_and_outputs_unchanged(checkpoint: Path, attention: str) -> None:
    before = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, attn_implementation=attention)
    original_forward = LlamaForCausalLM.forward
    tokens = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = before(tokens).logits
    install_transformers_security_fixes()
    after = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, attn_implementation=attention)
    assert type(before) is type(after) is LlamaForCausalLM
    assert LlamaForCausalLM.forward is original_forward
    assert before.config.to_dict() == after.config.to_dict()
    for key, value in before.state_dict().items():
        assert torch.equal(value, after.state_dict()[key])
    with torch.inference_mode():
        assert torch.equal(expected, after(tokens).logits)


def test_installation_is_idempotent_across_concurrent_loaders(kernel_requests: list[str]) -> None:
    original_forward = LlamaForCausalLM.forward

    def load(_: int) -> bool:
        installed = install_transformers_security_fixes()
        config = LlamaConfig(_attn_implementation_internal=_ATTACKER_KERNEL)
        assert config._attn_implementation is None
        return installed

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sum(executor.map(load, range(32))) == 1
    assert LlamaForCausalLM.forward is original_forward
    assert not kernel_requests


@pytest.mark.parametrize("version", ["4.57.5", "5.0.0", "5.2.0", "5.3.0rc1"])
def test_unsupported_vulnerable_runtime_fails_before_mutation(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    original = (PretrainedConfig.__init__, hub_kernels.is_kernel, modeling_utils.is_kernel)
    # Importing model classes can replace the root lazy module; resolve the
    # current sys.modules binding, as the optional installer import does.
    monkeypatch.setattr("transformers.__version__", version)
    with pytest.raises(RuntimeError, match=r"requires transformers==4\.57\.6"):
        install_transformers_security_fixes()
    assert original == (PretrainedConfig.__init__, hub_kernels.is_kernel, modeling_utils.is_kernel)


@pytest.mark.parametrize("version", ["5.3.0", "5.12.1", "5.17.0"])
def test_fixed_runtime_is_untouched(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    original = (PretrainedConfig.__init__, hub_kernels.is_kernel, modeling_utils.is_kernel)
    monkeypatch.setattr("transformers.__version__", version)
    assert not install_transformers_security_fixes()
    assert original == (PretrainedConfig.__init__, hub_kernels.is_kernel, modeling_utils.is_kernel)


def test_replaced_guard_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    install_transformers_security_fixes()
    monkeypatch.setattr(hub_kernels, "is_kernel", lambda _: True)
    with pytest.raises(RuntimeError, match="guard was replaced"):
        install_transformers_security_fixes()
