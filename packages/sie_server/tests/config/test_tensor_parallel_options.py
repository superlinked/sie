"""Config-time rules for a profile that declares a tensor-parallel width.

Each rule exists because the failure it prevents is silent. A placement flag
passed straight to the engine serves on devices the registry never reserved,
and a mistyped width is dropped rather than rejected, so the model serves on
one card while the profile declares several. A flag that moves a listener is
the same class: the passthrough is appended after the flags the adapter builds
and the engine keeps the last spelling, so the engine ends up somewhere other
than where SIE reserved and expects it.
"""

from __future__ import annotations

from typing import Any

import pytest
from sie_server.config.model import (
    AdapterOptions,
    EmbeddingDim,
    EncodeTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)


def _options(**loadtime: Any) -> AdapterOptions:
    return AdapterOptions(loadtime=dict(loadtime))


class TestDeclaredWidth:
    def test_absent_width_needs_nothing_else(self) -> None:
        """Every existing profile must keep loading unchanged."""
        assert _options(mem_fraction_static=0.85).loadtime["mem_fraction_static"] == 0.85

    def test_width_one_needs_nothing_else(self) -> None:
        assert _options(tensor_parallel_size=1).loadtime["tensor_parallel_size"] == 1

    @pytest.mark.parametrize("bad", [0, 9, -1, 2.5, "4", True, None])
    def test_an_unusable_width_is_refused(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="tensor_parallel_size"):
            _options(tensor_parallel_size=bad)


class TestEngineSpecificRules:
    def test_a_width_needs_no_engine_specific_option_at_the_config_layer(self) -> None:
        """Any adapter that accepts a width may be given one.

        A read cap is an SGLang requirement, enforced by the SGLang adapters
        when they are constructed. Requiring it here refused every other
        adapter's width, since those adapters do not accept the option.
        """
        assert _options(tensor_parallel_size=2).loadtime["tensor_parallel_size"] == 2


class TestPlacementIsNotSmuggled:
    @pytest.mark.parametrize(
        "flag",
        ["--tp", "--tp-size", "--tensor-parallel-size", "--dp-size", "--nnodes", "--base-gpu-id"],
    )
    def test_a_placement_flag_in_the_raw_passthrough_is_refused(self, flag: str) -> None:
        """The registry reserves devices from the declared width. A width reaching
        the engine another way serves on devices nothing reserved.
        """
        with pytest.raises(ValueError, match="placement flag"):
            _options(extra_launch_args=[flag, "8"])

    @pytest.mark.parametrize(
        ("abbreviation", "flag"),
        [
            ("--tensor-parallel", "--tensor-parallel-size"),
            ("--tp-s", "--tp-size"),
            ("--base-gpu", "--base-gpu-id"),
            ("--dist-init=10.0.0.1:5000", "--dist-init-addr"),
        ],
    )
    def test_an_abbreviated_placement_flag_is_refused(self, abbreviation: str, flag: str) -> None:
        """SGLang accepts any unambiguous prefix of a long option."""
        with pytest.raises(ValueError, match=flag):
            _options(extra_launch_args=[abbreviation, "4"])

    def test_the_flag_equals_value_spelling_is_refused_too(self) -> None:
        with pytest.raises(ValueError, match="placement flag"):
            _options(extra_launch_args=["--tensor-parallel-size=8"])

    @pytest.mark.parametrize("variable", ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"])
    def test_a_device_visibility_variable_is_refused(self, variable: str) -> None:
        """The launcher overwrites it with the registry's mask, so setting it
        here is silently discarded rather than honoured.
        """
        with pytest.raises(ValueError, match=variable):
            _options(extra_env={variable: "0,1"})

    def test_an_ordinary_passthrough_still_works(self) -> None:
        options = _options(
            extra_launch_args=["--mamba-scheduler-strategy", "extra_buffer"],
            extra_env={"SGLANG_ENABLE_SPEC_V2": "1"},
        )

        assert options.loadtime["extra_launch_args"] == ["--mamba-scheduler-strategy", "extra_buffer"]


class TestListenersAreNotSmuggled:
    @pytest.mark.parametrize("flag", ["--nccl-port", "--host", "--port"])
    def test_a_listener_flag_in_the_raw_passthrough_is_refused(self, flag: str) -> None:
        with pytest.raises(ValueError, match="listener flag"):
            _options(extra_launch_args=[flag, "30411"])

    @pytest.mark.parametrize(
        ("spelling", "flag"),
        [
            ("--nccl-po", "--nccl-port"),
            ("--nccl-port=30411", "--nccl-port"),
            ("--po", "--port"),
            ("--port=8000", "--port"),
            ("--ho", "--host"),
            ("--host=0.0.0.0", "--host"),
        ],
    )
    def test_an_abbreviated_or_inline_listener_flag_is_refused(self, spelling: str, flag: str) -> None:
        with pytest.raises(ValueError, match=flag):
            _options(extra_launch_args=[spelling])

    def test_the_rendezvous_refusal_names_the_sanctioned_option(self) -> None:
        with pytest.raises(ValueError, match=r"loadtime\.nccl_port"):
            _options(extra_launch_args=["--nccl-port", "30411"])

    @pytest.mark.parametrize("flag", ["--host", "--port"])
    def test_the_http_refusal_names_the_engine_listener(self, flag: str) -> None:
        """The refused flags are the engine's, not the worker container's own."""
        with pytest.raises(ValueError, match="engine's HTTP listener"):
            _options(extra_launch_args=[flag, "8000"])

    @pytest.mark.parametrize("flag", ["--page-size", "--hicache-ratio", "--num-continuous-decode-steps"])
    def test_a_flag_that_only_shares_a_prefix_region_still_passes(self, flag: str) -> None:
        """Only an abbreviation of a refused flag is refused, not a neighbour of one."""
        assert _options(extra_launch_args=[flag, "1"]).loadtime["extra_launch_args"] == [flag, "1"]

    def test_the_sanctioned_rendezvous_option_still_loads(self) -> None:
        options = _options(tensor_parallel_size=2, nccl_port=30411)

        assert options.loadtime["nccl_port"] == 30411

    def test_a_profile_declaring_the_sanctioned_option_still_loads(self) -> None:
        config = ModelConfig(
            sie_id="tp-embedding",
            hf_id="org/model",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
            profiles={
                "default": ProfileConfig(
                    adapter_path="sie_server.adapters.sglang.embedding:SGLangEmbeddingAdapter",
                    max_batch_tokens=8192,
                    adapter_options=AdapterOptions(loadtime={"tensor_parallel_size": 2, "nccl_port": 30411}),
                )
            },
        )

        loadtime = config.profiles["default"].adapter_options.loadtime
        assert (loadtime["tensor_parallel_size"], loadtime["nccl_port"]) == (2, 30411)
