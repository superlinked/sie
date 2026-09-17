"""Config-time rules for a profile that declares a tensor-parallel width.

Each rule exists because the failure it prevents is silent. A placement flag
passed straight to the engine serves on devices the registry never reserved,
and a mistyped width is dropped rather than rejected, so the model serves on
one card while the profile declares several.
"""

from __future__ import annotations

from typing import Any

import pytest
from sie_server.config.model import AdapterOptions


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
