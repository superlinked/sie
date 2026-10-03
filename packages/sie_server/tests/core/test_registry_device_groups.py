"""A declared tensor-parallel width is an exclusive claim on whole devices.

The registry has always placed one model on one device and kept the right to
evict it. A width above one changes that for the models that declare it: the
load owns every card in its group until it unloads, because the engine reserves
the large majority of each card it is given and the headroom left over is what
graph capture needs for the group to start at all.

These tests pin the parts a single-device deployment must never notice, and the
parts a multi-device one depends on.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_server.config.model import AdapterOptions, EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.load_errors import DevicePlacementError, LoadErrorClass
from sie_server.core.registry import ModelRegistry, declared_tensor_parallel_size


def _make_config(
    name: str = "test", *, width: int | None = None, extra_loadtime: dict[str, object] | None = None
) -> ModelConfig:
    loadtime: dict[str, object] = dict(extra_loadtime or {})
    if width is not None:
        loadtime["tensor_parallel_size"] = width
        if width > 1:
            # A width above one requires a finite streaming read cap.
            loadtime["request_read_timeout_s"] = 120.0
    return ModelConfig(
        sie_id=name,
        hf_id=f"org/{name}",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
        profiles={
            "default": ProfileConfig(
                # An adapter that accepts a width: the registry validates the
                # profile's load-time options against it before placing a group.
                adapter_path="sie_server.adapters.sglang.embedding:SGLangEmbeddingAdapter",
                max_batch_tokens=8192,
                adapter_options=AdapterOptions(loadtime=loadtime),
            )
        },
    )


@pytest.fixture(autouse=True)
def patch_ensure_model_cached():
    with patch("sie_sdk.cache.ensure_model_cached") as mock:
        mock.return_value = Path("/fake/cache/models--org--test")
        yield mock


def _registry(devices: list[str]) -> ModelRegistry:
    registry = ModelRegistry(device="cuda", devices=devices)
    for manager in registry.memory_managers.values():
        manager.check_pressure = MagicMock(return_value=False)  # type: ignore[method-assign]
    return registry


def _pinned_registry(devices: list[str], *, pinned: list[str]) -> ModelRegistry:
    registry = ModelRegistry(device="cuda", devices=devices, pinned_models=pinned)
    for manager in registry.memory_managers.values():
        manager.check_pressure = MagicMock(return_value=False)  # type: ignore[method-assign]
    return registry


def _adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.capabilities.outputs = ["dense"]
    adapter.memory_footprint.return_value = 1000
    return adapter


class TestDeclaredWidth:
    def test_absent_declaration_is_width_one(self) -> None:
        assert declared_tensor_parallel_size(_make_config()) == 1

    def test_declared_width_is_read_from_the_default_profile(self) -> None:
        assert declared_tensor_parallel_size(_make_config(width=4)) == 4

    def test_malformed_width_is_refused_rather_than_read_as_one(self) -> None:
        """A width that silently read as one would serve on a single card.

        The operator would believe several were in use while the model ran on
        one of the cards reserved for it.
        """
        with pytest.raises(ValueError, match="tensor_parallel_size"):
            declared_tensor_parallel_size(_make_config(width=0))


class TestExclusiveClaim:
    @patch("sie_server.core.model_loader.load_adapter")
    async def test_width_four_claims_every_member_of_its_group(self, mock_load_adapter: MagicMock) -> None:
        registry = _registry(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        registry.add_config(_make_config("wide", width=4))
        mock_load_adapter.return_value = _adapter()

        await registry.load_async("wide", "cuda")

        assert registry._device_claims == {
            "cuda:0": "wide",
            "cuda:1": "wide",
            "cuda:2": "wide",
            "cuda:3": "wide",
        }
        assert registry._loaded["wide"].device == "cuda:0"

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_the_claim_log_names_only_the_loading_models_devices(
        self, mock_load_adapter: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With two groups resident, the second claim must not report the first group's cards."""
        registry = _registry(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        registry.add_config(_make_config("first", width=2))
        registry.add_config(_make_config("second", width=2))
        mock_load_adapter.return_value = _adapter()
        await registry.load_async("first", "cuda:0")
        caplog.clear()

        with caplog.at_level("INFO", logger="sie_server.core.registry"):
            await registry.load_async("second", "cuda:2")

        claims = [record.getMessage() for record in caplog.records if "claimed devices" in record.getMessage()]
        assert claims == ["Model 'second' claimed devices ['cuda:2', 'cuda:3'] exclusively"]

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_single_device_model_never_claims_anything(self, mock_load_adapter: MagicMock) -> None:
        """The path every existing profile takes must be untouched."""
        registry = _registry(["cuda:0", "cuda:1"])
        registry.add_config(_make_config("ordinary"))
        mock_load_adapter.return_value = _adapter()

        await registry.load_async("ordinary", "cuda")

        assert registry._device_claims == {}

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_claimed_devices_are_released_on_unload(self, mock_load_adapter: MagicMock) -> None:
        registry = _registry(["cuda:0", "cuda:1"])
        registry.add_config(_make_config("wide", width=2))
        mock_load_adapter.return_value = _adapter()

        await registry.load_async("wide", "cuda")
        assert registry._device_claims
        await registry.unload_async("wide")

        assert registry._device_claims == {}

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_width_that_does_not_fit_fails_with_a_named_reason(self, mock_load_adapter: MagicMock) -> None:
        """Serving on fewer cards than declared is the failure to avoid.

        A width-four model quietly serving on one card holds four accelerators
        and uses one.
        """
        registry = _registry(["cuda:0", "cuda:1"])
        registry.add_config(_make_config("too-wide", width=4))
        mock_load_adapter.return_value = _adapter()

        with pytest.raises(RuntimeError, match="tensor_parallel_size=4"):
            await registry.load_async("too-wide", "cuda")

        assert registry._device_claims == {}

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_failed_load_strands_no_claim(self, mock_load_adapter: MagicMock) -> None:
        registry = _registry(["cuda:0", "cuda:1"])
        registry.add_config(_make_config("wide", width=2))
        mock_load_adapter.side_effect = RuntimeError("engine refused to start")

        with pytest.raises(RuntimeError, match="engine refused"):
            await registry.load_async("wide", "cuda")

        assert registry._device_claims == {}

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_group_does_not_land_on_a_device_another_model_occupies(self, mock_load_adapter: MagicMock) -> None:
        registry = _registry(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        registry.add_config(_make_config("resident"))
        registry.add_config(_make_config("wide", width=2))
        mock_load_adapter.side_effect = [_adapter(), _adapter()]

        await registry.load_async("resident", "cuda:0")
        await registry.load_async("wide", "cuda")

        assert registry._loaded["resident"].device == "cuda:0"
        # Which free block it picks is not the contract. Not overlapping the
        # resident is.
        claimed = set(registry._device_claims)
        assert len(claimed) == 2
        assert "cuda:0" not in claimed

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_naming_a_device_an_idle_group_holds_displaces_the_group(self, mock_load_adapter: MagicMock) -> None:
        """Naming a group member must never put a second model onto it while held.

        The group has sized its cache against the whole card, so the model is
        placed only after the group has gone, never beside it.
        """
        registry = _registry(["cuda:0", "cuda:1", "cuda:2"])
        registry.add_config(_make_config("wide", width=2))
        registry.add_config(_make_config("intruder"))
        mock_load_adapter.side_effect = [_adapter(), _adapter()]

        await registry.load_async("wide", "cuda")
        assert set(registry._device_claims) == {"cuda:0", "cuda:1"}

        await registry.load_async("intruder", "cuda:1")

        assert "wide" not in registry._loaded
        assert registry._device_claims == {}
        assert registry._loaded["intruder"].device == "cuda:1"

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_naming_a_device_a_pinned_group_holds_is_refused(self, mock_load_adapter: MagicMock) -> None:
        registry = _pinned_registry(["cuda:0", "cuda:1", "cuda:2"], pinned=["Org/Wide"])
        registry.add_config(_make_config("Org/Wide", width=2))
        registry.add_config(_make_config("intruder"))
        mock_load_adapter.side_effect = [_adapter(), _adapter()]

        await registry.load_async("Org/Wide", "cuda")

        with pytest.raises(DevicePlacementError, match="held exclusively by multi-device model 'Org/Wide'"):
            await registry.load_async("intruder", "cuda:1")

        assert "Org/Wide" in registry._loaded
        assert "intruder" not in registry._loaded

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_an_unclaimed_device_can_still_be_named(self, mock_load_adapter: MagicMock) -> None:
        registry = _registry(["cuda:0", "cuda:1", "cuda:2"])
        registry.add_config(_make_config("wide", width=2))
        registry.add_config(_make_config("ordinary"))
        mock_load_adapter.side_effect = [_adapter(), _adapter()]

        await registry.load_async("wide", "cuda")
        await registry.load_async("ordinary", "cuda:2")

        assert registry._loaded["ordinary"].device == "cuda:2"

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_fully_claimed_worker_displaces_an_idle_group_rather_than_overlapping(
        self, mock_load_adapter: MagicMock
    ) -> None:
        """With every card held, the group goes before anything else is placed.

        Refusing instead would leave every other model on the worker unloadable
        for as long as the group stayed resident, and nothing else unloads it.
        """
        registry = _registry(["cuda:0", "cuda:1"])
        registry.add_config(_make_config("wide", width=2))
        registry.add_config(_make_config("ordinary"))
        mock_load_adapter.side_effect = [_adapter(), _adapter()]

        await registry.load_async("wide", "cuda")
        await registry.load_async("ordinary", "cuda")

        assert "wide" not in registry._loaded
        assert registry._device_claims == {}
        assert "ordinary" in registry._loaded

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_fully_claimed_worker_refuses_when_the_group_is_pinned(self, mock_load_adapter: MagicMock) -> None:
        registry = _pinned_registry(["cuda:0", "cuda:1"], pinned=["Org/Wide"])
        registry.add_config(_make_config("Org/Wide", width=2))
        registry.add_config(_make_config("ordinary"))
        mock_load_adapter.side_effect = [_adapter(), _adapter()]

        await registry.load_async("Org/Wide", "cuda")

        with pytest.raises(DevicePlacementError, match="exclusively"):
            await registry.load_async("ordinary", "cuda")

        assert "ordinary" not in registry._loaded

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_single_device_model_avoids_a_claimed_device(self, mock_load_adapter: MagicMock) -> None:
        registry = _registry(["cuda:0", "cuda:1", "cuda:2"])
        registry.add_config(_make_config("wide", width=2))
        registry.add_config(_make_config("ordinary"))
        mock_load_adapter.side_effect = [_adapter(), _adapter()]

        await registry.load_async("wide", "cuda")
        await registry.load_async("ordinary", "cuda")

        assert set(registry._device_claims) == {"cuda:0", "cuda:1"}
        assert registry._loaded["ordinary"].device == "cuda:2"


class TestGroupMemoryAccounting:
    """A width-N load reserves memory on N cards, so N managers must know.

    Registering only the anchor leaves the other members looking free, so a
    later placement sizes itself against memory this model already holds and
    the eviction that would have prevented the collision never fires.
    """

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_every_member_accounts_for_the_load(self, mock_load_adapter: MagicMock) -> None:
        registry = _registry(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        registry.add_config(_make_config("wide", width=4))
        mock_load_adapter.return_value = _adapter()

        await registry.load_async("wide", "cuda")

        for device in ("cuda:0", "cuda:1", "cuda:2", "cuda:3"):
            assert "wide" in registry.memory_managers[device].loaded_models, device

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_the_group_total_matches_a_single_device_load(self, mock_load_adapter: MagicMock) -> None:
        """Sharding divides the weights, so the shares must divide too."""
        registry = _registry(["cuda:0", "cuda:1"])
        registry.add_config(_make_config("wide", width=2))
        adapter = _adapter()
        adapter.memory_footprint.return_value = 1000
        mock_load_adapter.return_value = adapter

        await registry.load_async("wide", "cuda")

        shares = [registry.memory_managers[device]._models["wide"].estimated_bytes for device in ("cuda:0", "cuda:1")]
        total = registry._loaded["wide"].memory_bytes

        assert all(share == total // 2 for share in shares)
        # Integer division may drop a remainder; the group must never claim
        # more than a single-device load of the same model would have.
        assert sum(shares) <= total

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_unload_clears_every_member(self, mock_load_adapter: MagicMock) -> None:
        registry = _registry(["cuda:0", "cuda:1", "cuda:2"])
        registry.add_config(_make_config("wide", width=2))
        mock_load_adapter.return_value = _adapter()

        await registry.load_async("wide", "cuda")
        await registry.unload_async("wide")

        for device in ("cuda:0", "cuda:1", "cuda:2"):
            assert "wide" not in registry.memory_managers[device].loaded_models, device
        assert registry._device_claims == {}

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_single_device_load_registers_exactly_where_it_did_before(
        self, mock_load_adapter: MagicMock
    ) -> None:
        """The path every existing profile takes must be untouched."""
        registry = _registry(["cuda:0", "cuda:1"])
        registry.add_config(_make_config("ordinary"))
        mock_load_adapter.return_value = _adapter()

        await registry.load_async("ordinary", "cuda")

        anchor = registry._loaded["ordinary"].device
        assert "ordinary" in registry.memory_managers[anchor].loaded_models
        other = "cuda:1" if anchor == "cuda:0" else "cuda:0"
        assert "ordinary" not in registry.memory_managers[other].loaded_models


class TestWidthAgainstVisibleDevices:
    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_width_this_worker_can_never_serve_says_so(self, mock_load_adapter: MagicMock) -> None:
        """Distinct from 'no free block right now', which is worth retrying."""
        registry = _registry(["cuda:0", "cuda:1"])
        registry.add_config(_make_config("too-wide", width=4))
        mock_load_adapter.return_value = _adapter()

        with pytest.raises(RuntimeError, match="sees 2 device"):
            await registry.load_async("too-wide", "cuda")

        assert registry._device_claims == {}

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_width_equal_to_the_device_count_is_servable(self, mock_load_adapter: MagicMock) -> None:
        registry = _registry(["cuda:0", "cuda:1"])
        registry.add_config(_make_config("exact", width=2))
        mock_load_adapter.return_value = _adapter()

        await registry.load_async("exact", "cuda")

        assert set(registry._device_claims) == {"cuda:0", "cuda:1"}


class TestTheRegistryAndTheAdapterAgree:
    """The registry reserves cards and the adapter masks cards, separately.

    They are kept in step by both deriving the group from the same anchor and
    width through one shared function rather than by passing a list. This pins
    that, because a silent disagreement would have the registry accounting for
    one set of cards while the engine ran on another.
    """

    @pytest.mark.parametrize("width", [1, 2, 4, 8])
    def test_the_claimed_devices_are_the_masked_devices(self, width: int) -> None:
        from sie_server.adapters.sglang._server import format_device_mask, parse_device_index
        from sie_server.config.device_groups import resolve_device_group

        registry = _registry([f"cuda:{index}" for index in range(width)])

        claimed = registry._group_members("cuda:0", width)
        assert claimed is not None

        masked = format_device_mask(resolve_device_group(parse_device_index("cuda:0"), width))

        assert masked == ",".join(str(index) for index in range(width))
        assert claimed == [f"cuda:{index}" for index in range(width)]

    def test_a_group_that_runs_past_the_device_list_is_not_claimable(self) -> None:
        registry = _registry(["cuda:0", "cuda:1"])

        assert registry._group_members("cuda:1", 2) is None

    def test_an_anchor_without_a_concrete_index_is_single_device_only(self) -> None:
        """A bare 'cuda' names no card, so it cannot anchor a group."""
        registry = _registry(["cuda"])

        assert registry._group_members("cuda", 1) == ["cuda"]
        assert registry._group_members("cuda", 2) is None


class TestGroupPlacementEvicts:
    """An occupied block is not an unavailable one.

    The registry is lazy-load with LRU eviction. If a group load refused to
    evict, one request for any sibling model would keep the wide model from
    ever loading again, and nothing would unload that sibling afterwards.
    """

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_an_idle_neighbour_is_evicted_to_free_the_group(self, mock_load_adapter: MagicMock) -> None:
        registry = _registry(["cuda:0", "cuda:1"])
        registry.add_config(_make_config("small"))
        registry.add_config(_make_config("wide", width=2))
        mock_load_adapter.return_value = _adapter()

        await registry.load_async("small", "cuda:0")
        assert "small" in registry._loaded

        await registry.load_async("wide", "cuda")

        assert "small" not in registry._loaded
        assert registry._device_claims == {"cuda:0": "wide", "cuda:1": "wide"}

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_free_block_is_preferred_over_evicting_one(self, mock_load_adapter: MagicMock) -> None:
        registry = _registry(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        registry.add_config(_make_config("small"))
        registry.add_config(_make_config("wide", width=2))
        mock_load_adapter.return_value = _adapter()

        await registry.load_async("small", "cuda:0")
        await registry.load_async("wide", "cuda")

        assert "small" in registry._loaded
        claimed = sorted(registry._device_claims)
        assert len(claimed) == 2
        assert "cuda:0" not in claimed

    @pytest.mark.parametrize(
        ("idle", "busy", "freed"),
        [("first", "second", ["cuda:0", "cuda:1"]), ("second", "first", ["cuda:2", "cuda:3"])],
    )
    @patch("sie_server.core.model_loader.load_adapter")
    async def test_between_equal_blocks_the_least_recently_used_group_is_evicted(
        self, mock_load_adapter: MagicMock, idle: str, busy: str, freed: list[str]
    ) -> None:
        """Device order must not decide which of two resident groups goes.

        Evicting the busy one would leave the idle group holding its cards while
        the busy model and the newcomer take turns evicting each other.
        """
        registry = _registry(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        for name in ("first", "second", "third"):
            registry.add_config(_make_config(name, width=2))
        mock_load_adapter.side_effect = [_adapter(), _adapter(), _adapter()]

        await registry.load_async("first", "cuda")
        await registry.load_async("second", "cuda")
        assert registry._device_claims == {"cuda:0": "first", "cuda:1": "first", "cuda:2": "second", "cuda:3": "second"}
        idle_info = registry._memory_manager_for_model(idle).get_model_info(idle)
        assert idle_info is not None
        idle_info.last_used_at = time.monotonic() - 100.0

        await registry.load_async("third", "cuda")

        assert idle not in registry._loaded
        assert busy in registry._loaded
        assert sorted(device for device, holder in registry._device_claims.items() if holder == "third") == freed

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_pinned_neighbour_is_never_evicted(self, mock_load_adapter: MagicMock) -> None:
        registry = ModelRegistry(device="cuda", devices=["cuda:0", "cuda:1"], pinned_models=["small"])
        for manager in registry.memory_managers.values():
            manager.check_pressure = MagicMock(return_value=False)  # type: ignore[method-assign]
        registry.add_config(_make_config("small"))
        registry.add_config(_make_config("wide", width=2))
        mock_load_adapter.return_value = _adapter()

        await registry.load_async("small", "cuda:0")

        with pytest.raises(RuntimeError, match="can be freed"):
            await registry.load_async("wide", "cuda")

        assert "small" in registry._loaded
        assert registry._device_claims == {}


def _rescanning_registry(devices: list[str], configs: list[dict[str, ModelConfig]]) -> ModelRegistry:
    """A registry whose on-disk catalog changes between scans, as a config update drops an entry."""
    with patch("sie_server.core.registry.load_model_configs", side_effect=configs):
        registry = ModelRegistry(models_dir=Path("/fake/models"), device="cuda", devices=devices)
    for manager in registry.memory_managers.values():
        manager.check_pressure = MagicMock(return_value=False)  # type: ignore[method-assign]
    return registry


class TestConfigDropDoesNotStrandAGroup:
    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_resident_whose_config_was_rescanned_away_can_still_be_unloaded(
        self, mock_load_adapter: MagicMock
    ) -> None:
        """A rescan replaces the whole catalog while the model stays loaded and claimed."""
        wide = _make_config("wide", width=2)
        registry = _rescanning_registry(["cuda:0", "cuda:1"], [{"wide": wide}])
        mock_load_adapter.return_value = _adapter()

        await registry.load_async("wide", "cuda")
        assert sorted(registry._device_claims) == ["cuda:0", "cuda:1"]

        with patch("sie_server.core.registry.load_model_configs", return_value={}):
            registry.rescan_configs()
        assert not registry.has_model("wide")

        await registry.unload_async("wide")

        assert registry._device_claims == {}
        assert "wide" not in registry._loaded

    async def test_an_unknown_name_is_still_a_key_error(self) -> None:
        registry = _registry(["cuda:0"])

        with pytest.raises(KeyError):
            await registry.unload_async("never-existed")


@patch("sie_server.core.model_loader.load_adapter")
def test_the_synchronous_unload_also_reaches_a_rescanned_away_resident(mock_load_adapter: MagicMock) -> None:
    wide = _make_config("wide", width=2)
    registry = _rescanning_registry(["cuda:0", "cuda:1"], [{"wide": wide}])
    mock_load_adapter.return_value = _adapter()

    registry.load("wide", "cuda")
    with patch("sie_server.core.registry.load_model_configs", return_value={}):
        registry.rescan_configs()

    registry.unload("wide")

    assert registry._device_claims == {}
    assert "wide" not in registry._loaded


class TestPlacementUnderContention:
    """The shapes the first review of eviction missed, through the load paths workers use."""

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_second_wide_model_displaces_the_first(self, mock_load_adapter: MagicMock) -> None:
        """With every card claimed, the only way a group can move is if another group can displace it."""
        registry = _registry(["cuda:0", "cuda:1"])
        registry.add_config(_make_config("wide-a", width=2))
        registry.add_config(_make_config("wide-b", width=2))
        mock_load_adapter.side_effect = [_adapter(), _adapter()]

        await registry._load_model_background("wide-a", "cuda")
        await registry._load_model_background("wide-b", "cuda")

        assert "wide-a" not in registry._loaded
        assert registry._device_claims == {"cuda:0": "wide-b", "cuda:1": "wide-b"}
        assert not registry.is_failed("wide-b")

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_placement_conflict_is_retryable_and_clears_on_unload(self, mock_load_adapter: MagicMock) -> None:
        """Recorded as an unclassified failure it stayed sticky after the cards came free."""
        registry = _pinned_registry(["cuda:0", "cuda:1"], pinned=["Org/Wide"])
        registry.add_config(_make_config("Org/Wide", width=2))
        registry.add_config(_make_config("small"))
        mock_load_adapter.side_effect = [_adapter(), _adapter()]

        await registry._load_model_background("Org/Wide", "cuda")
        await registry._load_model_background("small", "cuda")

        failure = registry.get_failure("small")
        assert failure is not None
        assert failure.error_class is LoadErrorClass.PLACEMENT
        assert not failure.is_permanent

        await registry.unload_async("Org/Wide")

        assert not registry.is_failed("small")

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_pinned_neighbour_with_a_hub_style_id_is_never_evicted(self, mock_load_adapter: MagicMock) -> None:
        """The pinned set is lowercased; loaded names keep their case."""
        registry = _pinned_registry(["cuda:0", "cuda:1"], pinned=["BAAI/bge-m3"])
        registry.add_config(_make_config("BAAI/bge-m3"))
        registry.add_config(_make_config("Org/Wide", width=2))
        mock_load_adapter.side_effect = [_adapter(), _adapter()]

        await registry.load_async("BAAI/bge-m3", "cuda:0")

        with pytest.raises(DevicePlacementError, match="can be freed"):
            await registry.load_async("Org/Wide", "cuda")

        assert "BAAI/bge-m3" in registry._loaded
        assert registry._device_claims == {}

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_a_profile_its_adapter_rejects_evicts_nothing(self, mock_load_adapter: MagicMock) -> None:
        """A typo must fail before a group evicts working models to make room for it."""
        registry = _registry(["cuda:0", "cuda:1"])
        registry.add_config(_make_config("small"))
        registry.add_config(_make_config("wide", width=2, extra_loadtime={"watchdog_timeout": 90}))
        mock_load_adapter.side_effect = [_adapter(), _adapter()]

        await registry.load_async("small", "cuda:0")

        with pytest.raises(ValueError, match="does not accept load-time option"):
            await registry.load_async("wide", "cuda")

        assert "small" in registry._loaded
        assert registry._device_claims == {}

    async def test_an_unservable_width_is_refused_before_any_download(self) -> None:
        registry = _registry(["cuda:0", "cuda:1"])
        registry.add_config(_make_config("too-wide", width=4))
        registry._loader.ensure_weights_cached_async = AsyncMock()  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="sees 2 device"):
            await registry.load_async("too-wide", "cuda")

        registry._loader.ensure_weights_cached_async.assert_not_awaited()
