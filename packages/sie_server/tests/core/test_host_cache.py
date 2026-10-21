"""Host-RAM tier bookkeeping: settings, budget resolution and LRU admission."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from sie_server.core import host_cache as hc
from sie_server.core.host_cache import HostCache, ParkedModel

GIB = 1024**3
MIB = 1024**2


def _entry(name: str, host_bytes: int, *, device: str = "cuda:0", holds_device: bool = False) -> ParkedModel:
    return ParkedModel(
        name=name,
        loaded=SimpleNamespace(device=device),
        host_bytes=host_bytes,
        retains_device_memory=holds_device,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("auto", None),
        (" AUTO ", None),
        ("0", 0),
        (0, 0),
        (1048576, 1048576),
        ("1048576", 1048576),
        ("8GiB", 8 * GIB),
        ("8Gi", 8 * GIB),
        ("1.5GiB", int(1.5 * GIB)),
        ("512 MiB", 512 * MIB),
        ("2GB", 2 * 1000**3),
    ],
)
def test_parse_host_cache_setting(raw: object, expected: int | None) -> None:
    assert hc.parse_host_cache_setting(raw) == expected


@pytest.mark.parametrize("raw", ["", "lots", "1.5", "-1", "8XB", "GiB", -5, True, 2.5])
def test_parse_host_cache_setting_rejects_garbage(raw: object) -> None:
    with pytest.raises(ValueError, match=hc.HOST_CACHE_ENV):
        hc.parse_host_cache_setting(raw)


def test_auto_is_off_without_a_cuda_device() -> None:
    assert hc.resolve_host_cache_budget("auto", ["cpu"], memory_limit=64 * GIB) == 0
    assert hc.resolve_host_cache_budget("auto", ["mps"], memory_limit=64 * GIB) == 0


def test_auto_is_a_quarter_of_the_memory_limit_on_cuda() -> None:
    assert hc.resolve_host_cache_budget("auto", ["cuda:0"], memory_limit=32 * GIB) == 8 * GIB
    assert hc.resolve_host_cache_budget("auto", ["cuda:0", "cuda:1"], memory_limit=128 * GIB) == 32 * GIB


def test_explicit_size_applies_to_any_device() -> None:
    assert hc.resolve_host_cache_budget("1GiB", ["cpu"]) == GIB
    assert hc.resolve_host_cache_budget(0, ["cuda:0"]) == 0


def test_container_limit_prefers_cgroup_v2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hc, "physical_memory_bytes", lambda: 64 * GIB)
    v2 = tmp_path / "memory.max"
    v1 = tmp_path / "memory.limit_in_bytes"
    v2.write_text(f"{32 * GIB}\n")
    v1.write_text(f"{16 * GIB}\n")
    assert hc.container_memory_limit_bytes(cgroup_v2=v2, cgroup_v1=v1) == 32 * GIB


def test_container_limit_falls_back_to_v1_then_physical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hc, "physical_memory_bytes", lambda: 64 * GIB)
    v2 = tmp_path / "memory.max"
    v1 = tmp_path / "memory.limit_in_bytes"
    v2.write_text("max\n")
    v1.write_text(f"{16 * GIB}\n")
    assert hc.container_memory_limit_bytes(cgroup_v2=v2, cgroup_v1=v1) == 16 * GIB
    # cgroup v1 reports "unlimited" as a huge number.
    v1.write_text(f"{1 << 62}\n")
    assert hc.container_memory_limit_bytes(cgroup_v2=v2, cgroup_v1=v1) == 64 * GIB
    # A limit above the machine is capped at the machine.
    v2.write_text(f"{128 * GIB}\n")
    assert hc.container_memory_limit_bytes(cgroup_v2=v2, cgroup_v1=v1) == 64 * GIB


def test_host_cache_configured_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(hc.HOST_CACHE_ENV, raising=False)
    assert hc.host_cache_configured()
    monkeypatch.setenv(hc.HOST_CACHE_ENV, "0")
    assert not hc.host_cache_configured()
    monkeypatch.setenv(hc.HOST_CACHE_ENV, "4GiB")
    assert hc.host_cache_configured()
    monkeypatch.setenv(hc.HOST_CACHE_ENV, "garbage")
    assert not hc.host_cache_configured()


def test_disabled_cache_admits_nothing() -> None:
    cache = HostCache(0)
    assert not cache.enabled
    assert cache.admission_victims("a", MIB) is None


def test_admission_drops_least_recently_parked_first() -> None:
    cache = HostCache(100)
    cache.put(_entry("a", 40))
    cache.put(_entry("b", 40))
    assert cache.admission_victims("c", 20) == []
    assert cache.admission_victims("c", 50) == ["a"]
    assert cache.admission_victims("c", 100) == ["a", "b"]
    assert cache.admission_victims("c", 101) is None
    assert cache.used_bytes == 80


def test_retained_bytes_count_against_the_budget_but_are_never_victims() -> None:
    cache = HostCache(100)
    cache.retain("resident", 60)
    cache.put(_entry("a", 30))
    assert cache.used_bytes == 90
    assert cache.admission_victims("b", 40) == ["a"]
    assert cache.admission_victims("b", 41) is None
    # Parking the resident model replaces its retained charge.
    assert cache.admission_victims("resident", 70) == []
    cache.put(_entry("resident", 70))
    assert cache.retained_bytes("resident") == 0
    assert cache.used_bytes == 100


def test_take_removes_the_charge_and_release_forgets_retention() -> None:
    cache = HostCache(100)
    cache.put(_entry("a", 30))
    entry = cache.take("a")
    assert entry is not None
    assert entry.host_bytes == 30
    assert "a" not in cache
    assert cache.used_bytes == 0
    cache.retain("a", 30)
    assert cache.used_bytes == 30
    cache.release("a")
    assert cache.used_bytes == 0
    assert cache.take("a") is None


def test_holding_device_memory_filters_by_device_and_flag() -> None:
    cache = HostCache(1000)
    cache.put(_entry("torch", 10, device="cuda:0"))
    cache.put(_entry("engine-0", 10, device="cuda:0", holds_device=True))
    cache.put(_entry("engine-1", 10, device="cuda:1", holds_device=True))
    assert cache.holding_device_memory("cuda:0") == ["engine-0"]
    assert cache.holding_device_memory("cuda:1") == ["engine-1"]
    assert cache.names == ["torch", "engine-0", "engine-1"]
