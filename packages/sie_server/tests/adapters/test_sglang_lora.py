"""Runtime LoRA contracts without importing SGLang or allocating a GPU."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from safetensors.numpy import save_file
from sie_server.adapters._generation_base import GenerationInvalidRequestError, GenerationLoraLoadingError
from sie_server.adapters.lora import LoraStateUncertainError, LoraUnavailableError
from sie_server.adapters.sglang._lora import SGLangLoraManager, SGLangLoraMixin, leased_generation
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter


def manager(**kwargs):
    return SGLangLoraManager(
        kwargs.pop("paths", {"a": "org/a", "b": "org/b"}),
        preload=kwargs.pop("preload", []),
        revisions={"org/a": "a" * 40},
        max_rank=8,
        target_modules=["q_proj"],
        max_per_batch=kwargs.pop("max_per_batch", 2),
        server_url=lambda: "http://localhost:1234",
        mutation_timeout_s=0.5,
        **kwargs,
    )


@pytest.mark.parametrize("preload", ["a", ["a", "a"], ["unknown"], [1], {}])
def test_preload_selector_is_strict(preload):
    with pytest.raises(ValueError, match="preload_loras"):
        manager(preload=preload)


def test_deferred_engine_is_enabled_with_empty_preload():
    loras = manager()
    args = loras.launch_args()
    assert "--enable-lora" in args
    assert "--lora-paths" not in args
    assert args[args.index("--max-lora-rank") + 1] == "8"
    assert loras.runtime_config().preloaded == frozenset()


def test_omitted_selector_preloads_and_pins_existing_sources(monkeypatch):
    loras = manager(preload=None)
    monkeypatch.setattr(loras, "fetch", lambda source: "/cache/" + source.rsplit("/", 1)[1])
    args = loras.launch_args()
    assert args[args.index("--lora-paths") + 1 :] == ["a=/cache/a", "b=/cache/b"]
    loras.started()
    assert loras.available == ["a", "b"]
    with pytest.raises(ValueError, match="pinned"):
        loras.unload("a")


@pytest.mark.parametrize("explicit_preload", [False, True])
@pytest.mark.parametrize(
    ("count", "max_per_batch", "configured", "expected"),
    [(12, 2, None, 12), (2, 16, 3, 16), (2, 2, 1, 2)],
)
def test_startup_only_capacity_preserves_all_pinned_sources_and_batch_size(
    monkeypatch, explicit_preload, count, max_per_batch, configured, expected
):
    monkeypatch.setenv("SIE_MAX_LORAS_PER_MODEL", "10")
    paths = {f"adapter{i}": f"org/adapter{i}" for i in range(count)}
    loras = manager(
        paths=paths,
        preload=list(paths) if explicit_preload else None,
        max_loaded=configured,
        max_per_batch=max_per_batch,
    )
    monkeypatch.setattr(loras, "fetch", lambda source: "/cache/" + source.rsplit("/", 1)[1])
    args = loras.launch_args()
    assert args[args.index("--max-loaded-loras") + 1] == str(expected)
    assert args[args.index("--max-loras-per-batch") + 1] == str(max_per_batch)
    assert set(args[args.index("--lora-paths") + 1 :]) == {f"{alias}=/cache/{alias}" for alias in paths}
    config = loras.runtime_config()
    assert config.max_loaded == expected
    assert config.preloaded == config.pinned == frozenset(paths.values())


@pytest.mark.parametrize("preload", [[], ["a"]])
def test_deferred_capacity_remains_configured_even_with_more_declared_sources(preload):
    paths = {"a": "org/a", "b": "org/b", "c": "org/c", "d": "org/d"}
    loras = manager(paths=paths, preload=preload, max_loaded=3)
    assert loras.runtime_config().max_loaded == 3
    with pytest.raises(ValueError, match="must not exceed"):
        manager(paths=paths, preload=preload, max_loaded=3, max_per_batch=4)
    with pytest.raises(ValueError, match="must not exceed"):
        manager(paths=paths, preload=["a", "b", "c"], max_loaded=2)


@pytest.mark.parametrize("field", ["max_loaded", "max_per_batch"])
@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, "3"])
def test_startup_only_capacity_does_not_mask_invalid_limits(field, invalid):
    with pytest.raises(ValueError, match="positive"):
        manager(preload=None, **{field: invalid})


def test_undeclared_source_never_fetches_or_calls_engine(monkeypatch):
    fetch = Mock()
    post = Mock()
    monkeypatch.setattr("sie_server.adapters.sglang._lora.snapshot_download", fetch)
    monkeypatch.setattr("sie_server.adapters.sglang._lora.requests.post", post)
    loras = manager()
    for operation in (loras.fetch, loras.load, loras.unload):
        with pytest.raises(ValueError, match="not declared"):
            operation("attacker/arbitrary")
    fetch.assert_not_called()
    post.assert_not_called()


def test_fetch_uses_immutable_revision_and_only_adapter_files(tmp_path, monkeypatch):
    (tmp_path / "adapter_config.json").write_text('{"peft_type":"LORA","r":4,"target_modules":["q_proj"]}')
    save_file(
        {"base_model.model.q_proj.lora_A.weight": np.ones((4, 2), dtype=np.float32)},
        str(tmp_path / "adapter_model.safetensors"),
    )
    fetch = Mock(return_value=str(tmp_path))
    monkeypatch.setattr("sie_server.adapters.sglang._lora.snapshot_download", fetch)
    (tmp_path / "unvalidated.safetensors").write_bytes(b"must not reach the engine")
    loras = manager()
    snapshot = Path(loras.fetch("a"))
    assert snapshot != tmp_path
    assert sorted(path.name for path in snapshot.iterdir()) == ["adapter_config.json", "adapter_model.safetensors"]
    assert (snapshot / "adapter_model.safetensors").read_bytes() == (
        tmp_path / "adapter_model.safetensors"
    ).read_bytes()
    fetch.assert_called_once_with(
        "org/a", revision="a" * 40, allow_patterns=["adapter_config.json", "adapter_model.safetensors"]
    )
    with pytest.raises(ValueError, match="revision"):
        loras.fetch("a", "b" * 40)
    loras.close()
    assert not snapshot.exists()
    assert (tmp_path / "adapter_model.safetensors").exists()


@pytest.mark.parametrize(
    "override",
    [
        {"r": 16},
        {"target_modules": ["v_proj"]},
        {"rank_pattern": {"q_proj": 2}},
        {"alpha_pattern": {"q_proj": 2}},
        {"use_rslora": True},
        {"init_lora_weights": "pissa"},
    ],
)
def test_fetch_refuses_unsupported_transform_or_engine_capacity(tmp_path, monkeypatch, override):
    config = {"peft_type": "LORA", "r": 4, "target_modules": ["q_proj"], **override}
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    monkeypatch.setattr("sie_server.adapters.sglang._lora.snapshot_download", lambda *args, **kwargs: str(tmp_path))
    with pytest.raises(ValueError, match="LoRA"):
        manager().fetch("a")


def test_fetch_finishing_after_model_close_does_not_retain_source(tmp_path, monkeypatch):
    (tmp_path / "adapter_config.json").write_text('{"peft_type":"LORA","r":4,"target_modules":["q_proj"]}')
    save_file(
        {"base.q_proj.lora_A.weight": np.ones((4, 2), dtype=np.float32)}, str(tmp_path / "adapter_model.safetensors")
    )
    entered, release = threading.Event(), threading.Event()

    def download(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return str(tmp_path)

    monkeypatch.setattr("sie_server.adapters.sglang._lora.snapshot_download", download)
    loras = manager()
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(loras.fetch, "a")
        assert entered.wait(1)
        loras.close()
        release.set()
        with pytest.raises(RuntimeError, match="unloaded during fetch"):
            pending.result(timeout=1)
    assert not loras._snapshots


def test_discard_source_keeps_attached_or_uncertain_views_until_model_close(tmp_path, monkeypatch):
    (tmp_path / "adapter_config.json").write_text('{"peft_type":"LORA","r":4,"target_modules":["q_proj"]}')
    save_file(
        {"base.q_proj.lora_A.weight": np.ones((4, 2), dtype=np.float32)}, str(tmp_path / "adapter_model.safetensors")
    )
    monkeypatch.setattr("sie_server.adapters.sglang._lora.snapshot_download", lambda *args, **kwargs: str(tmp_path))
    loras = manager()
    unused = Path(loras.fetch("a"))
    loras.discard_source("a")
    assert not unused.exists()
    attached = Path(loras.fetch("a"))
    mutate = Mock()
    monkeypatch.setattr(loras, "_mutate", mutate)
    loras.load("a", source=str(attached))
    loras.discard_source("a")
    assert attached.exists()
    mutate.side_effect = TimeoutError()
    with pytest.raises(LoraStateUncertainError, match="removal"):
        loras.unload("a")
    loras.discard_source("a")
    assert attached.exists()
    loras.close()
    assert not attached.exists()


@pytest.mark.parametrize("alias", ["a=b", "a b", "--option", "a\nb"])
def test_served_name_cannot_be_interpreted_as_cli_syntax(alias):
    with pytest.raises(ValueError, match="served names"):
        SGLangLoraManager(
            {alias: "org/a"},
            preload=None,
            revisions=None,
            max_rank=None,
            target_modules=None,
            max_per_batch=1,
            server_url=lambda: None,
        )


def test_deferred_configuration_requires_an_evictable_slot():
    with pytest.raises(ValueError, match="at least one slot"):
        manager(preload=["a"], max_loaded=1, max_per_batch=1)


@pytest.mark.parametrize("flag", ["--tokenizer-worker-num", "--dp-size", "--data-parallel-size"])
@pytest.mark.parametrize("joined", [True, False])
def test_deferred_generation_requires_single_workers_but_allows_explicit_one(flag, joined):
    config = {
        "mem_fraction_static": 0.5,
        "lora_paths": {"a": "org/a"},
        "preload_loras": [],
        "max_lora_rank": 8,
        "lora_target_modules": ["q_proj"],
    }
    for count in ("1", "2"):
        args = [f"{flag}={count}"] if joined else [flag, count]
        if count == "1":
            SGLangGenerationAdapter("base", **config, extra_launch_args=args)
        else:
            with pytest.raises(ValueError, match="one tokenizer worker"):
                SGLangGenerationAdapter("base", **config, extra_launch_args=args)


def test_fetch_rejects_shared_base_tensor(tmp_path, monkeypatch):
    (tmp_path / "adapter_config.json").write_text('{"peft_type":"LORA","r":4,"target_modules":["q_proj"]}')
    save_file(
        {"base_model.model.q_proj.weight": np.ones((4, 2), dtype=np.float32)},
        str(tmp_path / "adapter_model.safetensors"),
    )
    monkeypatch.setattr("sie_server.adapters.sglang._lora.snapshot_download", lambda *args, **kwargs: str(tmp_path))
    with pytest.raises(ValueError, match="outside the adapter"):
        manager().fetch("a")


def test_failed_attach_rolls_back_or_keeps_uncertain_reservation(monkeypatch):
    loras = manager()
    mutate = Mock(side_effect=[RuntimeError("private source failure"), None])
    monkeypatch.setattr(loras, "_mutate", mutate)
    with pytest.raises(RuntimeError, match="private source"):
        loras.load("a", source="/cache/a")
    assert mutate.call_args_list[1].args == ("unload_lora_adapter", {"lora_name": "a"})
    assert loras.available == []
    mutate.side_effect = [TimeoutError(), TimeoutError()]
    with pytest.raises(LoraStateUncertainError, match="rollback"):
        loras.load("a", source="/cache/a")
    mutate.reset_mock(side_effect=True)
    with pytest.raises(LoraStateUncertainError, match="reload"):
        loras.load("a", source="/cache/a")
    mutate.assert_not_called()


def test_eviction_rejects_new_leases_and_drains_before_http(monkeypatch):
    loras = manager()
    mutate = Mock()
    monkeypatch.setattr(loras, "_mutate", mutate)
    loras.load("a", source="/cache/a")
    mutate.reset_mock()
    with ThreadPoolExecutor() as pool:
        with loras.lease("a"):
            future = pool.submit(loras.unload, "a")
            with loras._condition:
                assert loras._condition.wait_for(lambda: "org/a" in loras._draining, timeout=0.2)
            with pytest.raises(LoraUnavailableError):
                with loras.lease("a"):
                    pytest.fail("new lease admitted while evicting")
            mutate.assert_not_called()
        future.result(timeout=0.2)
    mutate.assert_called_once_with("unload_lora_adapter", {"lora_name": "a"})
    assert loras.available == []


def test_failed_unload_keeps_adapter_unavailable(monkeypatch):
    loras = manager()
    monkeypatch.setattr(loras, "_mutate", Mock())
    loras.load("a", source="/cache/a")
    monkeypatch.setattr(loras, "_mutate", Mock(side_effect=TimeoutError()))
    with pytest.raises(LoraStateUncertainError, match="removal"):
        loras.unload("a")
    assert "org/a" in loras._loaded
    with pytest.raises(LoraUnavailableError):
        with loras.lease("a"):
            pytest.fail("uncertain adapter admitted")


def test_teardown_joins_mutation_before_releasing_port_and_rejects_late_posts(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    response = Mock()
    response.json.return_value = {"success": True}

    def post(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return response

    http = Mock(side_effect=post)
    monkeypatch.setattr("sie_server.adapters.sglang._lora.requests.post", http)
    loras = manager()
    with ThreadPoolExecutor() as pool:
        mutation = pool.submit(loras._mutate, "unload_lora_adapter", {"lora_name": "a"})
        assert entered.wait(1)
        closing = pool.submit(loras.stop_mutations)
        with loras._condition:
            assert loras._condition.wait_for(lambda: loras._closed, timeout=1)
        assert not closing.done()
        # This models a mutation that passed the registry identity gate but
        # did not reach the child before unload began.
        late = pool.submit(loras._mutate, "unload_lora_adapter", {"lora_name": "b"})
        release.set()
        mutation.result(timeout=1)
        closing.result(timeout=1)
        with pytest.raises(RuntimeError, match="unloading"):
            late.result(timeout=1)
    assert http.call_count == 1


def test_teardown_refuses_new_leases_while_existing_lease_can_finish():
    loras = manager(preload=None)
    loras.started()
    with loras.lease("a"):
        loras.stop_mutations()
        with pytest.raises(LoraUnavailableError):
            with loras.lease("a"):
                pytest.fail("adapter admitted during model teardown")
        assert loras._leases["org/a"] == 1
    assert loras._leases["org/a"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["close", "cancel", "error", "complete"])
async def test_generation_lease_released_for_all_iterator_endings(ending, monkeypatch):
    loras = manager()
    monkeypatch.setattr(loras, "_mutate", Mock())
    loras.load("a", source="/cache/a")
    gate = asyncio.Event()
    closed = threading.Event()

    class Adapter(SGLangLoraMixin):
        _lora = loras

        @leased_generation
        async def generate(self, *, lora_path=None):
            assert lora_path == "a"
            try:
                yield "first"
                if ending == "error":
                    raise RuntimeError("inference failed")
                await gate.wait()
            finally:
                closed.set()

    stream = Adapter().generate(lora_path="org/a")
    assert await anext(stream) == "first"
    assert loras._leases["org/a"] == 1
    if ending == "close":
        await stream.aclose()
    elif ending == "cancel":
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif ending == "error":
        with pytest.raises(RuntimeError, match="inference failed"):
            await anext(stream)
    else:
        gate.set()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
    assert loras._leases["org/a"] == 0
    assert closed.is_set()


@pytest.mark.asyncio
async def test_generation_missing_lora_and_unsafe_sampling_are_typed():
    class Adapter(SGLangLoraMixin):
        _lora = manager()

        @leased_generation
        async def generate(self, **kwargs):
            yield kwargs

    with pytest.raises(GenerationLoraLoadingError):
        await anext(Adapter().generate(lora_path="a"))
    with pytest.raises(GenerationInvalidRequestError):
        await anext(Adapter().generate(lora_path="a", n=2))
