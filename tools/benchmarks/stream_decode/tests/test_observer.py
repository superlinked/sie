import asyncio
import contextlib
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_sdk import SIEClient
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.api import openai_local

from tools.benchmarks.stream_decode import observer
from tools.benchmarks.stream_decode.analysis import analyze_trial


def manager_type():
    class Manager:
        model_path = "public/checkpoint"
        served_model_name = "engine-model"
        incremental_streaming_output = False

        def __init__(self):
            self.closed = []
            self.received = []

        async def _tokenize_one_request(self, obj):
            return SimpleNamespace(
                input_ids=list(range(10)), sampling_params=SimpleNamespace(top_k=1), rid=obj.rid, mm_inputs=None
            )

        async def generate_request(self, obj, request=None):
            self.received.append((obj, request))
            try:
                await self._tokenize_one_request(obj)
                for tokens, text in ((1, "a"), (3, "abc"), (4, "abcd")):
                    yield {
                        "text": text,
                        "meta_info": {
                            "id": obj.rid,
                            "prompt_tokens": 10,
                            "completion_tokens": tokens,
                            "finish_reason": {"type": "length"} if tokens == 4 else None,
                        },
                    }
                    await asyncio.sleep(0)
            finally:
                self.closed.append(obj.rid)

    return Manager


def request(body):
    async def as_json():
        return body

    return SimpleNamespace(url=SimpleNamespace(path="/v1/chat/completions"), json=as_json)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "streaming", "early"),
    [("/generate", False, False), (None, False, False), ("/health_generate", True, True)],
)
async def test_startup_and_health_generation_pass_through_without_payload_capture(tmp_path, route, streaming, early):
    cls = manager_type()
    restore = observer.patch_manager(SimpleNamespace(TokenizerManager=cls), tmp_path, {"cohort": "startup"})
    manager = cls()
    obj = SimpleNamespace(rid="warmup", stream=streaming, sampling_params={})
    raw = SimpleNamespace(url=SimpleNamespace(path=route)) if route is not None else None
    values = []
    try:
        async with contextlib.aclosing(manager.generate_request(obj, raw)) as generated:
            async for event in generated:
                values.append(event)
                if early:
                    break
    finally:
        restore()
    assert len(values) == (1 if early else 3)
    assert manager.closed == ["warmup"]
    assert manager.received == [(obj, raw)]
    assert not list(tmp_path.glob("stream-*.json"))
    paths = list(tmp_path.glob("background-*.json"))
    assert len(paths) == 1
    record = json.loads(paths[0].read_text())
    assert record["completed"] is (not early)
    assert record["route"] == route
    assert record["finished_ns"] > record["started_ns"]
    assert not {"chat_body", "tokenized", "events"} & record.keys()


@pytest.mark.asyncio
async def test_concurrent_stream_records_and_early_close_preserve_original_generator(tmp_path):
    cls = manager_type()
    original = (cls.generate_request, cls._tokenize_one_request)
    restore = observer.patch_manager(SimpleNamespace(TokenizerManager=cls), tmp_path, {"cohort": "test"})
    manager = cls()

    async def consume(rid, early):
        obj = SimpleNamespace(rid=rid, stream=True, sampling_params={"top_k": 1})
        raw = request({"model": "engine-model", "messages": [{"role": "user", "content": rid}]})
        values = []
        async with contextlib.aclosing(manager.generate_request(obj, raw)) as stream:
            async for value in stream:
                values.append(value)
                if early:
                    break
        assert (obj, raw) in manager.received
        return values

    try:
        complete, cancelled = await asyncio.gather(consume("full", False), consume("early", True))
    finally:
        restore()
    assert len(complete) == 3
    assert len(cancelled) == 1
    records = {
        row["tokenized"]["rid"]: row for row in (json.loads(p.read_text()) for p in tmp_path.glob("stream-*.json"))
    }
    assert records["full"]["completed"] is True
    assert records["early"]["completed"] is False
    assert records["early"]["error"]["type"] == "GeneratorExit"
    assert sorted(manager.closed) == ["early", "full"]
    assert (cls.generate_request, cls._tokenize_one_request) == original
    assert not hasattr(cls, observer.MARKER)


@pytest.mark.asyncio
async def test_chat_occupancy_is_visible_while_stream_is_open(tmp_path):
    cls = manager_type()
    restore = observer.patch_manager(SimpleNamespace(TokenizerManager=cls), tmp_path, {"cohort": "active"})
    obj = SimpleNamespace(rid="active-chat", stream=True, sampling_params={})
    manager = cls()
    try:
        async with contextlib.aclosing(manager.generate_request(obj, request({}))) as stream:
            await anext(stream)
            paths = list(tmp_path.glob("stream-*.json"))
            assert len(paths) == 1
            active = json.loads(paths[0].read_text())
            assert active["completed"] is False
            assert active["finished_ns"] is None
            assert active["events"] == []
        closed = json.loads(paths[0].read_text())
        assert closed["id"] == active["id"]
        assert closed["finished_ns"] > active["started_ns"]
        assert len(closed["events"]) == 1
        assert closed["error"]["type"] == "GeneratorExit"
    finally:
        restore()


def test_real_sie_cuda_chat_route_and_sdk_reach_child_observer(monkeypatch, tmp_path):
    """A native adapter.generate hook cannot make this integration pass."""
    monkeypatch.setattr(observer, "clock_domain", lambda: "test-boot")
    cls = manager_type()
    restore = observer.patch_manager(SimpleNamespace(TokenizerManager=cls), tmp_path, {"cohort": "route"})
    manager = cls()
    adapter = SGLangGenerationAdapter("public/checkpoint", served_model_name="engine-model")
    adapter._server_url = "http://127.0.0.1:30400"
    adapter.generate = MagicMock(side_effect=AssertionError("native route must not be used"))
    config = SimpleNamespace(
        tasks=SimpleNamespace(
            generate=SimpleNamespace(max_output_tokens=64, chat_template_kwargs={"enable_thinking": False}), score=None
        ),
        resolve_profile=MagicMock(
            return_value=SimpleNamespace(runtime={"default_sampling": {}, "stop_tokens": []}, loadtime={})
        ),
        inputs=SimpleNamespace(video=False),
        profiles={},
        routing=None,
    )
    registry = MagicMock()
    registry.device = "cuda:0"
    registry.has_model.return_value = registry.is_loaded.return_value = True
    registry.is_loading.return_value = registry.is_unloading.return_value = registry.is_failed.return_value = False
    registry.get_config.return_value = config
    registry.get.return_value = adapter

    class Stream(httpx.AsyncByteStream):
        def __init__(self, body):
            self.body = body

        async def __aiter__(self):
            obj = SimpleNamespace(rid="engine-request", stream=True, sampling_params={"top_k": 1})
            prior = ""
            async with contextlib.aclosing(manager.generate_request(obj, request(self.body))) as events:
                async for event in events:
                    meta = event["meta_info"]
                    finish = meta["finish_reason"]
                    chunk = {
                        "id": meta["id"],
                        "model": "engine-model",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": event["text"][len(prior) :]},
                                "finish_reason": finish["type"] if finish else None,
                            }
                        ],
                    }
                    prior = event["text"]
                    yield ("data: " + json.dumps(chunk) + "\n\n").encode()
            usage = {
                "id": "engine-request",
                "model": "engine-model",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            }
            yield ("data: " + json.dumps(usage) + "\n\ndata: [DONE]\n\n").encode()

    async def child(req):
        assert req.url.path == "/v1/chat/completions"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Stream(json.loads(req.content))
        )

    monkeypatch.setattr(
        openai_local,
        "_new_proxy_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(child), trust_env=False),
    )
    app = FastAPI()
    app.include_router(openai_local.router)
    app.state.registry = registry
    record = {"completed": False, "clock_domain": "test-boot", "http_requests": 1, "chunks": []}
    try:
        with (
            TestClient(app, base_url="http://127.0.0.1", follow_redirects=False) as http,
            SIEClient("http://127.0.0.1", remote="forbid", http_client=http) as sdk,
        ):
            record["started_ns"] = time.perf_counter_ns()
            for chunk in sdk.stream_chat_completions(
                "sie-model",
                [{"role": "user", "content": "public fixed prompt"}],
                max_completion_tokens=4,
                stream_options={"include_usage": True},
                wait_for_capacity=False,
                max_oom_retries=0,
            ):
                record["chunks"].append({"at_ns": time.perf_counter_ns(), "chunk": chunk})
            record.update(finished_ns=time.perf_counter_ns(), completed=True, retry_count=sdk.last_retry_count)
    finally:
        restore()
    traces = list(tmp_path.glob("stream-*.json"))
    assert len(traces) == 1
    backend = json.loads(traces[0].read_text())
    assert backend["chat_body"]["model"] == "engine-model"
    assert backend["chat_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert backend["tokenized"]["input_ids"] == list(range(10))
    assert analyze_trial(record, backend, 10, 4)["exact_first_token_observed"] is True
    adapter.generate.assert_not_called()
    assert manager.closed == ["engine-request"]
