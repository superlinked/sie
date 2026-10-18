"""Worker authorization and immediate LoRA-loading responses for generation."""

import time
from unittest.mock import AsyncMock, Mock

import msgpack
import pytest
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.config.model import ModelConfig
from sie_server.processors.streaming import StreamingProcessor, _ValidationError


def config():
    return ModelConfig.model_validate(
        {
            "sie_id": "test/model",
            "hf_id": "test/model",
            "inputs": {"text": True},
            "tasks": {"generate": {"context_length": 1024, "max_output_tokens": 64}},
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.sglang.generation:SGLangGenerationAdapter",
                    "max_batch_tokens": 1024,
                    "kv_budget_tokens": 1024,
                    "adapter_options": {"loadtime": {"lora_paths": {"allowed": "org/allowed"}}},
                },
                "private": {
                    "extends": "default",
                    "adapter_options": {"loadtime": {"lora_paths": {"private": "org/private"}}},
                },
            },
        }
    )


def work_item(lora="allowed", **generate):
    return {
        "work_item_id": "req-1.0",
        "request_id": "req-1",
        "item_index": 0,
        "total_items": 1,
        "operation": "generate",
        "model_id": "test/model",
        "profile_id": "default",
        "pool_name": "default",
        "machine_profile": "default",
        "router_id": "router-1",
        "reply_subject": "_INBOX.router-1.req-1",
        "timestamp": time.time(),
        "generate": {"prompt": "Hello", "max_new_tokens": 8, "lora_adapter": lora, **generate},
    }


@pytest.mark.parametrize("lora", ["private", "org/private", "attacker/arbitrary", "", {}, 3])
def test_worker_refuses_undeclared_other_profile_or_malformed_lora(lora):
    result = StreamingProcessor._validate_generate_params(work_item(lora), config())
    assert isinstance(result, _ValidationError)
    assert result.code == "invalid_request"


@pytest.mark.parametrize("options", [None, {"profile": None}, {"profile": "default"}])
def test_declared_selected_profile_alias_is_accepted(options):
    wi = work_item()
    wi["options"] = options
    result = StreamingProcessor._validate_generate_params(wi, config())
    assert not isinstance(result, _ValidationError)
    assert result.lora_adapter == "allowed"


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True])
async def test_cold_generate_and_chat_return_loading_without_inference(chat):
    adapter = SGLangGenerationAdapter("base", mem_fraction_static=0.5, lora_paths={"allowed": "org/allowed"})
    registry = Mock()
    registry.is_loaded.return_value = True
    registry.get.return_value = adapter
    registry.get_config.return_value = config()
    registry.ensure_lora_loaded_async = AsyncMock(return_value=(False, True))
    registry.lora_retry_after_s.return_value = 3
    registry.device = "cpu"
    nc = AsyncMock()
    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="worker-1")
    wi = work_item()
    if chat:
        wi["generate"].pop("prompt")
        wi["generate"]["messages"] = [{"role": "user", "content": "Hello"}]
    msg = AsyncMock()
    msg.data = msgpack.packb(wi, use_bin_type=True)
    await proc.process(msg, "test/model")
    registry.ensure_lora_loaded_async.assert_awaited_once_with("test/model", "allowed")
    chunks = [msgpack.unpackb(call.args[1], raw=False) for call in nc.publish.await_args_list]
    assert chunks[-1]["error"]["code"] == "LORA_LOADING"
    assert chunks[-1]["done"]
    msg.ack.assert_awaited_once()
