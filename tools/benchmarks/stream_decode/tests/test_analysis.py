import copy

import pytest

from tools.benchmarks.stream_decode.analysis import analyze, analyze_trial
from tools.benchmarks.stream_decode.observer import SOURCE_FILES, SOURCE_SHA256, VERSION, digest, save


@pytest.fixture
def records():
    def chunk(at, text=None, finish=None, usage=None):
        return {
            "at_ns": at,
            "chunk": {
                "id": "engine-id",
                "choices": [] if usage else [{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
                "usage": usage,
            },
        }

    client = {
        "completed": True,
        "retry_count": 0,
        "http_requests": 1,
        "clock_domain": "same-boot",
        "index": 0,
        "warmup": False,
        "started_ns": 1_000_000_000,
        "finished_ns": 6_000_000_000,
        "chunks": [
            chunk(2_500_000_000, "a"),
            chunk(4_500_000_000, "bcd"),
            chunk(5_000_000_000, finish="length"),
            chunk(5_500_000_000, usage={"prompt_tokens": 10, "completion_tokens": 4}),
        ],
    }
    backend = {
        "id": "only",
        "completed": True,
        "clock_domain": "same-boot",
        "started_ns": 1_100_000_000,
        "finished_ns": 4_200_000_000,
        "events": [],
        "incremental_streaming_output": False,
    }
    for at, tokens, text, finish in [(2, 1, "a", None), (3, 3, "abc", None), (4, 4, "abcd", "length")]:
        backend["events"].append(
            {
                "at_ns": at * 1_000_000_000,
                "event": {
                    "text": text,
                    "meta_info": {
                        "id": "engine-id",
                        "prompt_tokens": 10,
                        "completion_tokens": tokens,
                        "finish_reason": finish,
                    },
                },
            }
        )
    return client, backend


def test_token_counts_and_distinct_time_windows(records):
    result = analyze_trial(*records, 10, 4)
    assert result["client_ttft_s"] == 1.5
    assert result["client_total_s"] == 5
    assert result["completion_tokens_per_total_s"] == 0.8
    assert result["backend_after_first_token_tokens_per_s"] == 1.5
    assert result["largest_token_advance_group"] == 2
    assert result["backend_token_advance_groups"] == 3


def test_first_group_coalesced_has_no_exact_first_token_claim(records):
    client, backend = records
    backend["events"][0]["event"]["meta_info"]["completion_tokens"] = 2
    result = analyze_trial(client, backend, 10, 4)
    assert result["backend_after_first_observation_tokens_per_s"] == 1
    assert result["backend_after_first_token_tokens_per_s"] is None
    assert not result["exact_first_token_observed"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_usage",
        "wrong_length",
        "early_stop",
        "no_finish",
        "retry",
        "requests",
        "clock",
        "missing_tokens",
        "bool_tokens",
        "regress_tokens",
        "text",
        "incomplete",
        "clock_regression",
        "mixed_id",
        "backend_prompt",
        "tool_call",
        "backend_end",
        "backend_end_type",
        "sdk_id",
        "missing_sdk_id",
    ],
)
def test_incomplete_or_inconsistent_stream_is_rejected(records, mutation):
    client, backend = records
    if mutation == "missing_usage":
        client["chunks"][-1]["chunk"]["usage"] = None
    elif mutation == "wrong_length":
        client["chunks"][-1]["chunk"]["usage"]["completion_tokens"] = 3
    elif mutation == "early_stop":
        client["chunks"][-2]["chunk"]["choices"][0]["finish_reason"] = "stop"
    elif mutation == "no_finish":
        backend["events"][-1]["event"]["meta_info"]["finish_reason"] = None
    elif mutation == "retry":
        client["retry_count"] = 1
    elif mutation == "requests":
        client["http_requests"] = 2
    elif mutation == "clock":
        backend["clock_domain"] = "other-boot"
    elif mutation == "missing_tokens":
        del backend["events"][1]["event"]["meta_info"]["completion_tokens"]
    elif mutation == "bool_tokens":
        backend["events"][0]["event"]["meta_info"]["completion_tokens"] = True
    elif mutation == "regress_tokens":
        backend["events"][1]["event"]["meta_info"]["completion_tokens"] = 0
    elif mutation == "text":
        backend["events"][-1]["event"]["text"] = "abcE"
    elif mutation == "incomplete":
        client["completed"] = False
    elif mutation == "clock_regression":
        backend["events"][1]["at_ns"] = 1
    elif mutation == "mixed_id":
        backend["events"][1]["event"]["meta_info"]["id"] = "another"
    elif mutation == "backend_prompt":
        backend["events"][1]["event"]["meta_info"]["prompt_tokens"] = 11
    elif mutation == "tool_call":
        client["chunks"][0]["chunk"]["choices"][0]["delta"]["tool_calls"] = [{"index": 0}]
    elif mutation == "backend_end":
        backend["finished_ns"] = backend["events"][0]["at_ns"]
    elif mutation == "backend_end_type":
        backend["finished_ns"] = True
    elif mutation == "sdk_id":
        client["chunks"][0]["chunk"]["id"] = "wrong-id"
    elif mutation == "missing_sdk_id":
        del client["chunks"][0]["chunk"]["id"]
    with pytest.raises(ValueError, match=r".+"):
        analyze_trial(client, backend, 10, 4)


def test_terminal_usage_delay_does_not_inflate_decode_interval(records):
    client, backend = records
    duplicate = copy.deepcopy(backend["events"][-1])
    backend["events"][-1]["event"]["meta_info"]["finish_reason"] = None
    duplicate["at_ns"] = 5_000_000_000
    backend["events"].append(duplicate)
    backend["finished_ns"] = 5_100_000_000
    assert analyze_trial(client, backend, 10, 4)["backend_decode_interval_s"] == 2


def cohort(tmp_path, records):
    clients, servers = tmp_path / "client", tmp_path / "server"
    clients.mkdir()
    servers.mkdir()
    manifest = {"cohort": "fixed", "clock_domain": "same-boot", "source": "frozen"}
    sampling = {
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "repetition_penalty": 1,
        "seed": 17,
        "min_tokens": 4,
        "max_completion_tokens": 4,
    }
    plan = {
        "completion_tokens": 4,
        "warmups": 0,
        "repeats": 1,
        "cohort": "fixed",
        "model": "model",
        "prompt": "public",
        "prompt_sha256": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7",
        "server_manifest_sha256": digest(manifest),
        "sampling": sampling,
        "clock_domain": "same-boot",
    }
    body = {
        "model": "model",
        "messages": [{"role": "user", "content": "public"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        **sampling,
    }
    records[0].update(cohort="fixed", plan_sha256=digest(plan), request_body=copy.deepcopy(body))
    records[1].update(
        cohort="fixed",
        server_manifest_sha256=digest(manifest),
        source_sha256=SOURCE_SHA256,
        version=VERSION,
        model_path="public/checkpoint",
        served_model_name="model",
        pid=1,
        chat_body=copy.deepcopy(body),
        adapted_sampling={"temperature": 0},
        tokenized={
            "input_ids": list(range(10)),
            "rid": "engine-id",
            "has_multimodal_inputs": False,
            "sampling": {
                "temperature": 1,
                "top_p": 1,
                "top_k": 1,
                "min_p": 0,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "repetition_penalty": 1,
                "sampling_seed": 17,
                "n": 1,
                "min_new_tokens": 4,
                "max_new_tokens": 4,
            },
        },
    )
    save(clients / "plan.json", plan)
    save(
        clients / "controller.json",
        {"completed": True, "children": [{"index": 0, "returncode": 0, "timed_out": False, "signals": []}]},
    )
    save(servers / "server.json", manifest)
    save(
        servers / "installed-1.json",
        {
            "pid": 1,
            "server_manifest_sha256": digest(manifest),
            "source_sha256": SOURCE_SHA256,
            "source_files": SOURCE_FILES,
            "version": VERSION,
        },
    )
    save(clients / "trial-0000.json", records[0])
    save(servers / "stream-one.json", records[1])
    return clients, servers


def test_fixed_cohort_and_unique_request_join(tmp_path, records):
    clients, servers = cohort(tmp_path, records)
    assert analyze(clients, servers, 10, 4)["measured_trials"] == 1
    other = copy.deepcopy(records[1])
    other["id"] = "extra"
    save(servers / "stream-extra.json", other)
    with pytest.raises(ValueError, match="exactly one"):
        analyze(clients, servers, 10, 4)


@pytest.mark.parametrize(
    "mutation", ["prompt", "cohort", "source", "input_ids", "sampling", "request_id", "thinking", "model", "controller"]
)
def test_request_and_source_binding_refuses_mismatched_evidence(tmp_path, records, mutation):
    clients, servers = cohort(tmp_path, records)
    backend = records[1]
    if mutation == "prompt":
        backend["chat_body"]["messages"][0]["content"] = "different"
    elif mutation == "cohort":
        backend["cohort"] = "another"
    elif mutation == "source":
        backend["source_sha256"] = "unreviewed"
    elif mutation == "input_ids":
        backend["tokenized"]["input_ids"].pop()
    elif mutation == "sampling":
        backend["tokenized"]["sampling"]["top_k"] = 2
    elif mutation == "request_id":
        backend["tokenized"]["rid"] = "different"
    elif mutation == "thinking":
        backend["chat_body"]["chat_template_kwargs"]["enable_thinking"] = True
    elif mutation == "model":
        backend["chat_body"]["model"] = "another"
    elif mutation == "controller":
        save(clients / "controller.json", {"completed": False})
    save(servers / "stream-one.json", backend)
    with pytest.raises(ValueError, match=r".+"):
        analyze(clients, servers, 10, 4)


def test_backend_cleanup_can_finish_after_client_without_changing_decode(records):
    records[1]["finished_ns"] = records[0]["finished_ns"] + 1_000_000_000
    assert analyze_trial(*records, 10, 4)["backend_after_first_token_tokens_per_s"] == 1.5


def test_incremental_text_mode_uses_cumulative_counts(records):
    records[1]["incremental_streaming_output"] = True
    for row, text in zip(records[1]["events"], ["a", "bc", "d"], strict=True):
        row["event"]["text"] = text
    assert analyze_trial(*records, 10, 4)["backend_after_first_token_tokens_per_s"] == 1.5


@pytest.mark.parametrize("end", [None, 1_500_000_000, 6_500_000_000])
def test_background_started_before_schedule_still_invalidates_overlap(tmp_path, records, end):
    clients, servers = cohort(tmp_path, records)
    save(servers / "background-health.json", {"started_ns": 1, "finished_ns": end, "clock_domain": "same-boot"})
    with pytest.raises(ValueError, match="background generation overlaps"):
        analyze(clients, servers, 10, 4)


def test_finished_startup_generation_before_schedule_is_retained(tmp_path, records):
    clients, servers = cohort(tmp_path, records)
    save(servers / "background-startup.json", {"started_ns": 1, "finished_ns": 2, "clock_domain": "same-boot"})
    assert analyze(clients, servers, 10, 4)["background_calls_outside_schedule"] == 1


def test_unfinished_unrelated_chat_is_rejected_before_its_first_event(tmp_path, records):
    clients, servers = cohort(tmp_path, records)
    save(
        servers / "stream-open.json",
        {
            "id": "unrelated",
            "started_ns": 1_500_000_000,
            "finished_ns": None,
            "clock_domain": "same-boot",
            "completed": False,
            "events": [],
        },
    )
    with pytest.raises(ValueError, match="unexpected backend stream"):
        analyze(clients, servers, 10, 4)
