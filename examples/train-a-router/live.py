"""Apply one calibration-selected router and call exactly one answer model."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import numpy as np

import collect
from embedding import recipes as core
from embedding.cost_iteration import study as costs
from evaluation import utility_router as utility
from reader_io import sync_files
import workflow


def selected_policy(run, method, candidate_id):
    run = Path(run).resolve()
    if (run / "reproduction.json").exists():
        provenance = core.read(run / "reproduction.json")
        workflow.checked_run(provenance["artifact_directory"], run)
        calibration_path = run / "internal/calibration-frontiers.json"
        frozen_path = run / "internal/frozen-frontiers.json"
        envelope = core.read(run / "calibration.json")
        core.require(envelope["new_blind_evaluation"] is False
            and envelope["reproduction_sha256"] == core.sha(run / "reproduction.json")
            and envelope["calibration_sha256"] == core.sha(calibration_path)
            and envelope["frozen_sha256"] == core.sha(frozen_path), "Reproduction calibration changed")
    else:
        data = workflow.checked_data(run.parent)
        core.require(run == data / "reference", "Use a prepared run or the verified reference directory")
        calibration_path = run / "calibration-frontiers.json"
        frozen_path = run / "frozen-frontiers.json"
    calibration = core.read(calibration_path)
    frozen, plan = core.read(frozen_path), core.checked_plan(run / "embedding")
    core.require(calibration["format"] == "bbeh-calibration-frontiers-v1"
        and calibration["final_outcomes_accessed"] is False, "Expected calibration-only policy selection")
    core.require(frozen["calibration_sha256"] == core.sha(calibration_path)
        and calibration["source_plan_sha256"] == frozen["source_plan_sha256"] == plan["source_plan_sha256"]
        and calibration["pair_sha256"] == frozen["pair_sha256"] == plan["pair_sha256"]
        and calibration["plan_sha256"] == frozen["plan_sha256"] == core.sha(run / "embedding/plan.json"),
        "Policy source/pair/plan lineage changed")
    value = calibration["methods"][method]
    core.require(value["model_sha256"] == frozen["methods"][method]["model_sha256"]
        and value["candidates"] == frozen["methods"][method]["candidates"], "Frozen policy binding changed")
    core.require(not value["metadata_assisted"], "This control requires benchmark metadata unavailable in live requests")
    selected = [r for r in value["candidates"] if r["candidate_id"] == candidate_id]
    core.require(len(selected) == 1, "Unknown candidate ID; choose one from calibration")
    allowed = {r["candidate_id"] for r in value["frontier"]}
    core.require(candidate_id in allowed or selected[0]["policy"]["mode"].startswith("always_"),
                 "Choose a calibration-frontier candidate or always-arm baseline")
    return selected[0]["policy"], value["model_sha256"], core.sha(calibration_path)


def binding(run, method, expected):
    quality = method.removesuffix("__textcost")
    native = quality in {"base", "lora"}
    path = run / "cost" / ("native-combined-model.json" if native else "combined-model.json")
    saved = core.read(path)
    core.require(core.sha(path) == expected and quality in saved["methods"], "Quality/cost model binding changed")
    model_path = run / ("native/native/trained.json" if native else "embedding/model.json")
    core.require(core.sha(model_path) == saved["quality_model_sha256"]
        and core.sha(run / "cost/cost-model.json") == saved["cost_model_sha256"], "Fitted model bytes changed")
    return quality, core.read(model_path)


def embed(item, output, *, base_url=None, self_hosted=False):
    from embedding import wire  # Optional SIE/httpx dependencies.
    from transformers import AutoTokenizer
    from embedding import token_audit

    tokenizer = AutoTokenizer.from_pretrained(core.MODEL, revision=core.REVISION, trust_remote_code=False)
    count = token_audit.audit_rows([item], tokenizer)[0]["input_tokens"]
    core.require(count <= token_audit.MAX_LENGTH, "Question exceeds the encoder context; shorten it rather than truncate")

    endpoint = base_url or wire.BASE_URL
    parsed = urlsplit(endpoint)
    core.require(parsed.scheme in {"https", "http"} and parsed.hostname and not parsed.username
        and not parsed.password and not parsed.query and not parsed.fragment, "Use a plain SIE endpoint URL")
    if not self_hosted:
        core.require(endpoint.rstrip("/") == wire.BASE_URL, "Use --self-hosted for another endpoint")
        core.require(bool(os.environ.get("SIE_API_KEY")), "Set SIE_API_KEY for Cloud embeddings")
        capture = wire.WireCapture()
        with wire.make_client(capture) as client:
            receipt = wire.collect_batches([item], output, {
                "n": 1, "batch_size": 1, "instruction": core.INSTRUCTION, "max_seconds": 600,
                "budget_usd": "1.00", "model": core.MODEL, "hf_revision": core.REVISION, "dimensions": 2560,
            }, client, capture)
        core.require(receipt["status"] == "complete", "Embedding failed; physical attempt evidence is saved")
        _, matrix, _ = wire.load_vectors(output, [item])
        return matrix, {"encoder_usd": str(wire.decimal(receipt["known_credits_decimal"]) * wire.CREDIT_USD),
                        "deployment": "SIE Cloud", "hosting_usd": None}
    core.require(base_url is not None, "Supply your SIE server with --sie-base-url")
    from embedding import selfhost

    # Use the same single-attempt SDK/receipt adapter as dataset encoding.
    capture = wire.WireCapture()
    with selfhost.make_client(endpoint, capture) as client:
        metadata = client.get_model(core.MODEL)
        core.require(metadata.get("name") == core.MODEL and metadata.get("revision") == core.REVISION
                     and metadata.get("loaded") is True and metadata.get("dims", {}).get("dense") == 2560,
                     "Self-hosted encoder differs from the pinned model")
        body = wire.body_for([item], core.INSTRUCTION)
        core.write(output / "catalog.json", metadata)
        core.write(output / "intent.json", {"request": body, "physical_attempt": 1})
        sync_files(list(output.glob("*.json")), output)
        capture.arm(output)
        result = client.encode(core.MODEL, body["items"],
            output_types=["dense"], instruction=core.INSTRUCTION, output_dtype="float32", is_query=True,
            wait_for_capacity=False, max_oom_retries=0, provision_timeout_s=150)
        matrix, revision = selfhost.validate(result, [item], capture, client, body)
    core.write(output / "receipt.json", {"execution_bundle_sha256": revision,
        "http_sha256": core.sha(output / "http.json"), "encoder_usd": None, "hosting_usd": None})
    sync_files(list(output.glob("*.json")), output)
    return matrix, {"encoder_usd": None, "deployment": "self-hosted SIE", "hosting_usd": None}


def predict(run, item, output, method, quality, model, *, device, base_url, self_hosted):
    row = {**item, "task": "live"}
    info = {"encoder_usd": None, "hosting_usd": None}
    if quality in {"base", "lora"}:
        workflow.invoke("native/reader.py", "--run", run / "native", "--device", device,
            "--question-file", output / "question.json", "--variant", quality,
            "--output", output / "native-prediction.json")
        result = core.read(output / "native-prediction.json")
        probabilities, expected = np.asarray(result["probabilities"]), np.asarray(result["expected_usd"])
    elif quality in {"embedding", "knn"}:
        vector, info = embed(item, output / "encoder", base_url=base_url, self_hosted=self_hosted)
        if quality == "embedding":
            probabilities = core.predict_heads(vector, model["heads"])
        else:
            core.require(core.sha(run / "embedding/reference.npy") == model["reference_sha256"], "Neighbors changed")
            reference = np.load(run / "embedding/reference.npy", mmap_mode="r", allow_pickle=False)
            probabilities = core.nearest(vector, reference, np.asarray(model["labels"]), model["selected"]["knn"])
        expected = core.expected_cost(probabilities, model["conditional_cost_usd"])
    else:
        core.require(quality in {"text_length", "text_topic"}, "Unsupported live method")
        result = core.predict_controls(model, [row])[quality]
        probabilities = result["probabilities"]
        expected = result.get("expected_usd", core.expected_cost(probabilities, model["conditional_cost_usd"]))
    if method.endswith("__textcost"):
        expected = costs.predict_rows(core.read(run / "cost/cost-model.json"), [row], "text_length_log_ridge")
    return probabilities, expected, info


def answer(run, question, output, *, method, candidate_id, device="cpu", base_url=None, self_hosted=False):
    import openai_population  # Optional HTTP dependency; no provider call until the route is saved.

    run, output = Path(run).resolve(), Path(output).resolve()
    core.require(isinstance(question, str) and bool(question.strip()), "Provide the full question and answer format")
    core.require(not output.exists(), "Use a new output directory to avoid repeating unresolved calls")
    core.require(bool(os.environ.get("OPENAI_API_KEY")), "Set OPENAI_API_KEY for Sol or Luna answers")
    policy, model_hash, calibration_hash = selected_policy(run, method, candidate_id)
    output.mkdir(parents=True)
    item = {"id": "live:" + hashlib.sha256(question.encode()).hexdigest()[:20], "request": question,
            "response_contract": {"type": "exact_string", "max_length": 2048}}
    core.write(output / "question.json", item)
    started = time.perf_counter()
    if policy["mode"].startswith("always_"):
        strong = policy["mode"] == "always_strong"
        probabilities, expected, info = None, None, {"router_bypassed": True, "encoder_usd": "0"}
    else:
        quality, model = binding(run, method, model_hash)
        probabilities, expected, info = predict(run, item, output, method, quality, model,
            device=device, base_url=base_url, self_hosted=self_hosted)
        strong = bool(utility.replay(policy, probabilities, expected)[0])
    arm = "sol" if strong else "luna"
    route = {"method": method, "candidate_id": candidate_id, "policy": policy,
        "calibration_sha256": calibration_hash, "model": collect.ARMS[arm]["model"],
        "probabilities": None if probabilities is None else probabilities[0].tolist(),
        "expected_answer_usd": None if expected is None else expected[0].tolist(),
        "router_seconds_including_load": time.perf_counter() - started, **info}
    core.write(output / "route.json", route)
    core.write(output / "answer-intent.json", {"request": collect.body_for(item, arm), "physical_attempts": 1})
    sync_files(list(output.glob("*.json")), output)
    result = openai_population.acquire(item, arm)
    core.write(output / "answer.json", result)
    sync_files([output / "answer.json"], output)
    disposition = openai_population.decision(result, 1)
    core.require(disposition == "terminal" and result.get("status") == "complete"
        and result.get("valid_answer") is True,
        "Answer call did not produce a valid response under the requested model/settings; receipt saved, no retry")
    return {"model": route["model"], "answer": result.get("answer"), "status": result["status"],
            "answer_cost": result.get("cost"), "encoder_usd": info.get("encoder_usd"), "output": str(output)}
