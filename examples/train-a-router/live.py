"""Route a new question, then call exactly one of the evaluated answer models."""

from __future__ import annotations

from contextlib import nullcontext
from decimal import Decimal
import hashlib
import os
from pathlib import Path
import time

import httpx
import numpy as np
from sie_sdk import SIEClient

import answer_models
import frontier_analysis
import paired_native
import paired_training as core
import reproduction
import utility_router


class SingleSendClient(SIEClient):
    def _record_retry(self):
        raise RuntimeError("The request was not automatically repeated; inspect its saved receipt")


def sie_client(base_url, *, self_hosted=False):
    """Keep Cloud environment credentials scoped by the SDK; local servers get none."""
    return SingleSendClient(base_url=base_url, api_key="" if self_hosted else None, timeout_s=600)


def checked_model(run, method, target=None, *, candidate_id=None):
    run = Path(run)
    plan, rows = core.checked_plan(run)
    bundle, _ = reproduction.checked_routes(run / "frozen/frozen-final-routes.json", rows)
    core.require(bundle["plan_sha256"] == core.sha(run / "plan.json"), "Routing policy belongs to another run")
    core.require(method in ("embedding", "base", "lora"), "Unknown routing method")
    core.require((target is None) != (candidate_id is None), "Choose a calibration candidate or a recorded setting explicitly")
    core.require(target is None or (type(target) is int and target in (0, 1, 2, 5)), "Unknown recorded setting")
    family = "embedding" if method == "embedding" else "native"
    scores_path = run / f"{family}-scores.json"
    core.require(core.sha(scores_path) == bundle["scores_sha256"][family], "Saved input scores changed")
    scores = core.read(scores_path)
    model_path = run / ("embedding.json" if family == "embedding" else "native/trained.json")
    key = "model_sha256" if family == "embedding" else "trained_sha256"
    core.require(core.sha(model_path) == scores[key], "Saved router model changed")
    model = core.read(model_path)
    core.require(
        model["plan_sha256"] == bundle["plan_sha256"]
        and model["training_grades_sha256"] == bundle["training_grades_sha256"],
        "Training lineage changed",
    )
    if candidate_id is None:
        rule = bundle["policies"][f"{method}@{target}pp"]["rule"]
    else:
        checked_bundle, calibration = frontier_analysis.load_calibration(run)
        core.require(checked_bundle["calibration_sha256"] == bundle["calibration_sha256"], "Calibration binding changed")
        candidates = calibration["methods"][method]["candidates"]
        core.require(type(candidate_id) is int and 0 <= candidate_id < len(candidates), "Unknown calibration candidate")
        candidate = candidates[candidate_id]
        core.require(candidate["candidate_id"] == candidate_id, "Calibration candidate identity changed")
        rule = candidate["policy"]
    return plan, rule, model


def embed(client, request, identity, *, self_hosted=False):
    catalog = client.get_model(core.MODEL)
    core.require(
        catalog["revision"] == core.REVISION and catalog["dims"]["dense"] == 2560,
        "The serving encoder differs from the trained encoder",
    )
    reply = client.encode(
        core.MODEL,
        [{"id": identity, "text": request}],
        output_types=["dense"],
        instruction=core.INSTRUCTION,
        output_dtype="float32",
        is_query=True,
        wait_for_capacity=False,
        max_oom_retries=0,
        provision_timeout_s=150,
    )
    core.require(
        isinstance(reply, list) and len(reply) == 1 and reply[0]["id"] == identity and reply[0]["model"] == core.MODEL,
        "Embedding response does not match the question",
    )
    vector = np.asarray(reply[0]["dense"], dtype=np.float64)
    norm = np.linalg.norm(vector)
    core.require(
        vector.shape == (2560,) and np.isfinite(vector).all() and np.isfinite(norm) and norm > 0,
        "Embedding response is not a finite nonzero 2,560-dimensional vector",
    )
    item = reply[0]
    metadata = item.get("request") or {}
    usage = item.get("usage") or metadata.get("usage") or {}
    credits = metadata.get("credits_debited", usage.get("credits_charged_decimal", usage.get("credits_charged")))
    if not self_hosted:
        core.require(
            credits is not None and Decimal(str(credits)).is_finite() and Decimal(str(credits)) >= 0,
            "Cloud embedding response has no valid settled charge",
        )
    return (vector / norm).astype(np.float32), {
        "model": core.MODEL,
        "hf_revision": core.REVISION,
        "execution_bundle_sha256": client.last_model_revision,
        "request_id": metadata.get("id"),
        "observed_credits": None if self_hosted else credits,
        "observed_cloud_usd": None if self_hosted else str(Decimal(str(credits)) * Decimal("0.00001")),
    }


def route(run, question, method, target=None, *, candidate_id=None, client=None, device="cpu", self_hosted=False):
    plan, rule, model = checked_model(run, method, target, candidate_id=candidate_id)
    started = time.perf_counter()
    probabilities, expected, embedding = None, None, None
    if rule["mode"] in ("always_cheap", "always_strong"):
        strong = rule["mode"] == "always_strong"
    elif method == "embedding":
        core.require(client is not None, "The embedding route needs an SIE client")
        identity = "new-" + hashlib.sha256(question.encode()).hexdigest()[:16]
        vector, embedding = embed(client, question, identity, self_hosted=self_hosted)
        probabilities = np.array([[float(core.probability(vector, model["fit"]["models"][arm])) for arm in core.ARMS]])
        expected = core.expected_usd(probabilities, model["fit"]["conditional_cost_usd"])
        strong = bool(utility_router.replay(rule, probabilities, expected)[0])
    else:
        checkpoint = Path(run) / "native" / method
        actual = paired_native.checkpoint_files(checkpoint)
        core.require(
            actual == model["checkpoints"][method]["files_sha256"],
            "Decision-model files differ; fetch the optional weights before using this route",
        )
        probabilities = paired_native.checkpoint_probabilities(
            checkpoint, [{"request": question}], plan["pair"], device=device
        )
        expected = core.expected_usd(probabilities, model["conditional_cost_usd"])
        strong = bool(utility_router.replay(rule, probabilities, expected)[0])
    arm = "strong" if strong else "cheap"
    return {
        "method": method,
        "arm": arm,
        "model": plan["pair"]["arms"][arm]["model"],
        "calibration_target_gap_pp": target,
        "calibration_candidate_id": candidate_id,
        "rule": rule,
        "predicted_correctness": None if probabilities is None else probabilities[0].tolist(),
        "predicted_answer_usd": None if expected is None else expected[0].tolist(),
        "arm_order": list(core.ARMS),
        "embedding": embedding,
        "elapsed_seconds": time.perf_counter() - started,
        "timing_scope": "Decision time includes native checkpoint verification/loading or the embedding API call; initial saved-policy checks are excluded. This is not warm serving latency.",
    }, plan["pair"]


def solve(item, arm, pair, *, sie_client=None, self_hosted=False):
    config = pair["arms"][arm]
    key = config["arm"]
    core.require(
        key in answer_models.ARMS and config["model"] == answer_models.ARMS[key]["model"],
        "Answer-model identity differs from the recorded model pair",
    )
    core.require(
        core.sha(answer_models.__file__) == pair["request_builder_source_sha256"],
        "Answer request builder differs from the recorded experiment",
    )
    body = answer_models.body_for(item, key)
    core.require(
        all(body.get(name) == value for name, value in config["settings"].items()),
        "Live request settings differ from the evaluated settings",
    )
    started = time.perf_counter()
    if answer_models.ARMS[key]["provider"] == "sie":
        core.require(sie_client is not None, "This answer model needs an SIE client")
        metadata = sie_client.get_model(config["model"])
        core.require(metadata["revision"] == answer_models.ARMS[key]["revision"], "Answer checkpoint revision changed")
        reply = sie_client.chat_completions(
            **body, wait_for_capacity=True, provision_timeout_s=300, read_timeout_s=600, max_oom_retries=0
        )
        choices = reply.get("choices") or []
        status = choices[0].get("finish_reason") if len(choices) == 1 else None
        content = choices[0].get("message", {}).get("content") if len(choices) == 1 else None
        usage = reply.get("usage") or {}
        credits = usage.get("credits_charged_decimal", usage.get("credits_charged"))
        if not self_hosted:
            core.require(
                credits is not None and Decimal(str(credits)).is_finite() and Decimal(str(credits)) >= 0,
                "Cloud response has no valid settled charge",
            )
        record = {
            "model": config["model"],
            "provider_status": status,
            "usage": usage,
            "answer": answer_models.parse_answer(content, item["response_contract"]) if status == "stop" else None,
            "observed_credits": None if self_hosted else credits,
            "observed_cloud_usd": None if self_hosted else str(Decimal(str(credits)) * Decimal("0.00001")),
            "content": content,
        }
    else:
        core.require(pair["prices_usd_per_million"] == answer_models.PRICES, "Saved answer price basis changed")
        key_value = os.environ.get("OPENAI_API_KEY")
        core.require(bool(key_value), "Set OPENAI_API_KEY for the Sol/Luna answer models")
        with httpx.Client(
            timeout=httpx.Timeout(600, connect=30),
            follow_redirects=False,
            transport=httpx.HTTPTransport(retries=0),
            trust_env=False,
        ) as http:
            response = http.post(
                "https://api.openai.com/v1/responses",
                json=body,
                headers={"Authorization": "Bearer " + key_value, "Content-Type": "application/json"},
            )
            response.raise_for_status()
            reply = response.json()
        core.require(reply.get("service_tier") == "default", "Unexpected OpenAI service tier")
        resolved = reply.get("model", "")
        core.require(resolved == config["model"], "Response model differs from the recorded experiment")
        text = "".join(
            part.get("text", "")
            for message in reply.get("output", [])
            if message.get("type") == "message"
            for part in message.get("content", [])
            if part.get("type") == "output_text"
        )
        record = {
            "model": resolved,
            "provider_status": reply.get("status"),
            "response_id": reply.get("id"),
            "usage": reply.get("usage"),
            "content": text,
            "answer": answer_models.parse_answer(text, item["response_contract"])
            if reply.get("status") == "completed"
            else None,
            "cost": answer_models.openai_cost(reply.get("usage") or {}, body),
        }
    record.update(
        elapsed_seconds=time.perf_counter() - started,
        physical_answer_calls=1,
        valid_answer=record["answer"] is not None,
        quality="Not graded; no answer key was supplied",
    )
    return record


def answer(
    run,
    question,
    output,
    *,
    method="embedding",
    target=None,
    candidate_id=None,
    contract=None,
    sie_base_url=None,
    self_hosted=False,
    device="cpu",
):
    output = Path(output)
    core.require(not output.exists(), "Use a new output folder for each live question")
    core.require(
        isinstance(question, str) and question.strip() and len(question) <= 20000,
        "Use a nonempty question of at most 20,000 characters",
    )
    contract = contract or {"type": "exact_string", "max_length": 2048}
    answer_models.schema_for(contract, False)
    plan, rule, _ = checked_model(run, method, target, candidate_id=candidate_id)
    needs_sie = (method == "embedding" and not rule["mode"].startswith("always_")) or any(
        answer_models.ARMS[value["arm"]]["provider"] == "sie" for value in plan["pair"]["arms"].values()
    )
    url = sie_base_url or os.environ.get("SIE_BASE_URL") or "https://api.superlinked.com"
    output.mkdir(parents=True)
    item = {
        "id": "new-" + hashlib.sha256(question.encode()).hexdigest()[:16],
        "request": question,
        "response_contract": contract,
    }
    core.write(
        output / "intent.json",
        {
            "input": item,
            "method": method,
            "target_gap_pp": target,
            "calibration_candidate_id": candidate_id,
            "self_hosted": self_hosted,
            "automatic_retries": False,
        },
    )
    try:
        context = (
            sie_client(url, self_hosted=self_hosted)
            if needs_sie
            else nullcontext(None)
        )
        with context as client:
            routing, pair = route(run, question, method, target, candidate_id=candidate_id,
                                  client=client, device=device, self_hosted=self_hosted)
            core.write(output / "routing.json", routing)
            record = solve(item, routing["arm"], pair, sie_client=client, self_hosted=self_hosted)
            core.write(output / "answer.json", record)
    except Exception as error:
        core.write(
            output / "incomplete.json",
            {
                "status": "failed_or_unknown",
                "error_type": type(error).__name__,
                "http_status": getattr(getattr(error, "response", None), "status_code", None),
                "automatic_retries": False,
            },
        )
        raise
    result = {
        "model": routing["model"],
        "answer": record["answer"],
        "valid_answer": record["valid_answer"],
        "answer_quality": record["quality"],
        "physical_answer_calls": 1,
        "embedding_calls": int(routing["embedding"] is not None),
        "output": str(output),
    }
    core.write(output / "result.json", result)
    return result
