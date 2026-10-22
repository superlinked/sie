"""Pinned self-hosted SIE features; hosting charges remain explicitly unknown."""

import hashlib
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import httpx
import numpy as np

from embedding import collect_features, recipes as core, token_audit, wire
from reader_io import ChangedFileSync


def checked_url(value):
    url = urlsplit(value)
    core.require(url.scheme in {"http", "https"} and url.hostname
                 and url.username is None and url.password is None
                 and not url.query and not url.fragment and url.path in {"", "/"},
                 "Use an explicit HTTP(S) server origin without credentials or query parameters")
    core.require(url.hostname != "api.superlinked.com", "Use the Cloud path for the SIE Cloud origin")
    return value.rstrip("/")


def make_client(base_url, capture, *, transport=None):
    base_url = checked_url(base_url)
    session = httpx.Client(base_url=base_url, trust_env=False, follow_redirects=False,
                           timeout=httpx.Timeout(300, connect=10), transport=transport,
                           event_hooks={"response": [capture]})
    # Empty explicitly suppresses the SDK's SIE_API_KEY environment fallback.
    # A separately named credential is scoped only to this explicit server.
    return wire.SingleSendClient(base_url, api_key=os.environ.get("SIE_SELF_HOSTED_API_KEY", ""),
                                 remote="forbid", http_client=session, timeout_s=300)


def prepare(source, base_url, seconds, *, batch_size=8):
    core.require(type(seconds) is int and seconds > 0, "Invalid collection deadline")
    core.require(type(batch_size) is int and 1 <= batch_size <= 16, "Invalid batch size")
    source = Path(source)
    rows, hashes = collect_features.source_rows(source, core.sha(source / "source-plan.json"))
    return {"format": "bbeh-public-selfhost-feature-plan-v1", "reproduction": True,
            "source_plan_sha256": core.sha(source / "source-plan.json"), "role_inputs_sha256": hashes,
            "n": len(rows), "model": core.MODEL, "hf_revision": core.REVISION,
            "instruction": core.INSTRUCTION, "dimensions": 2560, "base_url": checked_url(base_url),
            "collection_seconds": seconds, "batch_size": batch_size, "automatic_redispatch": False,
            "encoder_cost_usd": None, "cost_scope": "Self-hosted execution; hosting and encoder fees are unpriced",
            "source_sha256": {"selfhost": core.sha(__file__), "source_reader": core.sha(collect_features.__file__),
                              "wire": core.sha(wire.__file__), "token_audit": core.sha(token_audit.__file__)},
            "cloud_credentials_used": False, "source_only": True}


def validate(reply, rows, capture, client, body):
    receipt = capture.receipt
    core.require(receipt is not None and receipt["http_status"] == 200, "Missing terminal encoder response")
    expected_wire = hashlib.sha256(wire.packb(body, use_bin_type=True)).hexdigest()
    core.require(receipt["request_wire_sha256"] == expected_wire, "SDK request settings changed")
    core.require(isinstance(reply, list) and len(reply) == len(rows), "Wrong encoder result count")
    by_id = {value.get("id"): value for value in reply}
    core.require(len(by_id) == len(rows) and set(by_id) == {row["id"] for row in rows}, "Encoder IDs changed")
    core.require(receipt["response_model"] == core.MODEL
                 and all(value.get("model") == core.MODEL for value in reply), "Response model changed")
    execution = client.last_model_revision
    core.require(isinstance(execution, str) and re.fullmatch(r"[0-9a-f]{64}", execution)
                 and receipt["headers"].get("x-sie-model-revision") == execution,
                 "Missing or inconsistent execution revision")
    core.require(client.last_retry_count == 0, "Unexpected SDK retry")
    vectors = np.asarray([by_id[row["id"]]["dense"] for row in rows], dtype=np.float64)
    core.require(vectors.shape == (len(rows), 2560) and np.isfinite(vectors).all(), "Invalid encoder vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    core.require(np.isfinite(norms).all() and (norms > 0).all(), "Invalid vector norms")
    return (vectors / norms).astype(np.float32), execution


def run(source, output, plan, tokenizer, tokenizer_binding, *, client_factory=make_client):
    source, output = Path(source), Path(output)
    core.require(not output.exists(), "Preserve previous intents; self-host requests are not automatically repeated")
    rows, input_hashes = collect_features.source_rows(source, plan["source_plan_sha256"])
    core.require(len(rows) == plan["n"] and input_hashes == plan["role_inputs_sha256"], "Source changed")
    core.require(plan == prepare(source, plan["base_url"], plan["collection_seconds"], batch_size=plan["batch_size"]),
                 "Self-host collection plan changed")
    core.require(tokenizer_binding["model"] == core.MODEL and tokenizer_binding["hf_revision"] == core.REVISION,
                 "Wrong tokenizer identity")
    output.mkdir(parents=True)
    checkpoint = ChangedFileSync()
    core.write(output / "plan.json", plan)
    for role in core.ROLES:
        selected = [row for row in rows if row["role"] == role]
        lengths = token_audit.audit_rows(selected, tokenizer)
        audit = {"format": "bbeh-encoder-token-audit-v1", "role": role, "model": core.MODEL,
                 "hf_revision": core.REVISION, "instruction": core.INSTRUCTION,
                 "source_plan_sha256": plan["source_plan_sha256"], "inputs_sha256": input_hashes[role],
                 "template": token_audit.TEMPLATE, "max_length": token_audit.MAX_LENGTH,
                 "tokenizer": tokenizer_binding, "rows": lengths, "truncation": False, "truncated_ids": [],
                 "overflow_ids": [r["id"] for r in lengths if r["input_tokens"] > token_audit.MAX_LENGTH]}
        core.write(output / "features" / role / "token-audit.json", audit)
        core.require(not audit["overflow_ids"], "Encoder context overflow; no requests dispatched or rows truncated")
    checkpoint(output)
    capture, batches, vectors, current, execution = wire.WireCapture(), [], [], None, None
    started = time.monotonic()
    terminal = {"status": "incomplete", "n": len(rows), "complete_rows": 0,
                "encoder_cost_usd": None, "cost_scope": plan["cost_scope"], "self_hosted": True}
    try:
        with client_factory(plan["base_url"], capture) as client:
            catalog = client.get_model(core.MODEL)
            core.require(catalog.get("name") == core.MODEL and catalog.get("revision") == core.REVISION
                         and catalog.get("dims", {}).get("dense") == 2560 and catalog.get("loaded") is True,
                         "Self-hosted encoder is not the ready pinned model")
            core.write(output / "catalog.json", catalog)
            for number, offset in enumerate(range(0, len(rows), plan["batch_size"])):
                core.require(time.monotonic() - started < plan["collection_seconds"], "Collection deadline reached")
                batch = rows[offset:offset + plan["batch_size"]]
                current = output / "batches" / f"{number:06d}"
                body = wire.body_for(batch, core.INSTRUCTION)
                core.write(current / "intent.json", {"request": body, "plan_sha256": core.sha(output / "plan.json"),
                           "ids": [row["id"] for row in batch], "physical_attempt": 1})
                checkpoint(output)
                capture.arm(current)
                before = time.monotonic()
                reply = client.encode(core.MODEL, body["items"], output_types=["dense"],
                                      instruction=core.INSTRUCTION, output_dtype="float32", is_query=True,
                                      wait_for_capacity=False, max_oom_retries=0, provision_timeout_s=150)
                matrix, revision = validate(reply, batch, capture, client, body)
                core.require(execution is None or execution == revision, "Execution changed during collection")
                execution = revision
                with (current / "vectors.npy").open("xb") as stream:
                    np.save(stream, matrix, allow_pickle=False)
                receipt = {"ids": [row["id"] for row in batch], "status": "complete",
                           "intent_sha256": core.sha(current / "intent.json"),
                           "http_sha256": core.sha(current / "http.json"),
                           "vectors_sha256": core.sha(current / "vectors.npy"),
                           "execution_bundle_sha256": execution, "catalog_hf_revision": core.REVISION,
                           "elapsed_seconds": time.monotonic() - before,
                           "usage": capture.receipt["usage"], "encoder_cost_usd": None}
                core.write(current / "receipt.json", receipt)
                batches.append({"path": (current / "receipt.json").relative_to(output).as_posix(),
                                "sha256": core.sha(current / "receipt.json")})
                vectors.append(matrix)
                terminal["complete_rows"] += len(batch)
                current = None
                checkpoint(output)
        collection = {"format": "bbeh-selfhost-feature-collection-v1", "status": "complete", "n": len(rows),
                      "plan_sha256": core.sha(output / "plan.json"), "batches": batches,
                      "encoder_cost_usd": None, "cost_scope": plan["cost_scope"]}
        core.write(output / "collection.json", collection)
        matrix = np.concatenate(vectors)
        for role in core.ROLES:
            indices = [i for i, row in enumerate(rows) if row["role"] == role]
            directory = output / "features" / role
            core.write(directory / "ids.json", [rows[i]["id"] for i in indices])
            with (directory / "vectors.npy").open("xb") as stream:
                np.save(stream, matrix[indices], allow_pickle=False)
            core.write(directory / "features.json", {
                "format": "routerbench-fixed-embeddings-v1", "model": core.MODEL,
                "hf_revision": core.REVISION, "instruction": core.INSTRUCTION, "dimensions": 2560,
                "n": len(indices), "source_plan_sha256": plan["source_plan_sha256"],
                "inputs_sha256": input_hashes[role], "ids_sha256": core.sha(directory / "ids.json"),
                "vectors_sha256": core.sha(directory / "vectors.npy"),
                "token_audit_sha256": core.sha(directory / "token-audit.json"),
                "collection_sha256": core.sha(output / "collection.json"), "dtype": "float32",
                "normalization": "float64 L2 then float32", "outcomes_accessed": False, "self_hosted": True})
        core.write(output / "unpriced-encoder.json", {
            "format": "bbeh-unpriced-selfhost-encoder-v1", "cost_basis": "unpriced",
            "source_plan_sha256": plan["source_plan_sha256"], "model": core.MODEL,
            "hf_revision": core.REVISION, "instruction": core.INSTRUCTION,
            "rows": [{"id": row["id"], "projected_singleton_usd": None} for row in rows],
            "hosting_cost_usd": None, "cloud_fee_projection_applicable": False})
        terminal.update(status="complete", collection_sha256=core.sha(output / "collection.json"))
        return terminal
    except Exception as error:
        terminal.update(error_type=type(error).__name__,
                        unresolved_batch=None if current is None else current.relative_to(output).as_posix())
        raise
    finally:
        terminal["elapsed_seconds"] = time.monotonic() - started
        core.write(output / "terminal.json", terminal)
        checkpoint(output)
