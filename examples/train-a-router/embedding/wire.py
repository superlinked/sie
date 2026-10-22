"""Durable question-only SIE embedding batches; no solver or grading calls."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx
import numpy as np
from sie_sdk import SIEClient
from sie_sdk._msgpack import packb, unpackb

MODEL = "Qwen/Qwen3-Embedding-4B"
REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
DIMENSIONS = 2560
INSTRUCTION = "Represent the knowledge and reasoning required to answer this question correctly."
BASE_URL = "https://api.superlinked.com"
CREDIT_USD = Decimal("0.00001")
HEADER_NAMES = (
    "content-type",
    "x-sie-request-id",
    "x-sie-model-revision",
    "x-sie-credits-debited",
    "x-sie-credits-debited-decimal",
    "x-sie-execution-identity-sha256",
    "x-sie-execution-binding-sha256",
    "x-sie-runtime-instance",
    "x-sie-server-version",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode()


def write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(replace or not path.exists(), f"Preserve existing artifact: {path.name}")
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def decimal(value):
    require(not isinstance(value, bool) and isinstance(value, (str, int, float)), "Invalid charge type")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("Invalid charge") from error
    require(result.is_finite() and result >= 0, "Nonfinite or negative charge")
    return result


def body_for(rows, instruction):
    require(isinstance(instruction, str) and bool(instruction.strip()), "Explicit embedding instruction required")
    return {
        "items": [{"id": row["id"], "text": row["request"]} for row in rows],
        "params": {
            "output_types": ["dense"],
            "instruction": instruction,
            "output_dtype": "float32",
            "options": {"is_query": True},
        },
    }


class SingleSendClient(SIEClient):
    def _record_retry(self):
        raise RuntimeError("Automatic retries disabled; retain the first physical attempt")


class WireCapture:
    """A public httpx response hook: preserve received bytes before SDK parsing."""

    def __init__(self):
        self.directory = None
        self.receipt = None

    def arm(self, directory):
        self.directory, self.receipt = Path(directory), None

    def __call__(self, response):
        if response.request.url.path != f"/v1/encode/{MODEL}":
            return
        require(self.directory is not None and self.receipt is None, "Unexpected extra embedding dispatch")
        raw = response.read()
        write(self.directory / "response.bin", raw)
        receipt = {
            "http_status": response.status_code,
            "request_wire_sha256": hashlib.sha256(response.request.content).hexdigest(),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "headers": {key: response.headers[key] for key in HEADER_NAMES if key in response.headers},
            "usage": None,
            "response_model": None,
        }
        try:
            if "json" in response.headers.get("content-type", ""):
                envelope = response.json()
            else:
                envelope = unpackb(raw, numeric_arrays=True)
            if isinstance(envelope, dict):
                receipt["usage"] = envelope.get("usage")
                receipt["response_model"] = envelope.get("model")
        except Exception:
            receipt["body_parse_status"] = "unparsed; exact raw bytes preserved"
        write(self.directory / "http.json", receipt)
        self.receipt = receipt
        require(
            response.status_code != 202, "Asynchronous continuation unsupported; preserve intent without redispatch"
        )


def charge_from_wire(receipt):
    if receipt is None:
        return None
    usage, headers = receipt.get("usage") or {}, receipt["headers"]
    for value in (
        usage.get("credits_charged_decimal"),
        headers.get("x-sie-credits-debited-decimal"),
        usage.get("credits_charged"),
        headers.get("x-sie-credits-debited"),
    ):
        if value is not None:
            return decimal(value)
    return None


def validate_reply(reply, rows, wire, execution, quote, expected_wire_sha):
    require(isinstance(reply, list) and len(reply) == len(rows), "Wrong batch result count")
    ids = [item.get("id") for item in reply]
    require(len(set(ids)) == len(ids) and set(ids) == {row["id"] for row in rows}, "Wrong or duplicate response IDs")
    require(
        all(item.get("model") == MODEL for item in reply) and wire["response_model"] == MODEL,
        "Actual response model changed",
    )
    require(wire["request_wire_sha256"] == expected_wire_sha, "SDK wire request differs from the quoted request")
    request_ids = {item.get("request", {}).get("id") for item in reply}
    require(
        len(request_ids) == 1 and None not in request_ids and "" not in request_ids, "Missing/inconsistent request ID"
    )
    request_id = next(iter(request_ids))
    require(request_id == wire["headers"].get("x-sie-request-id"), "Body/header request IDs differ")
    require(
        isinstance(execution, str) and re.fullmatch(r"[0-9a-f]{64}", execution), "Missing execution bundle revision"
    )
    require(execution == wire["headers"].get("x-sie-model-revision"), "Execution revision differs")
    # Execution bundle SHA64 is deliberately NOT compared with the HF weights SHA40.
    charge = charge_from_wire(wire)
    require(charge is not None, "Response has no settled charge; it is not free")
    estimate = decimal(quote.get("estimated_credits_decimal", quote.get("estimated_credits")))
    require(charge <= estimate, "Settled charge exceeds the quoted budget ceiling")
    rate_book = (wire.get("usage") or {}).get("rate_book_version")
    require(rate_book == quote.get("rate_book_version"), "Rate book changed between quote and dispatch")
    by_id = {item["id"]: item for item in reply}
    vectors = np.asarray([by_id[row["id"]]["dense"] for row in rows], dtype=np.float64)
    require(vectors.shape == (len(rows), DIMENSIONS) and np.isfinite(vectors).all(), "Bad vector shape or values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    require(np.isfinite(norms).all() and (norms > 0).all(), "Zero/nonfinite vector norm")
    return (vectors / norms).astype(np.float32), {
        "request_id": request_id,
        "model": MODEL,
        "execution_bundle_sha256": execution,
        "catalog_hf_revision": REVISION,
        "credits_decimal": str(charge),
        "usd_at_declared_credit_price": str(charge * CREDIT_USD),
        "rate_book_version": rate_book,
        "usage": wire["usage"],
        "request_metadata": [item["request"] for item in reply],
    }


def collect_batches(rows, output, plan, client, capture, *, commit=lambda: None):
    """One sequential attempt. Any failure stops; this never resumes intents."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    require(len(rows) == len({row["id"] for row in rows}) == plan["n"], "Wrong input IDs/count")
    require(all(isinstance(row["id"], str) and row["id"] for row in rows), "Invalid input ID")
    require(all(isinstance(row["request"], str) and row["request"].strip() for row in rows), "Empty request")
    require(type(plan["batch_size"]) is int and 1 <= plan["batch_size"] <= 16, "Invalid batch size")
    require(decimal(plan["budget_usd"]) > 0 and plan["max_seconds"] > 180, "Invalid budget/deadline")
    require(plan["instruction"] == INSTRUCTION, "Frozen task-neutral instruction changed")
    require(
        plan.get("model", MODEL) == MODEL
        and plan.get("hf_revision", REVISION) == REVISION
        and plan.get("dimensions", DIMENSIONS) == DIMENSIONS,
        "Planned encoder identity changed",
    )
    write(output / "plan.json", plan)
    commit()
    started, known, completed, execution = time.monotonic(), Decimal(0), [], None
    terminal = {
        "status": "incomplete",
        "expected_rows": len(rows),
        "complete_rows": 0,
        "known_credits_decimal": "0",
        "unknown_charge_batches": [],
        "batch_receipts": [],
    }
    current, quote, intent, dispatched = None, None, None, False
    try:
        metadata = client.get_model(MODEL)
        require(
            metadata.get("revision") == REVISION and metadata.get("dims", {}).get("dense") == DIMENSIONS,
            "Catalog model differs from the pinned encoder",
        )
        require(metadata.get("name", metadata.get("id", MODEL)) == MODEL, "Catalog model ID differs")
        write(output / "catalog.json", metadata)
        commit()
        for number, offset in enumerate(range(0, len(rows), plan["batch_size"])):
            require(time.monotonic() - started < plan["max_seconds"] - 180, "Internal collection deadline reached")
            batch = rows[offset : offset + plan["batch_size"]]
            current, quote, intent, dispatched = output / "batches" / f"{number:06d}", None, None, False
            capture.receipt = None
            body = body_for(batch, plan["instruction"])
            body_bytes = packb(body, use_bin_type=True)
            quote = client.estimate(f"/v1/encode/{MODEL}", body, timeout=30)
            require(
                quote.get("identity", {}).get("model") == MODEL and quote["identity"].get("operation") == "encode",
                "Quote identity changed",
            )
            require(
                isinstance(quote.get("rate_book_version"), str)
                and re.fullmatch(r"[0-9a-f]{64}", quote.get("rate_book_sha256", "")),
                "Missing quoted price-book binding",
            )
            ceiling = decimal(quote.get("estimated_credits_decimal", quote.get("estimated_credits")))
            require(known + ceiling <= decimal(plan["budget_usd"]) / CREDIT_USD, "Next batch exceeds remaining budget")
            prepared = {
                "batch": number,
                "ids": [row["id"] for row in batch],
                "request": body,
                "request_sha256": hashlib.sha256(encoded(body)).hexdigest(),
                "expected_wire_sha256": hashlib.sha256(body_bytes).hexdigest(),
                "quote": quote,
                "physical_attempt": "one; never automatically repeated",
            }
            write(current / "intent.json", prepared)
            intent = prepared
            commit()  # A durable intent must precede any chargeable request.
            capture.arm(current)
            before = time.monotonic()
            dispatched = True
            reply = client.encode(
                MODEL,
                body["items"],
                output_types=["dense"],
                instruction=plan["instruction"],
                output_dtype="float32",
                is_query=True,
                wait_for_capacity=False,
                max_oom_retries=0,
                provision_timeout_s=150,
            )
            vectors, receipt = validate_reply(
                reply, batch, capture.receipt, client.last_model_revision, quote, intent["expected_wire_sha256"]
            )
            require(client.last_retry_count == 0, "Unexpected SDK retry")
            require(
                execution is None or execution == receipt["execution_bundle_sha256"],
                "Execution bundle changed mid-collection",
            )
            execution = receipt["execution_bundle_sha256"]
            with (current / "vectors.npy.pending").open("xb") as stream:
                np.save(stream, vectors, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            (current / "vectors.npy.pending").replace(current / "vectors.npy")
            receipt.update(
                status="complete",
                batch=number,
                ids=intent["ids"],
                elapsed_seconds=time.monotonic() - before,
                intent_sha256=sha(current / "intent.json"),
                response_sha256=sha(current / "response.bin"),
                http_sha256=sha(current / "http.json"),
                vectors_sha256=sha(current / "vectors.npy"),
            )
            write(current / "receipt.json", receipt)
            known += decimal(receipt["credits_decimal"])
            completed.append(receipt)
            terminal.update(
                complete_rows=sum(len(item["ids"]) for item in completed),
                known_credits_decimal=str(known),
                batch_receipts=[
                    {
                        "path": f"batches/{item['batch']:06d}/receipt.json",
                        "sha256": sha(output / "batches" / f"{item['batch']:06d}" / "receipt.json"),
                    }
                    for item in completed
                ],
            )
            write(output / "progress.json", terminal, replace=True)
            commit()  # Vectors AND charge are durable before the next batch starts.
            print(f"Embedded {terminal['complete_rows']}/{len(rows)}; settled credits {known}", flush=True)
            current, quote, intent, dispatched = None, None, None, False
        terminal["status"] = "complete"
    except Exception as error:
        terminal.update(error_type=type(error).__name__, error=str(error))
        if current is not None and intent is not None and dispatched:
            try:
                charge = charge_from_wire(capture.receipt)
            except ValueError:
                charge = None
            failed = {
                "status": "failed_or_unknown",
                "ids": intent["ids"],
                "error_type": type(error).__name__,
                "intent_sha256": sha(current / "intent.json"),
                "credits_decimal": None if charge is None else str(charge),
                "quoted_credit_ceiling": str(
                    decimal(quote.get("estimated_credits_decimal", quote.get("estimated_credits")))
                ),
                "response_received": capture.receipt is not None,
                "retry_permitted": False,
            }
            if not (current / "receipt.json").exists():
                write(current / "failure.json", failed)
                if charge is None:
                    terminal["unknown_charge_batches"].append(failed)
                else:
                    known += charge
            terminal["known_credits_decimal"] = str(known)
    finally:
        terminal.update(
            elapsed_seconds=time.monotonic() - started,
            execution_bundle_sha256=execution,
            plan_sha256=sha(output / "plan.json"),
            ordered_ids_sha256=hashlib.sha256(encoded([row["id"] for row in rows])).hexdigest(),
            catalog_hf_revision=REVISION,
            price_usd_per_credit=str(CREDIT_USD),
            known_usd_at_declared_credit_price=str(known * CREDIT_USD),
            estimated_unknown_usd_upper_bound=str(
                sum((decimal(x["quoted_credit_ceiling"]) for x in terminal["unknown_charge_batches"]), Decimal(0))
                * CREDIT_USD
            ),
            cost_scope="Observed batch charges, not singleton deployment fees. Unknown charges are not zero.",
            runtime_scope="Question-only input packet; the shared artifact Volume is technically accessible.",
        )
        write(output / "collection.json", terminal)
        commit()
    return terminal


def load_vectors(output, rows):
    """Verify a complete collection and return aligned IDs, normalized vectors, metadata."""
    output = Path(output)
    collection = json.loads((output / "collection.json").read_text())
    require(collection["status"] == "complete" and not collection["unknown_charge_batches"], "Collection incomplete")
    require(sha(output / "plan.json") == collection["plan_sha256"], "Collection plan changed")
    ids = [row["id"] for row in rows]
    require(hashlib.sha256(encoded(ids)).hexdigest() == collection["ordered_ids_sha256"], "Input order changed")
    accumulated, vectors, charges = [], [], Decimal(0)
    for number, item in enumerate(collection["batch_receipts"]):
        require(item["path"] == f"batches/{number:06d}/receipt.json", "Noncanonical batch path")
        receipt_path = output / item["path"]
        require(sha(receipt_path) == item["sha256"], "Batch receipt changed")
        receipt = json.loads(receipt_path.read_text())
        require(
            receipt["status"] == "complete"
            and receipt["model"] == MODEL
            and receipt["catalog_hf_revision"] == REVISION
            and receipt["execution_bundle_sha256"] == collection["execution_bundle_sha256"],
            "Batch identity changed",
        )
        for name, key in (
            ("intent.json", "intent_sha256"),
            ("response.bin", "response_sha256"),
            ("http.json", "http_sha256"),
            ("vectors.npy", "vectors_sha256"),
        ):
            require(sha(receipt_path.parent / name) == receipt[key], f"Batch {name} changed")
        intent = json.loads((receipt_path.parent / "intent.json").read_text())
        count = len(receipt["ids"])
        expected = rows[len(accumulated) : len(accumulated) + count]
        require(
            intent["request"] == body_for(expected, INSTRUCTION) and intent["ids"] == receipt["ids"],
            "Vector request differs from exact input text",
        )
        matrix = np.load(receipt_path.parent / "vectors.npy", allow_pickle=False)
        require(
            matrix.dtype == np.float32 and matrix.shape == (count, DIMENSIONS) and np.isfinite(matrix).all(),
            "Invalid stored vectors",
        )
        require(
            np.allclose(np.linalg.norm(matrix.astype(np.float64), axis=1), 1, rtol=1e-6, atol=1e-7),
            "Stored vectors are not normalized",
        )
        accumulated.extend(receipt["ids"])
        vectors.append(matrix)
        charges += decimal(receipt["credits_decimal"])
    require(
        accumulated == ids and len(ids) == collection["complete_rows"] == collection["expected_rows"],
        "Incomplete vector ID alignment",
    )
    require(charges == decimal(collection["known_credits_decimal"]), "Collection charge sum differs")
    return ids, np.concatenate(vectors), collection


def make_client(capture):
    http = httpx.Client(base_url=BASE_URL, follow_redirects=False, timeout=150, event_hooks={"response": [capture]})
    return SingleSendClient(
        base_url=BASE_URL,
        api_key=os.environ["SIE_API_KEY"],
        timeout_s=150,
        read_timeout_s=150,
        connect_timeout_s=15,
        http_client=http,
    )
