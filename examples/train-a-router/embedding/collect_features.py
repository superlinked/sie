"""Question-only feature acquisition on SSD with compact durable checkpoints."""

from __future__ import annotations

from decimal import Decimal
import hashlib
from pathlib import Path
import time

import numpy as np

from embedding import recipes as core, token_audit, wire

RATE = {"rate_numerator": 3, "rate_denominator": 500, "unit": "input_tokens"}


def credits(tokens):
    return (tokens * 3 + 499) // 500


def source_rows(source, source_sha):
    source = Path(source)
    core.require(
        core.sha(source / "source-plan.json") == source_sha, "Frozen source changed"
    )
    plan = core.read(source / "source-plan.json")
    metadata = source / "meta.json"
    core.require(
        core.sha(metadata) == plan["files_sha256"]["meta.json"],
        "Source metadata changed",
    )
    by_id = {row["id"]: row for row in core.read(metadata)}
    rows, role_inputs = [], {}
    families = {}
    for role in core.ROLES:
        relative = f"question-only/{role}-inputs.json"
        path = source / relative
        core.require(
            core.sha(path) == plan["files_sha256"][relative], "Input hash changed"
        )
        role_inputs[role] = core.sha(path)
        for item in core.read(path):
            core.require(
                set(item) == {"id", "request", "response_contract"},
                "Question-only payload contains extra fields",
            )
            meta = by_id[item["id"]]
            core.require(
                meta["retained"] is True and meta["role"] == role and item["request"],
                "Source eligibility mismatch",
            )
            core.require(
                families.setdefault(meta["family"], role) == role,
                "Family crosses source roles",
            )
            rows.append({**item, "role": role, "task": meta["task"]})
    core.require(
        len(rows)
        == len({r["id"] for r in rows})
        == sum(r["retained"] for r in by_id.values()),
        "Active source population mismatch",
    )
    return rows, role_inputs


def singleton_rows(rows):
    development = [row for row in rows if row["role"] == "development"]
    order = sorted(
        development,
        key=lambda row: hashlib.sha256(
            ("bbeh-native-timing89:" + row["id"]).encode()
        ).hexdigest(),
    )
    core.require(len(order) >= 2, "Need a separate warmup and timed request")
    count = min(90, len(order) - 1)
    return [order[count], *order[:count]]


class TimedClient:
    def __init__(self, client):
        self.client, self.timings = client, []

    def __getattr__(self, name):
        return getattr(self.client, name)

    def encode(self, model, items, **kwargs):
        start = time.perf_counter()
        result = self.client.encode(model, items, **kwargs)
        self.timings.append(
            {
                "ids": [item["id"] for item in items],
                "seconds": time.perf_counter() - start,
            }
        )
        return result


def check_tokens(root, counts):
    collection = core.read(Path(root) / "collection.json")
    core.require(
        collection["status"] == "complete" and not collection["unknown_charge_batches"],
        "Complete settled features required",
    )
    pricebook = None
    for binding in collection["batch_receipts"]:
        path = Path(root) / binding["path"]
        core.require(core.sha(path) == binding["sha256"], "Batch binding changed")
        receipt, intent = core.read(path), core.read(path.parent / "intent.json")
        quote = intent["quote"]
        core.require(
            quote["applied_rates"] == [RATE]
            and quote["rounding_rule"] == "ceil-once-per-terminal-event"
            and quote["minimum_billed_units"] is None,
            "Encoder pricing basis changed",
        )
        binding = {key: quote[key] for key in ("rate_book_version", "rate_book_sha256")}
        core.require(
            pricebook is None or pricebook == binding,
            "Pricebook changed across batches",
        )
        pricebook = binding
        tokens = sum(counts[item] for item in receipt["ids"])
        core.require(
            receipt["usage"]["input_tokens"] == tokens
            and Decimal(receipt["credits_decimal"]) == credits(tokens),
            "Exact token/fee projection differs from observed charge",
        )
    return pricebook


def run(
    source,
    output,
    plan,
    tokenizer,
    tokenizer_binding,
    checkpoint,
    *,
    client_factory=wire.make_client,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(source)
    rows, input_hashes = source_rows(source, plan["source_plan_sha256"])
    core.require(len(rows) == plan["n"], "Launch/source count changed")
    counts = {}
    for role in core.ROLES:
        selected = [row for row in rows if row["role"] == role]
        lengths = token_audit.audit_rows(selected, tokenizer)
        audit = {
            "format": "bbeh-encoder-token-audit-v1",
            "role": role,
            "model": core.MODEL,
            "hf_revision": core.REVISION,
            "instruction": core.INSTRUCTION,
            "template": token_audit.TEMPLATE,
            "max_length": token_audit.MAX_LENGTH,
            "truncation": False,
            "source_plan_sha256": plan["source_plan_sha256"],
            "inputs_sha256": input_hashes[role],
            "tokenizer": tokenizer_binding,
            "rows": lengths,
            "truncated_ids": [],
            "overflow_ids": [
                r["id"] for r in lengths if r["input_tokens"] > token_audit.MAX_LENGTH
            ],
        }
        core.write(output / "features" / role / "token-audit.json", audit)
        core.require(
            not audit["overflow_ids"],
            "Encoder context overflow; stop before model calls, never truncate",
        )
        counts.update({row["id"]: row["input_tokens"] for row in lengths})
    core.write(output / "plan.json", plan)
    core.write(
        output / "question-inputs.json",
        [
            {k: r[k] for k in ("id", "request", "response_contract", "role", "task")}
            for r in rows
        ],
    )
    core.write(output / "tokenizer.json", tokenizer_binding)
    checkpoint(output)
    common = {
        "instruction": core.INSTRUCTION,
        "model": core.MODEL,
        "hf_revision": core.REVISION,
        "dimensions": 2560,
        "max_seconds": plan["collection_seconds"],
        "budget_usd": plan["budget_usd"],
    }
    capture = wire.WireCapture()
    with client_factory(capture) as client:
        batch = wire.collect_batches(
            rows,
            output / "batches",
            {**common, "n": len(rows), "batch_size": 16},
            client,
            capture,
            commit=lambda: checkpoint(output),
        )
        core.require(
            batch["status"] == "complete",
            "Batch collection stopped; retained intents must never be automatically repeated",
        )
        ids, matrix, _ = wire.load_vectors(output / "batches", rows)
        pricebook = check_tokens(output / "batches", counts)
        for role in core.ROLES:
            indices = [i for i, row in enumerate(rows) if row["role"] == role]
            directory = output / "features" / role
            with (directory / "vectors.npy").open("xb") as stream:
                np.save(stream, matrix[indices], allow_pickle=False)
            core.write(directory / "ids.json", [ids[i] for i in indices])
            core.write(
                directory / "features.json",
                {
                    "format": "routerbench-fixed-embeddings-v1",
                    "model": core.MODEL,
                    "hf_revision": core.REVISION,
                    "instruction": core.INSTRUCTION,
                    "dimensions": 2560,
                    "n": len(indices),
                    "inputs_sha256": input_hashes[role],
                    "source_plan_sha256": plan["source_plan_sha256"],
                    "ids_sha256": core.sha(directory / "ids.json"),
                    "vectors_sha256": core.sha(directory / "vectors.npy"),
                    "token_audit_sha256": core.sha(directory / "token-audit.json"),
                    "collection_sha256": core.sha(output / "batches/collection.json"),
                    "dtype": "float32",
                    "normalization": "float64 L2 then float32",
                    "outcomes_accessed": False,
                },
            )
        checkpoint(output)
        sample = singleton_rows(rows)
        core.write(
            output / "singleton-sample.json",
            {
                "warmup_id": sample[0]["id"],
                "timed_ids": [r["id"] for r in sample[1:]],
                "selection": "First90 development IDs by SHA256('bbeh-native-timing89:'+id); nextID is separate warmup executed first; all if fewer than91",
            },
        )
        remaining = Decimal(plan["budget_usd"]) - Decimal(
            batch["known_usd_at_declared_credit_price"]
        )
        timed = TimedClient(client)
        singles = wire.collect_batches(
            sample,
            output / "singletons",
            {**common, "n": len(sample), "batch_size": 1, "budget_usd": str(remaining)},
            timed,
            capture,
            commit=lambda: checkpoint(output),
        )
        core.require(
            singles["status"] == "complete",
            "Singleton validation incomplete; preserve all completed features",
        )
        core.require(
            check_tokens(output / "singletons", counts) == pricebook,
            "Singleton pricebook differs from batches",
        )
        durations = [row["seconds"] for row in timed.timings[1:]]
        core.write(
            output / "singleton-timing.json",
            {
                "model": core.MODEL,
                "hf_revision": core.REVISION,
                "sample_sha256": core.sha(output / "singleton-sample.json"),
                "timings": timed.timings,
                "mean_ms": float(np.mean(durations) * 1000),
                "p50_ms": float(np.median(durations) * 1000),
                "median_ms": float(np.median(durations) * 1000),
                "p95_ms": float(np.quantile(durations, 0.95) * 1000),
                "scope": "One SDK encode request, including synchronous raw-response SSD persistence; excludes quote/catalog/Volume commits and downstream head scoring. One warmup excluded.",
                "batch_scoring_or_solver_calls": False,
            },
        )
    projection = {
        "format": "bbeh-singleton-embedding-fees-v1",
        "source_plan_sha256": plan["source_plan_sha256"],
        "model": core.MODEL,
        "hf_revision": core.REVISION,
        "instruction": core.INSTRUCTION,
        "cost_basis": "validated_singleton_projection",
        "pricebook": pricebook,
        "batch_collection_sha256": core.sha(output / "batches/collection.json"),
        "singleton_collection_sha256": core.sha(output / "singletons/collection.json"),
        "all_observed_batch_and_singleton_tokens_and_charges_reproduced": True,
        "rows": [
            {
                "id": row["id"],
                "role": row["role"],
                "input_tokens": counts[row["id"]],
                "projected_singleton_credits": credits(counts[row["id"]]),
                "projected_singleton_usd": str(
                    wire.CREDIT_USD * credits(counts[row["id"]])
                ),
            }
            for row in rows
        ],
    }
    core.write(output / "fee-projection.json", projection)
    total = Decimal(batch["known_usd_at_declared_credit_price"]) + Decimal(
        singles["known_usd_at_declared_credit_price"]
    )
    terminal = {
        "status": "complete",
        "n": len(rows),
        "known_api_usd": str(total),
        "unknown_charge_batches": [],
        "batch_rows": len(rows),
        "singleton_requests_including_warmup": len(sample),
        "fee_projection_sha256": core.sha(output / "fee-projection.json"),
        "features": {
            role: core.sha(output / "features" / role / "features.json")
            for role in core.ROLES
        },
    }
    core.write(output / "terminal.json", terminal)
    checkpoint(output)
    return terminal
