"""Accept observed Cloud rounding labels without changing the frozen fee rule."""

from decimal import Decimal
from pathlib import Path

from embedding import collect_features
from embedding import recipes as core

ROUNDING_LABELS = {"ceil-once-per-terminal-event", "ceil-once-per-terminal-event-micro"}


def check_tokens(root, counts):
    """Validate actual settlement; a rounding label never substitutes for a charge."""
    root = Path(root)
    collection = core.read(root / "collection.json")
    core.require(
        collection["status"] == "complete" and not collection["unknown_charge_batches"],
        "Complete settled features required",
    )
    pricebook = None
    for binding in collection["batch_receipts"]:
        path = root / binding["path"]
        core.require(core.sha(path) == binding["sha256"], "Batch binding changed")
        receipt = core.read(path)
        intent_path = path.parent / "intent.json"
        core.require(core.sha(intent_path) == receipt["intent_sha256"], "Intent changed")
        intent = core.read(intent_path)
        quote = intent["quote"]
        core.require(
            quote["applied_rates"] == [collect_features.RATE]
            and quote["rounding_rule"] in ROUNDING_LABELS
            and quote["minimum_billed_units"] is None,
            "Encoder pricing basis changed",
        )
        core.require(receipt["ids"] == intent["ids"], "Batch IDs changed")
        current = {key: quote[key] for key in ("rate_book_version", "rate_book_sha256")}
        core.require(pricebook is None or pricebook == current, "Pricebook changed across batches")
        pricebook = current
        tokens = sum(counts[item] for item in receipt["ids"])
        core.require(
            receipt["usage"]["input_tokens"] == tokens
            and Decimal(receipt["credits_decimal"]) == collect_features.credits(tokens),
            "Exact token/fee projection differs from observed charge",
        )
        core.require(
            receipt["usage"]["rate_book_version"] == quote["rate_book_version"],
            "Usage/quote pricebook differs",
        )
    return pricebook


def run(*args, **kwargs):
    """Use the reader check during this sequential CLI collection, then restore it."""
    original = collect_features.check_tokens
    collect_features.check_tokens = check_tokens
    try:
        return collect_features.run(*args, **kwargs)
    finally:
        collect_features.check_tokens = original
