"""Shared wiring for this example; Cognee is imported only after configuration."""

import asyncio
import importlib
import json
import os
import re
import tomllib
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import load_dotenv
from sie_cognee.config import CogneeSettings, Endpoint, configure
from sie_cognee.decision import EvidenceOverflowError, decide, evidence_packet
from sie_cognee.documents import SourceDocument
from sie_cognee.retrieval import SourceChunk, rerank_chunks, retrieve_chunks
from sie_sdk import SIEClient

ROOT = Path(__file__).resolve().parent
SEEDS = (
    ("POLICY-01", "readiness-policy.txt"),
    ("CHECK-01", "projector-check.txt"),
    ("CHECK-02", "microphone-check.txt"),
)


def json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported artifact type: {type(value).__name__}")


def origin(url: str) -> tuple[str, str | None, int | None]:
    parsed = urlsplit(url)
    return parsed.scheme.lower(), parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def endpoint_for(role: str, env: dict[str, str]) -> Endpoint:
    base = env.get("SIE_BASE_URL", "https://api.superlinked.com")
    url = env.get(f"SIE_{role.upper()}_BASE_URL", base)
    key_name = f"SIE_{role.upper()}_API_KEY"
    key = env.get(key_name) if key_name in env else env.get("SIE_API_KEY") if origin(base) == origin(url) else None
    if origin(url) == origin("https://api.superlinked.com") and not key:
        raise ValueError(f"Set {key_name} or SIE_API_KEY for the Serverless {role} endpoint.")
    endpoint = Endpoint(base_url=url, api_key=key or None)
    endpoint.compatible_url()  # Validate before an endpoint can appear in a printed artifact.
    return endpoint


@dataclass
class Demo:
    models: dict[str, Any]
    limits: dict[str, Any]
    decision: dict[str, Any]
    endpoints: dict[str, Endpoint]
    namespace: str
    storage: Path
    run: Path

    @property
    def dataset(self) -> str:
        return f"sie-{self.namespace}"

    def native(self, role: str) -> SIEClient:
        endpoint = self.endpoints[role]
        return SIEClient(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key or "",
            timeout_s=self.limits["request_timeout_seconds"],
        )

    def show(self, name: str, value: Any) -> None:
        payload = json_value(value)
        rendered = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        (self.run / f"{name}.json").write_text(rendered + "\n")
        print(f"\n--- {name} ---\n{rendered}", flush=True)


def load_demo(experiment: str) -> Demo:
    load_dotenv(ROOT / ".env", override=False)
    config = tomllib.loads((ROOT / "config.toml").read_text())
    for role in ("llm", "embedding", "ocr", "rerank", "decision"):
        if model := os.environ.get(f"SIE_{role.upper()}_MODEL"):
            config["models"][role] = model
    if dimensions := os.environ.get("SIE_EMBEDDING_DIMENSIONS"):
        config["models"]["embedding_dimensions"] = int(dimensions)
    namespace = os.environ.get("WORKSHOP_NAMESPACE", "workshop-v1")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,49}", namespace):
        raise ValueError("WORKSHOP_NAMESPACE must contain 1–50 lowercase letters, digits or hyphens.")
    for name, value in config["limits"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"limits.{name} must be positive.")
    if config["decision"]["query"] == config["decision"]["new_query"]:
        raise ValueError("The learning demonstration requires a distinct new_query.")
    storage = ROOT / ".state" / namespace
    storage.mkdir(parents=True, exist_ok=True)
    identity = {key: config["models"][key] for key in ("embedding", "embedding_dimensions")}
    manifest = storage / "embedding.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Embedding configuration changed. Choose a fresh WORKSHOP_NAMESPACE.")
    manifest.write_text(json.dumps(identity, indent=2) + "\n")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run = ROOT / "runs" / namespace / f"{run_id}-{experiment}"
    run.mkdir(parents=True)
    endpoints = {
        role: endpoint_for(role, dict(os.environ)) for role in ("llm", "embedding", "ocr", "rerank", "decision")
    }
    demo = Demo(config["models"], config["limits"], config["decision"], endpoints, namespace, storage, run)
    demo.show(
        "configuration",
        {
            "models": demo.models,
            "endpoints": {role: endpoint.base_url for role, endpoint in endpoints.items()},
            "dataset": demo.dataset,
            "storage": str(storage.relative_to(ROOT)),
            "run": str(run.relative_to(ROOT)),
        },
    )
    return demo


def initialize(demo: Demo, *, learning: bool = False) -> ModuleType:
    configure(
        CogneeSettings(
            llm=demo.endpoints["llm"],
            embedding=demo.endpoints["embedding"],
            llm_model=demo.models["llm"],
            embedding_model=demo.models["embedding"],
            embedding_dimensions=demo.models["embedding_dimensions"],
            storage_root=demo.storage / "cognee",
            auto_feedback=learning,
        )
    )
    return importlib.import_module("cognee")


async def seed_memory(demo: Demo, cognee: ModuleType) -> None:
    for source_id, filename in SEEDS:
        document = SourceDocument(text=(ROOT / "data" / filename).read_text(), source_id=source_id, filename=filename)
        demo.show(f"input-{source_id}", document)
        await cognee.remember(document.to_data_item(), dataset_name=demo.dataset, self_improvement=False)


def decision_chunks(chunks: list[SourceChunk]) -> list[SourceChunk]:
    """Keep source IDs in the artifact; use short aliases in the small decision context."""
    return [
        SourceChunk(
            chunk_id=f"E{index}",
            text=("[model-generated lesson] " if "# Session learning (session " in chunk.text else "") + chunk.text,
            metadata={**chunk.metadata, "original_chunk_id": chunk.chunk_id},
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


async def run_decision(demo: Demo, query: str, *, prefix: str = "decision") -> dict[str, Any]:
    chunks = await retrieve_chunks(query, dataset=demo.dataset, top_k=demo.limits["retrieve_top_k"])
    demo.show(f"{prefix}-source-candidates", chunks)
    if not chunks:
        outcome = {"status": "no_evidence", "choice": "review", "probabilities": None}
        demo.show(f"{prefix}-outcome", outcome)
        return outcome
    with demo.native("rerank") as client:
        ranked = await asyncio.to_thread(
            rerank_chunks,
            client,
            query,
            chunks,
            model=demo.models["rerank"],
            top_k=demo.limits["rerank_top_k"],
            provision_timeout_s=demo.limits["provision_timeout_seconds"],
        )
    demo.show(f"{prefix}-reranked", ranked)
    selected = decision_chunks([item.chunk for item in ranked])
    demo.show(f"{prefix}-evidence-map", selected)
    try:
        packet = evidence_packet(query, selected, max_chars=demo.limits["evidence_chars"])
    except EvidenceOverflowError:
        outcome = {"status": "evidence_overflow", "choice": "review", "probabilities": None}
        demo.show(f"{prefix}-outcome", outcome)
        print("Evidence exceeds the budget. Reduce top-k or choose a larger-context model; nothing was truncated.")
        return outcome
    demo.show(f"{prefix}-exact-packet", packet)
    with demo.native("decision") as client:
        result = await asyncio.to_thread(
            decide,
            client,
            packet,
            model=demo.models["decision"],
            choices=demo.decision["choices"],
            instructions=demo.decision["instructions"],
            max_chars=demo.limits["evidence_chars"],
            provision_timeout_s=demo.limits["provision_timeout_seconds"],
        )
    outcome = {"status": "model_result", "choice": result.choice, "probabilities": result.probabilities}
    demo.show(f"{prefix}-outcome", outcome)
    return outcome


def launch(experiment: str, action: Callable[[Demo], Awaitable[None]]) -> None:
    demo = load_demo(experiment)

    async def bounded() -> None:
        async with asyncio.timeout(demo.limits["experiment_timeout_seconds"]):
            await action(demo)

    try:
        asyncio.run(bounded())
    except Exception as error:
        # Upstream errors can contain request headers. Persist the type, not secrets.
        demo.show("failure", {"type": type(error).__name__, "completed": False})
        raise SystemExit(
            f"{experiment} stopped: {type(error).__name__}. Inspect the completed artifacts in {demo.run}; "
            "check endpoint availability, cold-start budget and context limits. No success was recorded."
        ) from None
