"""Shared wiring for this example; Cognee is imported only after configuration."""

import asyncio
import importlib
import json
import os
import re
import tomllib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn
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
MEMORY_STATE_RECOVERY = (
    "Stored memory state is missing, unreadable, invalid or mismatched. Restore the original intact memory "
    "directory and teaching-run artifacts with matching namespace and embedding settings, or use a fresh "
    "WORKSHOP_NAMESPACE and rebuild memory before verification."
)
CONFIGURATION_MESSAGES = {
    "endpoint_url": "Endpoint must be an HTTP(S) URL without credentials, query or fragment, and with a valid port.",
    "tutorial_inputs": "Download the tutorial inputs first with uv run fetch.py, then rerun the experiment.",
    **{
        f"{role}_key": f"Set SIE_{role.upper()}_API_KEY or SIE_API_KEY for the Serverless {role} endpoint."
        for role in ("llm", "embedding", "ocr", "rerank", "decision")
    },
}


class ConfigurationError(ValueError):
    """An authored setup diagnostic selected by a fixed code, never upstream text."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(CONFIGURATION_MESSAGES.get(code, "Check the endpoint configuration."))


class MemoryStateError(ValueError):
    """Local stored state cannot safely be reused with this configuration."""


def read_state_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise MemoryStateError(MEMORY_STATE_RECOVERY) from None
    if not isinstance(value, dict):
        raise MemoryStateError(MEMORY_STATE_RECOVERY)
    return value


def record_failure(run: Path, error: Exception) -> bool:
    """Record only the error type, without replacing recovery if the disk is unwritable."""
    try:
        (run / "failure.json").write_text(
            json.dumps({"type": type(error).__name__, "completed": False}) + "\n", encoding="utf-8"
        )
    except (OSError, UnicodeError):
        return False
    return True


def stop_run(run: Path, error: Exception) -> NoReturn:
    saved = record_failure(run, error)
    recovery = (
        MEMORY_STATE_RECOVERY
        if isinstance(error, MemoryStateError)
        else "Check endpoint availability, cold-start budget and context limits. No success was recorded."
    )
    if isinstance(error, ConfigurationError):
        recovery = CONFIGURATION_MESSAGES.get(error.code, "Check the endpoint configuration.")
    artifact = f" Inspect {run / 'failure.json'}." if saved else " Could not write failure.json."
    raise SystemExit(f"{type(error).__name__}: {recovery}{artifact}") from None


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
    try:
        Endpoint(base_url=base).compatible_url()
        Endpoint(base_url=url).compatible_url()
        base_origin, url_origin = origin(base), origin(url)
    except ValueError:
        raise ConfigurationError("endpoint_url") from None
    key_name = f"SIE_{role.upper()}_API_KEY"
    key = env.get(key_name) if key_name in env else env.get("SIE_API_KEY") if base_origin == url_origin else None
    if url_origin == origin("https://api.superlinked.com") and not key:
        raise ConfigurationError(f"{role}_key")
    return Endpoint(base_url=url, api_key=key or None)


def require_input(path: Path) -> Path:
    if not path.is_file():
        raise ConfigurationError("tutorial_inputs")
    return path


@dataclass
class Demo:
    models: dict[str, Any]
    limits: dict[str, Any]
    decision: dict[str, Any]
    endpoints: dict[str, Endpoint]
    namespace: str
    storage: Path
    run: Path
    learning: dict[str, Any] = field(default_factory=dict)

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
        rendered = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False)
        (self.run / f"{name}.json").write_text(rendered + "\n", encoding="utf-8")
        print(f"\n--- {name} ---\n{rendered}", flush=True)


def load_demo(experiment: str) -> Demo:
    load_dotenv(ROOT / ".env", override=False)
    config = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
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
    if not config["learning"]["query"].strip():
        raise ValueError("The learning demonstration requires a nonempty query for the new session.")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run = ROOT / "runs" / namespace / f"{run_id}-{experiment}"
    run.mkdir(parents=True)
    try:
        storage = ROOT / ".state" / namespace
        identity = {key: config["models"][key] for key in ("embedding", "embedding_dimensions")}
        manifest = storage / "embedding.json"
        try:
            storage.mkdir(parents=True, exist_ok=True)
            if manifest.exists():
                if read_state_object(manifest) != identity:
                    raise MemoryStateError(MEMORY_STATE_RECOVERY)
            else:
                manifest.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        except (OSError, UnicodeError):
            raise MemoryStateError(MEMORY_STATE_RECOVERY) from None
        endpoints = {
            role: endpoint_for(role, dict(os.environ)) for role in ("llm", "embedding", "ocr", "rerank", "decision")
        }
        demo = Demo(
            config["models"],
            config["limits"],
            config["decision"],
            endpoints,
            namespace,
            storage,
            run,
            config["learning"],
        )
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
    except Exception as error:
        stop_run(run, error)
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
    inputs = [(source_id, require_input(ROOT / "data" / filename)) for source_id, filename in SEEDS]
    for source_id, path in inputs:
        document = SourceDocument(text=path.read_text(encoding="utf-8"), source_id=source_id, filename=path.name)
        demo.show(f"input-{source_id}", document)
        await cognee.remember(document.to_data_item(), dataset_name=demo.dataset, self_improvement=False)


def decision_chunks(chunks: list[SourceChunk]) -> list[SourceChunk]:
    """Keep source IDs in the artifact; use short aliases in the small decision context."""
    return [
        SourceChunk(
            chunk_id=f"E{index}",
            text=chunk.text,
            metadata={**chunk.metadata, "original_chunk_id": chunk.chunk_id},
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


async def run_decision(
    demo: Demo,
    query: str,
    *,
    prefix: str = "decision",
    decision_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    question = demo.decision if decision_config is None else decision_config
    abstention = "unknown" if "unknown" in question["choices"] else "review"
    chunks = await retrieve_chunks(query, dataset=demo.dataset, top_k=demo.limits["retrieve_top_k"])
    demo.show(f"{prefix}-source-candidates", chunks)
    if not chunks:
        outcome = {"status": "no_evidence", "choice": abstention, "probabilities": None}
        demo.show(f"{prefix}-outcome", outcome)
        return outcome
    with demo.native("rerank") as client:
        ranked = await asyncio.to_thread(
            rerank_chunks,
            client,
            query,
            chunks,
            model=demo.models["rerank"],
            top_k=question.get("rerank_top_k", demo.limits["rerank_top_k"]),
            provision_timeout_s=demo.limits["provision_timeout_seconds"],
        )
    demo.show(f"{prefix}-reranked", ranked)
    selected = decision_chunks([item.chunk for item in ranked])
    demo.show(f"{prefix}-evidence-map", selected)
    try:
        packet = evidence_packet(query, selected, max_chars=demo.limits["evidence_chars"])
    except EvidenceOverflowError:
        outcome = {"status": "evidence_overflow", "choice": abstention, "probabilities": None}
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
            choices=question["choices"],
            instructions=question["instructions"],
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
        stop_run(demo.run, error)
