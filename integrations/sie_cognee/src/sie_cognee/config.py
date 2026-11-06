"""Configure Cognee's public compatible transports before importing Cognee."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class Endpoint:
    """One SIE endpoint; compatible clients require a key even on localhost."""

    base_url: str
    api_key: str | None = field(default=None, repr=False)

    def compatible_url(self) -> str:
        parts = urlsplit(self.base_url.strip())
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError("Endpoint must be an HTTP(S) URL without credentials, query or fragment")
        path = parts.path.rstrip("/")
        if not path.endswith("/v1"):
            path += "/v1"
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))

    def compatible_key(self) -> str:
        self.compatible_url()
        if self.api_key and self.api_key.strip():
            return self.api_key
        if urlsplit(self.base_url).hostname in {"localhost", "127.0.0.1", "::1"}:
            return "sie-local-no-key"
        raise ValueError("Remote compatible endpoints require an explicit API key")


@dataclass(frozen=True)
class CogneeSettings:
    """One process's Cognee inference routes and dedicated local storage."""

    llm: Endpoint
    embedding: Endpoint
    llm_model: str
    embedding_model: str
    embedding_dimensions: int
    storage_root: Path
    auto_feedback: bool = False


def configure(settings: CogneeSettings) -> None:
    """Initialize isolated Cognee 1.6.7 settings once, before any Cognee import.

    Explicitly owns Cognee's LLM/embedding routing, local storage and session
    cache settings. No model is loaded or called here. Start a fresh process to
    change configuration, and use a new storage directory for different vectors.
    """
    if "cognee" in sys.modules:
        raise RuntimeError("Call configure before importing Cognee; use a fresh process to change settings")
    dimensions = settings.embedding_dimensions
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
        raise ValueError("embedding_dimensions must be a positive integer")
    if not settings.llm_model.strip() or not settings.embedding_model.strip():
        raise ValueError("Both model identifiers are required")
    llm_model = settings.llm_model.removeprefix("openai/")
    if settings.embedding_model.startswith("openai/"):
        raise ValueError("embedding_model must be the raw SIE model identifier")
    llm_config: dict[str, object] = {
        "llm_provider": "custom",
        "llm_model": f"openai/{llm_model}",
        "llm_endpoint": settings.llm.compatible_url(),
        "llm_api_key": settings.llm.compatible_key(),
        "llm_api_version": "",
        "structured_output_framework": "litellm_native",
        "fallback_model": "",
        "fallback_endpoint": "",
        "fallback_api_key": "",
    }
    # Prevent stage-specific or fallback routes from silently overriding SIE.
    for stage in ("extraction", "summarization", "query"):
        for name in ("provider", "model", "endpoint", "api_key", "api_version"):
            llm_config[f"llm_{stage}_{name}"] = ""
    conflicts = [key.upper() for key, value in llm_config.items() if value == "" and os.environ.get(key.upper())]
    if os.environ.get("LLM_ARGS", "").strip():
        conflicts.append("LLM_ARGS")
    if conflicts:
        raise ValueError(f"Unset conflicting Cognee inference routes: {', '.join(sorted(conflicts))}")
    embedding_config: dict[str, object] = {
        "embedding_provider": "openai_compatible",
        "embedding_model": settings.embedding_model,
        "embedding_endpoint": settings.embedding.compatible_url(),
        "embedding_api_key": settings.embedding.compatible_key(),
        "embedding_dimensions": dimensions,
    }
    root = settings.storage_root.expanduser().resolve()
    marker = root / "sie-cognee.json"
    identity = {"embedding_model": settings.embedding_model, "embedding_dimensions": dimensions}
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Storage already uses different embeddings; choose a new storage_root")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Use a new or empty storage_root; existing unowned storage will not be changed")
    root.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        with marker.open("x") as stream:
            json.dump(identity, stream)
    system_root = root / "system"
    data_root = root / "data"
    system_root.mkdir(exist_ok=True)
    data_root.mkdir(exist_ok=True)
    environment = {
        **{key.upper(): str(value) for key, value in llm_config.items()},
        **{key.upper(): str(value) for key, value in embedding_config.items()},
        "GRAPH_EXTRACTOR": "llm",
        "LLM_ARGS": "",
        "MOCK_EMBEDDING": "false",
        "SYSTEM_ROOT_DIRECTORY": str(system_root),
        "COGNEE_LOGS_DIR": str(root / "logs"),
        "DATA_ROOT_DIRECTORY": str(data_root),
        "DB_PROVIDER": "sqlite",
        "DB_PATH": str(system_root / "databases"),
        "DB_NAME": "cognee_db",
        "GRAPH_DATABASE_PROVIDER": "ladybug",
        "GRAPH_DATASET_DATABASE_HANDLER": "ladybug",
        "GRAPH_FILE_PATH": str(system_root / "databases"),
        "GRAPH_FILENAME": "",
        "VECTOR_DB_PROVIDER": "lancedb",
        "VECTOR_DATASET_DATABASE_HANDLER": "lancedb",
        "VECTOR_DB_URL": str(system_root / "databases" / "cognee.lancedb"),
        "CACHING": "true",
        "AUTO_FEEDBACK": str(settings.auto_feedback).lower(),
        "SESSION_SEARCH_MODE": "sequential",
        "CACHE_BACKEND": "sqlite",
        "CACHE_DB_URL": f"sqlite+aiosqlite:///{system_root / 'cache.db'}",
    }
    os.environ.update(environment)
    import cognee

    cognee.config.set_llm_config(llm_config)
    cognee.config.set_embedding_config(embedding_config)
    cognee.config.system_root_directory(str(system_root))
    cognee.config.data_root_directory(str(data_root))
