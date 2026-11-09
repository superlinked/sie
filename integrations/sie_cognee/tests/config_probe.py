"""Fresh-process initialization against installed Cognee, with no network or models."""

import asyncio
import faulthandler
import logging
import os
import socket
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sie_cognee.config import CogneeSettings, Endpoint, configure

faulthandler.dump_traceback_later(30)
root = Path(sys.argv[1])
scenario = sys.argv[2]
attempts = []


def block_network(event, args):
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"} or (
        event in {"socket.connect", "socket.sendto"} and args[0].family in {socket.AF_INET, socket.AF_INET6}
    ):
        attempts.append(event)
        raise AssertionError("External network activity is forbidden in this probe")


sys.addaudithook(block_network)
project = root.parent / "project"
cwd = project / "nested"
cwd.mkdir(parents=True)
os.chdir(cwd)
outside = root.parent / "unselected-storage"
unselected_log = root.parent / "unselected.log"
os.environ["LOG_FILE_NAME"] = str(unselected_log)
hostile = {
    "LLM_PROVIDER": "anthropic",
    "LLM_MODEL": "unselected",
    "LLM_ENDPOINT": "https://unselected.invalid/v1",
    "LLM_API_KEY": "synthetic-unselected-key",
    "LLM_API_VERSION": "unselected",
    "STRUCTURED_OUTPUT_FRAMEWORK": "instructor",
    "LLM_ARGS": '{"api_base":"https://unselected.invalid"}',
    "FALLBACK_MODEL": "unselected",
    "FALLBACK_ENDPOINT": "https://unselected.invalid",
    "FALLBACK_API_KEY": "synthetic-unselected-key",
    "EMBEDDING_PROVIDER": "ollama",
    "EMBEDDING_MODEL": "unselected",
    "EMBEDDING_ENDPOINT": "https://unselected.invalid",
    "EMBEDDING_API_KEY": "synthetic-unselected-key",
    "EMBEDDING_DIMENSIONS": "999",
    "SYSTEM_ROOT_DIRECTORY": str(outside / "system"),
    "DATA_ROOT_DIRECTORY": str(outside / "data"),
    "COGNEE_LOGS_DIR": str(outside / "logs"),
    "DB_PROVIDER": "postgres",
    "DB_PATH": str(outside / "relational"),
    "DB_NAME": "unselected",
    "GRAPH_DATABASE_PROVIDER": "neo4j",
    "GRAPH_DATASET_DATABASE_HANDLER": "neo4j",
    "GRAPH_FILE_PATH": str(outside / "graph"),
    "GRAPH_FILENAME": "unselected",
    "VECTOR_DB_PROVIDER": "pgvector",
    "VECTOR_DATASET_DATABASE_HANDLER": "pgvector",
    "VECTOR_DB_URL": str(outside / "vector"),
    "GRAPH_EXTRACTOR": "unselected",
    "MOCK_EMBEDDING": "true",
    "CACHING": "false",
    "AUTO_FEEDBACK": "false",
    "SESSION_SEARCH_MODE": "concurrent",
    "CACHE_BACKEND": {"project": "redis", "explicit": "postgres", "hot_reload": "tapes"}[scenario],
    "CACHE_HOST": "unselected.invalid",
    "CACHE_DB_URL": "postgresql+asyncpg://synthetic:synthetic@unselected.invalid/cache",
    "TAPES_INGEST_URL": "https://unselected.invalid/ingest",
    "LLM_MAX_COMPLETION_TOKENS": "999",
    "APP_MARKER": "unselected",
}
for stage in ("EXTRACTION", "SUMMARIZATION", "QUERY"):
    for field in ("PROVIDER", "MODEL", "ENDPOINT", "API_KEY", "API_VERSION"):
        hostile[f"LLM_{stage}_{field}"] = "unselected"
env_file = root.parent / "external.env" if scenario == "explicit" else project / ".env"
env_bytes = "".join(f"{key}='{value}'\n" for key, value in hostile.items()).encode()
env_file.write_bytes(env_bytes)
app_env = root.parent / "application.env"
app_env.write_text("LLM_MAX_COMPLETION_TOKENS=321\nAPP_MARKER=caller\n", encoding="utf-8")
load_dotenv(app_env)

preloaded_litellm = None
if scenario == "explicit":
    os.environ["LITELLM_MODE"] = "PRODUCTION"
    import litellm

    preloaded_litellm = litellm
    os.environ["LITELLM_MODE"] = ""
    os.environ["COGNEE_ENV_FILE"] = str(env_file)
elif scenario == "hot_reload":
    # Interactive dotenv discovery starts at CWD, as in a notebook or REPL.
    sys.ps1 = "probe> "
    os.environ.update(COGNEE_ENV_FILE="", LITELLM_MODE="DEV", LITELLM_DEV_ENV_HOT_RELOAD="True")
selectors = {key: os.environ.get(key) for key in ("COGNEE_ENV_FILE", "LITELLM_MODE")}
settings = CogneeSettings(
    Endpoint("http://localhost:8080"),
    Endpoint("http://localhost:8081/v1/"),
    "Qwen/test",
    "BAAI/test",
    3,
    root,
    auto_feedback=True,
)

# Files are ignored, but forbidden routes explicitly loaded by the caller still reject.
conflicts = ("LLM_QUERY_ENDPOINT", "FALLBACK_API_KEY", "LLM_API_VERSION", "LLM_ARGS")
os.environ.update(dict.fromkeys(conflicts, "synthetic-private-value"))
with pytest.raises(ValueError, match="Unset conflicting Cognee inference routes") as raised:
    configure(settings)
assert all(key in str(raised.value) for key in conflicts)
assert "synthetic-private-value" not in str(raised.value)
assert "cognee" not in sys.modules
assert not root.exists()
for key in conflicts:
    del os.environ[key]
configure(settings)

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine
from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.relational.config import get_relational_config
from cognee.infrastructure.databases.vector.config import get_vectordb_config
from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
from cognee.infrastructure.llm.config import get_llm_config
from litellm import supports_response_schema

llm = get_llm_config()
embedding = get_embedding_config()
cache = get_cache_config()
assert llm.llm_provider == "custom"
assert llm.llm_model == "openai/Qwen/test"
assert llm.llm_endpoint == "http://localhost:8080/v1"
assert llm.llm_api_key == "sie-local-no-key"
assert llm.structured_output_framework == "litellm_native"
assert not llm.llm_args
for key in (*conflicts[:-1], "FALLBACK_MODEL", "FALLBACK_ENDPOINT"):
    assert not getattr(llm, key.lower())
for stage in ("extraction", "summarization", "query"):
    for field in ("provider", "model", "endpoint", "api_key", "api_version"):
        assert not getattr(llm, f"llm_{stage}_{field}")
assert llm.llm_max_completion_tokens == 321
assert os.environ["LLM_MAX_COMPLETION_TOKENS"] == "321"
assert os.environ["APP_MARKER"] == "caller"
assert embedding.embedding_provider == "openai_compatible"
assert embedding.embedding_model == "BAAI/test"
assert embedding.embedding_endpoint == "http://localhost:8081/v1"
assert embedding.embedding_dimensions == 3
assert embedding.embedding_api_key == "sie-local-no-key"
assert cache.auto_feedback
assert cache.caching
assert cache.session_search_mode == "sequential"
assert cache.cache_backend == "sqlite"
cache_url = f"sqlite+aiosqlite:///{root / 'system' / 'cache.db'}"
assert cache.cache_db_url == cache_url
base = get_base_config()
assert Path(base.system_root_directory) == root / "system"
assert Path(base.data_root_directory) == root / "data"
assert Path(base.logs_root_directory) == root / "logs"
handlers = [handler for handler in logging.getLogger().handlers if isinstance(handler, logging.FileHandler)]
assert handlers
assert all(Path(handler.baseFilename).is_relative_to(root / "logs") for handler in handlers)
assert not unselected_log.exists()
relational = get_relational_config()
assert relational.db_provider == "sqlite"
assert relational.db_name == "cognee_db"
assert Path(relational.db_path) == root / "system" / "databases"
graph = get_graph_config()
assert graph.graph_database_provider == graph.graph_dataset_database_handler == "ladybug"
assert Path(graph.graph_file_path) == root / "system" / "databases" / "cognee_graph_ladybug"
vector = get_vectordb_config()
assert vector.vector_db_provider == vector.vector_dataset_database_handler == "lancedb"
assert Path(vector.vector_db_url) == root / "system" / "databases" / "cognee.lancedb"
assert os.environ["GRAPH_EXTRACTOR"] == "llm"
assert os.environ["MOCK_EMBEDDING"] == "false"
assert supports_response_schema("openai/Qwen/test")
assert {key: os.environ.get(key) for key in selectors} == selectors
assert env_file.read_bytes() == env_bytes
assert not outside.exists()
assert not list(Path(os.environ["TMPDIR"]).glob("sie-cognee-*"))
if preloaded_litellm is not None:
    assert sys.modules["litellm"] is preloaded_litellm


async def check_cache():
    adapter = get_cache_engine()
    assert isinstance(adapter, SqlCacheAdapter)
    assert adapter.db_uri == cache_url
    assert str(adapter.engine.url) == cache_url
    try:
        await adapter.set_value("isolation-probe", "local-value")
        assert await adapter.get_value("isolation-probe") == "local-value"
        assert get_cache_config() is cache
        assert get_cache_engine() is adapter
        assert (root / "system" / "cache.db").is_file()
    finally:
        await adapter.close()


asyncio.run(check_cache())
with pytest.raises(RuntimeError, match="fresh process"):
    configure(settings)
faulthandler.cancel_dump_traceback_later()
assert not attempts, attempts  # A swallowed network exception must also fail the probe.
