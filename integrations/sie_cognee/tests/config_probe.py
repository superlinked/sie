"""A fresh-process contract probe; no network, models or global storage."""

import os
import sys
from pathlib import Path

from sie_cognee.config import CogneeSettings, Endpoint, configure

root = Path(sys.argv[1])
configure(
    CogneeSettings(
        Endpoint("http://localhost:8080"),
        Endpoint("http://localhost:8081/v1/"),
        "Qwen/test",
        "BAAI/test",
        3,
        root,
        auto_feedback=True,
    )
)

from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.infrastructure.databases.relational.config import get_relational_config
from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
from cognee.infrastructure.llm.config import get_llm_config

llm = get_llm_config()
embedding = get_embedding_config()
cache = get_cache_config()
assert llm.llm_provider == "custom"
assert llm.llm_model == "openai/Qwen/test"
assert llm.llm_endpoint == "http://localhost:8080/v1"
assert llm.llm_api_key == "sie-local-no-key"
assert not llm.fallback_model
assert embedding.embedding_provider == "openai_compatible"
assert embedding.embedding_model == "BAAI/test"
assert embedding.embedding_endpoint == "http://localhost:8081/v1"
assert embedding.embedding_dimensions == 3
assert embedding.embedding_api_key == "sie-local-no-key"
assert cache.auto_feedback
assert cache.caching
assert cache.cache_backend == "sqlite"
assert str(root) in cache.cache_db_url
assert Path(get_relational_config().db_path).is_relative_to(root)
assert os.environ["GRAPH_EXTRACTOR"] == "llm"
