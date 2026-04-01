# sie-crewai

SIE integration for CrewAI.

## Installation

```bash
pip install sie-crewai
```

## Usage

### Embeddings (via OpenAI-compatible API)

SIE provides an OpenAI-compatible `/v1/embeddings` endpoint, so you can use it directly with CrewAI's OpenAI provider:

```python
from crewai import Crew, Agent, Task

crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,
    embedder={
        "provider": "openai",
        "config": {
            "model": "BAAI/bge-m3",
            "api_base": "http://localhost:8080/v1",
            "api_key": "not-needed",  # SIE doesn't require API key
        }
    }
)
```

### Reranker Tool

Use `SIERerankerTool` to rerank documents in agent workflows:

```python
from sie_crewai import SIERerankerTool

reranker = SIERerankerTool(
    base_url="http://localhost:8080",
    model="jinaai/jina-reranker-v2-base-multilingual",
)

agent = Agent(
    role="Research Analyst",
    tools=[reranker],
    ...
)
```

### Extractor Tool

Use `SIEExtractorTool` to extract entities from text:

```python
from sie_crewai import SIEExtractorTool

extractor = SIEExtractorTool(
    base_url="http://localhost:8080",
    model="urchade/gliner_multi-v2.1",
    labels=["person", "organization", "location"],
)

agent = Agent(
    role="Information Extractor",
    tools=[extractor],
    ...
)
```
