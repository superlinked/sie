<div align="center">

<picture>
  <source srcset="https://cdn.prod.website-files.com/65dce6831bf9f730421e2915/66ef0317ed8616151ee1d451_superlinked_logo_white.png"
          media="(prefers-color-scheme: dark)">
  <img width="320"
       src="https://cdn.prod.website-files.com/65dce6831bf9f730421e2915/65dce6831bf9f730421e2929_superlinked_logo.svg"
       alt="Superlinked logo">
</picture>

<h1>SIE: Superlinked Inference Engine</h1>

<p><strong>Self-hosted inference for agents. Every open model your agents call, served from one cluster in your cloud.</strong></p>

<p>
  <a href="https://superlinked.com/docs/">Docs</a> |
  <a href="https://superlinked.com/docs/quickstart/">Quickstart</a> |
  <a href="https://superlinked.com/docs/reference/api/">API Reference</a> |
  <a href="https://superlinked.com/models">Models</a>
</p>

[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/sie-sdk?style=flat-square)](https://pypi.org/project/sie-sdk/)
[![GitHub stars](https://img.shields.io/github/stars/superlinked/sie?style=flat-square)](https://github.com/superlinked/sie/stargazers)

⭐ _Help us reach more developers and grow the SIE community. Star this repo!_

</div>

## About

SIE is an open-source inference engine that runs the models behind every agent task through one API: search and retrieval, document-to-markdown conversion, structured output, content safety, and the agent loop itself. It replaces the patchwork of a separate model server per task with one system that serves 100+ models, loading each on demand.

- OpenAI-compatible API for drop-in migration: `/v1/embeddings`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`
- Pre-configured model catalog: Stella, SPLADE, Qwen3, GLiNER, SigLIP, and more; embedding and retrieval models benchmarked on MTEB
- Serves multiple models simultaneously with on-demand loading and LRU eviction
- Ships the full production stack: load-balancing gateway, KEDA autoscaling, Grafana dashboards, Terraform for GKE, EKS, and AKS
- Integrates with LangChain, LlamaIndex, Haystack, DSPy, CrewAI, Chroma, Qdrant, Weaviate, and LanceDB

## Tasks

One SIE cluster runs the inference behind a whole agent. Each task is a handful of swappable models; browse [`packages/sie_server/models/`](https://github.com/superlinked/sie/tree/main/packages/sie_server/models) for the full set.

| Task | What it does | Models |
|---|---|---|
| **Search** | Embed, match, and rerank to retrieve the right context. | `bge-m3`, `splade-v3`, `colbertv2`, `qwen3-reranker` |
| **Document to markdown** | PDFs, Office files, and scans become clean markdown. | `lightonocr`, `glm-ocr`, `mineru`, `paddleocr-vl`, `docling` |
| **Structured output** | Schema-valid JSON, extracted or generated. | `gliner2`, `nuner-zero`, `qwen3.6-27b` |
| **Guard content** | A safety verdict with a probability you threshold. | `granite-guardian-2b` |
| **Run the agent loop** | Plan steps and call tools with an open LLM, streaming included. | `qwen3.6-27b` |

## Quickstart

Prefer a notebook? [`examples/quickstart.ipynb`](examples/quickstart.ipynb) runs this same flow, on your machine or a free Colab GPU.

**1. Start the server**

```bash
# macOS (Apple Silicon) or Linux, native (requires Python 3.12)
pip install "sie-server[local]" && sie-server serve

# Linux, NVIDIA GPU
docker run --gpus all -p 8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cuda12-default

# Linux, NVIDIA GPU — Transformers 5 OCR models (LightOnOCR and GLM-OCR)
docker run --gpus all -p 8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cuda12-transformers5

# Linux, CPU
docker run -p 8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cpu-default
```

Docker images are bundle-specific so dependency-incompatible model families stay isolated. Use the
`transformers5` image for LightOnOCR or GLM-OCR; the `default` image intentionally does not advertise them.

```bash
# in a second terminal
curl http://localhost:8080/readyz   # expect: ok
```

The server speaks the OpenAI API out of the box, embeddings and generation alike (the cluster gateway serves `/v1/chat/completions`, `/v1/completions`, and `/v1/responses`). Your first call needs nothing but curl:

```bash
curl http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "sentence-transformers/all-MiniLM-L6-v2", "input": "Hello world"}'
# {"object": "list", "data": [{"object": "embedding", "embedding": [-0.0344, 0.0310, ...
```

Each model's first call downloads its weights (progress appears in the server terminal). Later calls skip the
download; inference latency depends on the model, task, hardware, and batch size.

**2. Install the SDK**

```bash
pip install sie-sdk                # Python
npm install @superlinked/sie-sdk   # TypeScript (pnpm and yarn work too)
```

**3. Generate embeddings, rerank, and extract entities**

```python
from sie_sdk import SIEClient
from sie_sdk.types import Item

client = SIEClient("http://localhost:8080")

# Generate embeddings
result = client.encode("sentence-transformers/all-MiniLM-L6-v2", Item(text="Hello world"))
print(result["dense"].shape)  # (384,)

# Rerank search results
scores = client.score(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    Item(text="What is machine learning?"),
    [Item(text="ML learns from data."), Item(text="The weather is sunny.")],
)
print(scores["scores"][0])  # {'item_id': 'item-0', 'score': -7.1, 'rank': 0}

# Extract entities
result = client.extract(
    "urchade/gliner_multi-v2.1",
    Item(text="Tim Cook is the CEO of Apple."),
    labels=["person", "organization"],
)
print(result["entities"][0])
# {'text': 'Tim Cook', 'label': 'person', 'score': 0.992, 'start': 0, 'end': 8, ...}
```

Text generation runs on the GPU generation image; stop the first server, then start this one on the same port:

```bash
# Linux, NVIDIA GPU (for generation on Apple Silicon via MLX, see the docs below)
docker run --gpus all -p 8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cuda12-sglang
```

```python
result = client.generate(
    "Qwen/Qwen3-0.6B",
    "Reply with a single word: the capital of France.",
    max_new_tokens=16,
    temperature=0.0,
)
print(result["text"])  # Paris
```

For generation on Apple Silicon (MLX), the TypeScript walkthrough, and every configuration in between, see the [quickstart guide](https://superlinked.com/docs/quickstart/), [TypeScript SDK docs](https://superlinked.com/docs/reference/typescript-sdk/), and [SDK reference](https://superlinked.com/docs/reference/sdk/).

---

### Production

The same code works against a production cluster. SIE ships a load-balancing gateway, KEDA autoscaling (scale to zero), Grafana dashboards, and Terraform modules for [GKE](https://github.com/superlinked/terraform-google-sie), [EKS](https://github.com/superlinked/terraform-aws-sie), and [AKS](https://github.com/superlinked/terraform-azure-sie). Not just the server, the whole stack. All Apache 2.0.

```bash
# pick one values overlay: values-gke.yaml / values-aws.yaml / values-aks.yaml
# (pin a chart version for reproducible installs, e.g. --version 0.6.18)
helm upgrade --install sie-cluster oci://ghcr.io/superlinked/charts/sie-cluster \
  --namespace sie --create-namespace \
  --set hfToken.create=true \
  --set hfToken.value=YOUR_HF_TOKEN \
  -f https://raw.githubusercontent.com/superlinked/sie/main/deploy/helm/sie-cluster/values-gke.yaml
```

See the [deployment guide](https://superlinked.com/docs/deployment/).

> **Telemetry**: SIE collects anonymous usage data (version, OS, architecture, GPU type) to understand adoption. No IP addresses, hostnames, or request data are collected. Disable with `SIE_TELEMETRY_DISABLED=1` or `DO_NOT_TRACK=1`.

---

### Explore

[**Model catalog**](https://superlinked.com/models): every model is a config in [`packages/sie_server/models/`](https://github.com/superlinked/sie/tree/main/packages/sie_server/models); pass its Hugging Face ID to the SDK.

[**Integrations**](https://superlinked.com/docs/integrations/): setup guides for all nine framework and vector-store integrations, in Python and TypeScript.

[**Examples**](examples/): A quickstart notebook and an end-to-end project gallery.

[**MCP edge**](packages/sie_mcp/): offload document work from Claude and other MCP clients to your cluster and save agent tokens.

[**Why we built SIE**](https://www.youtube.com/watch?v=qdh_x-uRs9g): The motivation, told at AI Engineer Europe 2026.

---

<p align="center">
  <a href="https://superlinked.com/docs"><strong>superlinked.com/docs</strong></a> | Apache 2.0
</p>
