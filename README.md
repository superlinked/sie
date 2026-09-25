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

SIE is an open-source inference engine that runs the models behind every agent task through one API: search and retrieval, document-to-markdown conversion, structured output, content safety, the agent loop itself, and the translation, transcription, and vision calls around it. It replaces the patchwork of a separate model server per task with one system that serves 100+ models, loading each on demand.

- OpenAI-compatible API for drop-in migration: `/v1/embeddings`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`
- Pre-configured model catalog: Stella, SPLADE, Qwen3, GLiNER, SigLIP, and more; embedding and retrieval models benchmarked on MTEB
- Serves multiple models simultaneously with on-demand loading and LRU eviction
- Ships Kubernetes and Helm deployment configs for the load-balancing gateway, KEDA autoscaling, and Grafana dashboards
- Integrates with LangChain, LlamaIndex, Haystack, DSPy, CrewAI, Chroma, Qdrant, Weaviate, and LanceDB

## Tasks

One SIE cluster runs the inference behind a whole agent. Each task is a handful of swappable models; every name below links to its config, whose `sie_id` is the model name you pass to the SDK (`hf_id` is the Hugging Face repository the weights load from; the two usually match). Browse [`packages/sie_server/models/`](https://github.com/superlinked/sie/tree/main/packages/sie_server/models) for the full set.

| Task | What it does | Models |
|---|---|---|
| **Search** | Embed, match, and rerank to retrieve the right context. | [`bge-m3`](packages/sie_server/models/BAAI__bge-m3.yaml), [`splade-v3`](packages/sie_server/models/naver__splade-v3.yaml), [`colbertv2`](packages/sie_server/models/colbert-ir__colbertv2.0.yaml), [`qwen3-reranker`](packages/sie_server/models/Qwen__Qwen3-Reranker-4B.yaml) |
| **Document to markdown** | PDFs, Office files, and scans become clean markdown. | [`lightonocr`](packages/sie_server/models/lightonai__LightOnOCR-2-1B.yaml), [`glm-ocr`](packages/sie_server/models/zai-org__GLM-OCR.yaml), [`mineru`](packages/sie_server/models/opendatalab__MinerU2.5-Pro-2604-1.2B.yaml), [`paddleocr-vl`](packages/sie_server/models/PaddlePaddle__PaddleOCR-VL-1.5.yaml), [`docling`](packages/sie_server/models/docling.yaml) |
| **Structured output** | Schema-valid JSON, extracted or generated. | [`gliner2`](packages/sie_server/models/fastino__gliner2-large-v1.yaml), [`gliner-relex`](packages/sie_server/models/knowledgator__gliner-relex-large-v1.0.yaml), [`gliformer`](packages/sie_server/models/knowledgator__gliformer-large-v1.yaml), [`nuner-zero`](packages/sie_server/models/numind__NuNER_Zero.yaml), [`qwen3.8-27b`](packages/sie_server/models/Qwen__Qwen3.8-27B-FP8.yaml), [`qwen3.6-27b`](packages/sie_server/models/Qwen__Qwen3.6-27B.yaml) |
| **Decide** | Choice, yes/no, and score answers with probabilities to typed questions about a text or JSON state. | [`laya`](packages/sie_server/models/convaiinnovations__laya.yaml), [`laya-multilingual`](packages/sie_server/models/convaiinnovations__laya-multilingual.yaml), [`laya-typed-decisions`](packages/sie_server/models/convaiinnovations__laya-typed-decisions.yaml), [`gliner2.5-decide`](packages/sie_server/models/fastino__GLiNER2.5-Decide.yaml), [`gliner2.5-multi-decide`](packages/sie_server/models/fastino__GLiNER2.5-multi-Decide.yaml), [`gliner2.5-decide-1b`](packages/sie_server/models/fastino__GLiNER2.5-Decide-1B.yaml) |
| **Classify** | Zero-shot labels, with several label groups answered in one call. The instruct models also follow a task instruction and few-shot examples. | [`gliclass-large-v3`](packages/sie_server/models/knowledgator__gliclass-large-v3.0.yaml), [`gliclass-instruct-large`](packages/sie_server/models/knowledgator__gliclass-instruct-large-v1.0.yaml), [`gliclass-multilang-mini`](packages/sie_server/models/knowledgator__gliclass-multilang-mini.yaml) |
| **Guard content** | A safety verdict: Yes/No with the threshold set in the model config, or safe/unsafe and policy-label scores with the threshold chosen per request. | [`granite-guardian-2b`](packages/sie_server/models/ibm-granite__granite-guardian-3.0-2b.yaml), [`opir-multitask-large`](packages/sie_server/models/knowledgator__opir-multitask-large-v1.0.yaml), [`opir-edge`](packages/sie_server/models/knowledgator__opir-edge-v1.0.yaml) |
| **Run the agent loop** | Plan steps and call tools with an open LLM, streaming included. | [`qwen3.8-27b`](packages/sie_server/models/Qwen__Qwen3.8-27B-FP8.yaml), [`qwen3.6-27b`](packages/sie_server/models/Qwen__Qwen3.6-27B.yaml) |
| **Translate** | Text between 400+ languages. | [`madlad400-3b-mt`](packages/sie_server/models/google__madlad400-3b-mt.yaml) |
| **See images** | Caption, detect objects, and answer questions about images. | [`florence-2`](packages/sie_server/models/microsoft__Florence-2-large.yaml), [`owlv2`](packages/sie_server/models/google__owlv2-base-patch16-ensemble.yaml), [`grounding-dino`](packages/sie_server/models/IDEA-Research__grounding-dino-base.yaml) |
| **Transcribe audio** | Speech to text. | [`whisper-large-v3-turbo`](packages/sie_server/models/openai__whisper-large-v3-turbo.yaml) |

## Quickstart

The quickstart uses the smallest model in each family so the first run is fast on a laptop. Swap in any model from the table above for production quality.

**1. Start the server**

```bash
# macOS (Apple Silicon) or Linux, native (requires Python 3.12)
pip install "sie-server[local]" && sie-server serve

# Linux, NVIDIA GPU
docker run --gpus all -p 8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cuda12-default

# Linux, NVIDIA GPU: Transformers 5 models (LightOnOCR, GLM-OCR, and the GLiNER2.5-Decide models)
docker run --gpus all -p 8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cuda12-transformers5

# Linux, NVIDIA GPU — SGLang vision OCR (default profiles of LightOnOCR, GLM-OCR, PaddleOCR-VL)
docker run --gpus all -p 8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cuda12-sglang-vision-extract

# Linux, CPU
docker run -p 8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cpu-default
```

Docker images are bundle-specific so dependency-incompatible model families stay isolated. Use the
`sglang-vision-extract` image for LightOnOCR, GLM-OCR, and PaddleOCR-VL, or the `transformers5` image for the
`:transformers` profiles of LightOnOCR and GLM-OCR and for the GLiNER2.5-Decide models; the `default` image
intentionally does not advertise them.

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

# Entities and a schema-valid record from one forward pass
result = client.extract(
    "knowledgator/gliformer-base-v1",
    Item(text="Refund request from Maria Lopez for order 4471 was denied: the item was returned after 45 days."),
    labels=["person"],
    output_schema={
        "type": "object",
        "properties": {
            "order number": {"type": "string"},
            "reason": {"type": "string"},
            "decision": {"type": "string", "enum": ["approved", "denied", "escalated"]},
        },
    },
)
print(result["data"])
# e.g. {'order number': '4471', 'reason': 'the item was returned after 45 days', 'decision': 'denied'}
```

Typed decisions ask one or more typed questions about each item (a text, or a JSON object or conversation passed as
`metadata={"state": ...}`) and return an answer with probabilities per question in `data`. `laya` and
`laya-typed-decisions` calibrate the probabilities with their shipped temperatures; `laya-multilingual` ships none, so
its probabilities are the raw softmax. `usage.input_tokens` counts every (item, question) row the model encodes: the
state's tokens plus the question's, once per question. An item may use up to 32,768 row tokens (questions × `max_len`),
so the 1024-token `laya-multilingual` and `laya-typed-decisions` take up to 32 questions per request and `laya` up to 64.

```python
result = client.extract(
    "convaiinnovations/laya",
    Item(text="Hi, we were billed twice for March. Please refund the duplicate today."),
    output_schema={
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages"}},
        "refund_requested": {"type": "noul", "instructions": "Does the user ask for a refund?"},
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]},
    },
)
print(result["data"]["department"])  # values are illustrative and rounded
# {'type': 'choice', 'choice': 'billing', 'probabilities': {'billing': 0.987, 'technical': 0.013}, 'confidence': 0.9}
```

The GLiNER2.5-Decide models (`fastino/GLiNER2.5-Decide` for English, `GLiNER2.5-multi-Decide`, and
`GLiNER2.5-Decide-1B`), served by the `transformers5` image from step 1, take the same questions, GLiClass-style `options={"label_groups": {...}}`, or plain `labels`,
and return every option's probability. They read all of a call's questions next to the document in one row per item:
one forward pass answers them all, and each question's probabilities depend on the other questions sent with it. A
`score` question is read as the ordinal labels `"0"` to `"k-1"`, each described by its criterion; a `noul` question as
`"yes"`/`"no"`. `usage.input_tokens` counts the document tokens the model reads plus the questions' instructions and
criteria text; question ids and label names are not counted.

```python
result = client.extract(
    "fastino/GLiNER2.5-Decide",
    Item(text="Guest in room 1408 says the AC has been out since yesterday and wants to move rooms tonight."),
    output_schema={
        "intent": {"type": "choice", "instructions": "What does the guest want?",
                   "criteria": {"room_change": "move to another room", "maintenance": "fix something", "checkout": None}},
        "needs_human": {"type": "noul", "instructions": "Must a person act on this?"},
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "normal", "high", "urgent"]},
    },
)
print(result["data"]["intent"])  # values are illustrative and rounded
# {'type': 'choice', 'choice': 'room_change',
#  'probabilities': {'room_change': 0.871, 'maintenance': 0.085, 'checkout': 0.045}, 'confidence': 0.57}
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

The same code works against a production cluster. SIE ships a load-balancing gateway and Kubernetes deployment surface, including Helm charts, KEDA autoscaling (scale to zero), and Grafana dashboards. Public Terraform modules are maintained separately for [Alibaba Cloud ACK](https://github.com/superlinked/terraform-alicloud-sie), [EKS](https://github.com/superlinked/terraform-aws-sie), [AKS](https://github.com/superlinked/terraform-azure-sie), and [GKE](https://github.com/superlinked/terraform-google-sie). Not just the server, the whole stack. All Apache 2.0.

```bash
# pick one values overlay: values-ack.yaml / values-aws.yaml / values-aks.yaml / values-gke.yaml
# (pin a chart version for reproducible installs, e.g. --version 0.7.3)
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

[**Model catalog**](https://superlinked.com/models): every model is a config in [`packages/sie_server/models/`](https://github.com/superlinked/sie/tree/main/packages/sie_server/models); pass its `sie_id` to the SDK.

[**Integrations**](https://superlinked.com/docs/integrations/): setup guides for all nine framework and vector-store integrations, in Python and TypeScript.

[**Examples**](examples/): An end-to-end project gallery.

[**MCP edge**](packages/sie_mcp/): offload document, image, and structured-output work from Claude and other MCP clients to your cluster and save agent tokens.

[**Why we built SIE**](https://www.youtube.com/watch?v=qdh_x-uRs9g): The motivation, told at AI Engineer Europe 2026.

---

## Development

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the complete development
workflow and [AGENTS.md](AGENTS.md) for repository automation boundaries.

Install [mise](https://mise.jdx.dev/getting-started.html), then bootstrap the
versioned Python, Rust, Node.js, and Helm toolchains from the repository root:

```bash
./tools/init.sh
```

The common development checks and local server are available as mise tasks
(`mise tasks` lists them all):

```bash
mise run test
mise run lint
mise run typecheck
mise run serve
mise run rust-check
mise run rust-test
mise run gateway-test
mise run server-sidecar-test
mise run helm -- dependencies
mise run helm -- lint --set payloadStore.enabled=false
mise run helm -- template --set payloadStore.enabled=false
```

The Python workspace uses the committed root lock. Package membership is
explicit in `pyproject.toml`; a package joins the workspace only in the same
change that adds its complete source. Native audio is always an opt-in build:
install cmake, then run
`mise exec -- uv sync --frozen --project . --all-packages --all-extras`.

---

<p align="center">
  <a href="https://superlinked.com/docs"><strong>superlinked.com/docs</strong></a> | Apache 2.0
</p>
