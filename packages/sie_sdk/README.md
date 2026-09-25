# SIE SDK

Python client SDK for the SIE inference server.

## Installation

```bash
pip install sie-sdk
```

## Quick Start

```python
from sie_sdk import SIEClient
from sie_sdk.types import Item

client = SIEClient("http://localhost:8080")

# Encode text. Results are TypedDicts — access fields by key.
result = client.encode("BAAI/bge-m3", Item(text="Hello world"))
print(result["dense"].shape)  # (1024,)

# Score items against a query with a reranker model.
# Scores come back sorted by relevance (rank 0 = most relevant).
scores = client.score(
    "BAAI/bge-reranker-v2-m3",
    query=Item(text="What is machine learning?"),
    items=[
        Item(id="doc-1", text="Machine learning is a subfield of AI."),
        Item(id="doc-2", text="Python is a programming language."),
    ],
)
for entry in scores["scores"]:
    print(entry["item_id"], entry["score"])
```

## Entities and relations

GLiNER models extract entity spans for the labels you pass. Joint
entity-relation models such as `knowledgator/gliner-relex-large-v1.0` also
return relations between those entities when you name relation types in
`options`:

```python
result = client.extract(
    "knowledgator/gliner-relex-large-v1.0",
    Item(text="Steve Jobs founded Apple in Cupertino."),
    labels=["person", "organization", "location"],
    options={"relation_labels": ["founded", "located in"], "relation_threshold": 0.7},
)
for relation in result["relations"]:
    print(relation["head"], relation["relation"], relation["tail"], relation["score"])
```

Without `relation_labels` these models return entities only. Relations are
scored among at most 100 entity candidates per item, taken in document order,
so a very long document may not get relations for its later entities; its
entities are still all returned. `threshold` and `relation_threshold` must be
between 0 and 1, and these models need a `threshold` of at least 0.1.

## Zero-shot classification

GLiClass models (`knowledgator/gliclass-*` and the `knowledgator/opir-*`
guardrail models) score text against labels passed with each request. Every
label comes back in `classifications`, sorted by score. By default the scores
form one distribution that sums to 1; `options={"classification_type":
"multi-label"}` scores each label independently instead.

```python
ticket = Item(text="I was charged twice and support has not replied in three days.")

result = client.extract(
    "knowledgator/gliclass-instruct-large-v1.0",
    ticket,
    labels=["billing", "bug report", "feature request"],
    instruction="Classify the support ticket by its main topic.",
)
print(result["classifications"][0]["label"])  # billing
```

`instruction` is the model's task prompt; the `gliclass-instruct-*` and
`opir-*` models are trained to follow one. Few-shot examples go in
`options={"examples": [{"text": ..., "labels": [...]}]}`, with labels taken
from the request's label set. They can move scores a long way, so check them
on your own data. The model reads examples after the document, so a long
document can push them out of the model's window; send
`options={"overflow_policy": "truncate_text"}` to shorten the document instead.

To answer several questions in one call, pass named `label_groups` instead of
`labels`. `data` holds one answer per group:

```python
result = client.extract(
    "knowledgator/gliclass-instruct-large-v1.0",
    ticket,
    instruction="Triage the support ticket.",
    options={
        "label_groups": {
            "topic": ["billing", "bug report", "feature request"],
            "urgency": ["low", "medium", "high"],
            "needs_human": ["yes", "no"],
        }
    },
)
urgency = result["data"]["urgency"]
print(urgency["choice"], urgency["confidence"])  # e.g. medium 0.78
print(urgency["probabilities"])  # {"low": ..., "medium": ..., "high": ...}
```

By default the model reads each group as its own row: the document with only
that group's labels, plus the instruction and examples. A group then scores as
a request whose `labels` are that group's labels would. The rows of a call
share forward passes, so on a GPU, fp16 rounding can move a probability by a
few thousandths against the one-group request. Batching several items into one
call does the same. A call takes at most 64 groups, or 32 on models with a
1,024-token window. Every group reads the part of the document that fits in
the model's window again, so that part may span at most 64 characters per token
of the window: 32,768 characters on a 512-token model, 65,536 on a 1,024-token
one. With few groups the bound is 524,288 characters divided by the number of
groups, when that is larger. Prose fills a 512-token window in about 2,300
characters, and text laid out with long runs of spaces in about 20,000. Text
written without spaces between words is read whole, so the bound applies to
the whole document. A document over the bound comes back with an
`INPUT_TOO_LONG` error, and the other items still succeed.

`options={"group_encoding": "joint"}` reads all the groups in one row per item
instead: the document next to every group's labels, written as `group.label`.
That row costs less, but each group's scores then depend on the other groups'
labels. Single-label scores are still normalized within each group.

Each group answers `{"type": "choice", "choice", "probabilities",
"confidence"}`, where `confidence` is `1 - entropy / log(number of labels)`.
With `"classification_type": "multi-label"` a group answers `{"labels",
"probabilities"}`: every label scored independently, and `labels` lists those
at or above `options.threshold` (0.5 when no threshold is set).
`classifications` lists the same scores under `group.label` names. In a
grouped request, an example's labels may be written as `"urgency.high"` or as
`{"urgency": "high"}`. With separate rows, each row reads every example with
only that group's labels.

Usage counts the tokens of every row the model encodes: the document plus the
instruction and example texts sent with it. Label names are not counted. A
labels request or a joint call encodes one row per item. A call with separate
groups encodes one row per item and group, so it counts the document once per
group: three groups cost what three one-group requests cost. With an
instruction or examples, each row's count is capped at the model window minus
that row's label prompt, unless the document count alone is already higher.
An item whose document pushes the labels out of the window, in any of its rows,
comes back with an `INPUT_TOO_LONG` error in its `error` field and is not
billed. The other items still succeed.

On a CUDA server, the operator can load the DeBERTa-based GLiClass models with
CUDA graphs, which cut the CPU time spent launching kernels. With `bucketed`
graphs, sequence lengths are padded to buckets, which moves probabilities
slightly, as batching requests together does. `gliclass-base-v1.0`,
`gliclass-large-v1.0` and `opir-multitask-large-v1.0` load with `bucketed`
graphs by default: their probabilities differed from eager execution by up to
0.0144 in our tests, with no top label changed. Send
`options={"cuda_graphs": "off"}` to run a request eagerly; a request cannot
turn graphs on. See the server README for each model's measurements and the
memory graphs use.

## Typed decisions

Typed-decision models answer typed questions about each item and return a
probability for every option. The Laya models and the GLiNER2.5-Decide models
take the same question mapping as `output_schema`: `choice` (pick one of the
criteria), `noul` (yes or no), and `score` (an ordinal scale, index 0 first).
Answers come back in `data`, keyed by question id.

```python
result = client.extract(
    "fastino/GLiNER2.5-Decide",
    Item(text="This is the third time I have explained the same missing refund. Get me a person."),
    output_schema={
        "intent": {
            "type": "choice",
            "instructions": "What does the customer want?",
            "criteria": {"refund_request": "money back", "cancel": None, "complaint": "unhappy with service"},
        },
        "handoff": {"type": "noul", "instructions": "Should a person take over?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["calm", "annoyed", "angry"]},
    },
)
answers = result["data"]
print(answers["intent"]["choice"], answers["intent"]["probabilities"])
# e.g. refund_request {'refund_request': 0.59, 'cancel': 0.07, 'complaint': 0.34}
print(answers["handoff"]["answer"], answers["handoff"]["noul"])  # True 0.99 (the probability of yes)
print(answers["frustration"]["score"])  # e.g. 1.13, the expected level from 0 to 2
```

`choice` and `score` answers carry `probabilities` and `confidence`
(`1 - entropy / log(number of options)`); a `noul` answer carries `noul`, the
probability of yes, `answer`, and `confidence` (`max(p, 1 - p)`). The
GLiNER2.5-Decide models also accept `options={"label_groups": {...}}` with
`"classification_type": "multi-label"` for independent per-label scores, and
plain `labels`, which return every label in `classifications`. They read all
of a call's questions in one row per item, so each question's probabilities
depend on the other questions sent with it.

## Generation prompts and guard verdicts

`generate` and `stream_generate` treat text-only prompts as raw continuation
input. They do not render a chat template, including model settings such as
`enable_thinking` or `guardian_config`. Already-rendered prompts stay unchanged.
Native requests with images render the prompt and images as one user turn.

Use `chat_completions` or `stream_chat_completions` with messages for chat,
instruction-based structured output (`response_format`), and guard checks:

```python
answer = client.chat_completions(
    "Qwen/Qwen3-4B-Instruct-2507",
    [{"role": "user", "content": "Write a haiku about the sea."}],
    max_completion_tokens=64,
)
print(answer["choices"][0]["message"]["content"])
```

The worker renders the selected model's template and applies served template
settings with operator configuration taking precedence over request kwargs.
Granite Guardian's shipped risk dimension is `harm`; prose requesting a
different dimension does not change that setting. Its configured threshold
produces `Yes` (unsafe) or `No` (safe). A missing or invalid verdict returns
`invalid_guard_verdict`; never treat an error or an empty response as safe.
Private reasoning is hidden on both input surfaces. If it consumes the entire
generation budget without usable output, the request fails with
`empty_model_output`.

## Connecting to a managed SIE platform

The examples above target a local server. For a managed SIE gateway,
pass the gateway URL as `base_url` and your API key (sent as a Bearer
token):

```python
from sie_sdk import SIEClient

client = SIEClient(
    "https://your-gateway.example.com",
    api_key="YOUR_API_KEY",
)
```

## Generation execution evidence

`SIEClient.last_model_revision` retains the `X-SIE-Model-Revision` response
header from the latest call in the current thread. On buffered gateway
responses, this is the lowercase 64-hex executed bundle/config SHA-256 when
worker evidence matches the routing snapshot. It is distinct from a catalog
weights revision such as a 40-hex Hugging Face commit.

Gateway SSE responses omit that header: headers are sent before terminal
execution evidence is available. Fully consuming `stream_generate()` leaves
`last_model_revision` as `None`. A successful terminal `GenerateChunk` may
instead carry `execution_identity_sha256` and `execution_binding_sha256` as
an optional complete pair of lowercase 64-hex SHA-256 digests. Both Python
clients preserve those fields. Older or self-hosted deployments may omit
both; absence is compatible, but cannot prove which deployment executed.
The terminal digests are distinct from the weights revision and config hash.

## Object storage and model caches

Install the `storage` extra to use `s3://`, `gs://`, `abfs(s)://`, or native
Alibaba `oss://` model/cache paths:

```bash
pip install 'sie-sdk[storage]'
```

Alibaba OSS always uses region-scoped Signature V4. Set `SIE_OSS_REGION` to
the bucket's region (for example, `eu-central-1`). Set
`SIE_OSS_USE_INTERNAL_ENDPOINT=true` only inside the matching Alibaba Cloud
network; the SDK derives the HTTPS endpoint and does not accept endpoint URLs
from storage paths.

On ACK, RRSA supplies `ALIBABA_CLOUD_ROLE_ARN`,
`ALIBABA_CLOUD_OIDC_PROVIDER_ARN`, and `ALIBABA_CLOUD_OIDC_TOKEN_FILE`. The SDK
requires all three together and refreshes short-lived credentials through the
Alibaba Credentials client. Local operators can use that client's standard
credential sources instead. Never place credentials, queries, or fragments in
an `oss://` URL.

`oss://` is supported for model discovery, cache population, and object copies.
It is deliberately not a mutable `sie-config` epoch store because OSS
PutObject cannot provide the required non-empty ETag compare-and-swap contract;
use local/PVC, S3, GCS, or Azure storage for that store.

## Error handling

Server-reported errors are raised as typed exceptions from
`sie_sdk.client.errors`; all inherit from `SIEError`, and errors that
carry a server response expose `.code` and `.status_code`. Invalid
client-side arguments (for example, a bad `base_url_headers` value)
raise ordinary Python exceptions such as `ValueError`.

Several `503` codes are transient and retried automatically (see the
next section for the `RESOURCE_EXHAUSTED` budget):

- `PROVISIONING` — the cluster is scaling capacity from zero. Retry is
  governed by `wait_for_capacity`: retried under `provision_timeout_s`
  when `True` (the default); surfaces immediately as
  `ProvisioningError` when `False`.
- `MODEL_LOADING` — the worker accepted the request and is cold-loading
  the target model. Retried until `provision_timeout_s`; raises
  `ModelLoadingError` if the budget is exhausted.
- `LORA_LOADING` — the requested LoRA adapter is still loading. Retried
  a bounded number of times; raises `LoraLoadingError` when the retry
  budget is exhausted.
- `RESOURCE_EXHAUSTED` — the server ran out of GPU memory and exhausted
  its internal recovery. Retried with bounded backoff; raises
  `ResourceExhaustedError` when retries run out. Pass
  `max_oom_retries=0` to disable these retries and fail fast.

One generation-specific error is terminal and never retried:

- `empty_model_output` — the generation finished nominally but produced
  no visible output text (for example, private reasoning consumed the
  whole token budget). Tokens were genuinely consumed, so the request
  is not re-run. Surfaces as `ServerError` with
  `code == "empty_model_output"`; on streaming calls it is raised
  mid-stream with the gateway request id attached for correlation.

## Handling resource exhaustion

The SDK automatically retries requests that the server signals as
transient — model still loading, scale-from-zero in progress, or **GPU
memory pressure (`RESOURCE_EXHAUSTED`)**. You don't have to write
retry logic for these.

### What happens by default

When the server's GPU runs out of memory mid-request, the worker first
attempts an internal recovery (clear cache → evict an idle sibling
model → recursively halve the batch). If that succeeds you get a normal
200 response — slightly slower than usual.

If recovery is exhausted, the server returns `503 RESOURCE_EXHAUSTED`
with a `Retry-After: 5` header. The SDK then retries with bounded
exponential backoff (5s → 10s → 20s, capped at 30s, max 3 attempts).
The first retry logs at WARNING so you can see it at default log
levels:

```text
WARNING sie_sdk.client.sync: Server resource exhausted, retrying in 5.0s (attempt 1/3, elapsed: 0.4s, timeout: 900.0s)
```

If all retries are exhausted, the SDK raises
`sie_sdk.client.errors.ResourceExhaustedError` (a subclass of
`ServerError`).

### Tuning the behaviour

| Parameter | Default | Effect |
|--|--|--|
| `max_oom_retries=N` | `3` | Cap on auto-retries. Pass `0` to fail fast. |
| `provision_timeout_s=T` | `900` (15 min) | Total wall-clock budget. OOM retries are clamped to the remaining budget — you'll never sleep past your timeout. |

### Examples

**Default (resilient) — recommended for most callers:**

```python
result = client.encode("BAAI/bge-m3", Item(text="Hello"))
# Auto-retries on RESOURCE_EXHAUSTED. May take up to ~35s extra
# if recovery + retries are needed.
```

**Fail-fast (CI tests, latency-critical hot paths):**

```python
from sie_sdk.client.errors import ResourceExhaustedError

try:
    result = client.encode(
        "BAAI/bge-m3",
        Item(text="Hello"),
        max_oom_retries=0,  # No retries; surface failure immediately
    )
except ResourceExhaustedError:
    # Server is under memory pressure — fall back to a smaller model,
    # batch later, or surface to the user.
    ...
```

**Tight wall-clock budget:**

```python
result = client.encode(
    "BAAI/bge-m3",
    Item(text="Hello"),
    provision_timeout_s=10.0,  # Total budget; OOM retries clamped to it
)
```

### What you'll see in your logs

| Server state | Client outcome | Log level |
|--|--|--|
| GPU OK | 200, normal latency | (none) |
| OOM, server-side recovery succeeds | 200, +1-3s latency | (none) |
| OOM, SDK retries succeed | 200, +5-35s latency | WARNING on 1st retry |
| OOM, SDK retries exhausted | `ResourceExhaustedError` | WARNING + traceback |

If you see frequent `Server resource exhausted, retrying...` warnings,
your cluster's GPU pool is undersized for the workload. Talk to the
operator running SIE — they have observability and tuning knobs
(`SIE_OOM_RECOVERY__*`) that aren't visible from the SDK side.

## License

Apache 2.0
