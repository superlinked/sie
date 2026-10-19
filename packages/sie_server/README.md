# SIE Server

GPU inference server for embeddings, reranking, and entity extraction.

## Features

- Multi-model serving with LRU eviction
- Token-based dynamic batching
- Hot reload model configs without restart
- Unified API: `encode()`, `score()`, `extract()`
- Prometheus metrics and OpenTelemetry tracing
- Reactive GPU OOM recovery + proactive idle eviction

## Remote backends

Serve a model through another SIE deployment or an OpenAI-compatible upstream
using [remote profiles](REMOTE_BACKENDS.md). The guide covers configuration,
caller controls, current routing limits and cluster workers. See
[Moving to self-hosted serving](MIGRATING_TO_SELF_HOSTED.md) to migrate one model
at a time while keeping its application name.

## Optional bounded batch progress

Model profiles can opt into `bounded_batch_progress: true`. The default is
`false`; no shipped profile enables it. An extending profile inherits the
parent's setting when the field is omitted or `null`, and an explicit `false`
disables it for that child.

```yaml
profiles:
  default:
    adapter_path: "my_adapter:Adapter"
    max_batch_tokens: 16384
  progress:
    extends: default
    bounded_batch_progress: true
```

This option applies to the Python worker's batcher for each base model or
LoRA selection. After an ordinary cost-sorted batch skips the oldest batchable
item, the next batchable turn starts at that item and packs its cost-sorted
suffix. It does not wrap around to cheaper items. An item with `k` older
batchable items therefore runs within at most `2 * (k + 1)` nonempty batchable
turns. Existing `runs_alone` turns may interleave, cost/request limits still
apply, and an individually oversized item still runs alone. This is a
dispatch-turn bound, not a wall-clock deadline. Cancellation and worker
shutdown retain their existing behavior. Preformed/sidecar batches bypass
the Python batcher and are unaffected.

Enabled scheduling changes batch composition and can change model scores or
chosen labels, as well as throughput. It does not change model precision or
make outputs invariant to batch shape. Evaluate the intended model's output
and throughput contract before enabling it. The unchanged default uses the
original batcher without progress bookkeeping.

## Installation

```bash
pip install sie-server
```

`sie-server` ships several model bundles, and a single environment can only hold one
`transformers` version — install the one your bundle needs:

- **Default bundle** (embeddings, reranking, extraction) — verified on `transformers` 4.x:

  ```bash
  pip install sie-server "transformers<5"
  ```

- **Transformers 5 bundle** (LightOnOCR, GLM-OCR, GLiGuard, GLiNER2.5 entity and Decide models, Privacy Filter, and the TopK-Embed-V1
  multi-vector models, and Perplexity decisions) — requires `transformers` 5.x, and is served with `-b transformers5`. The GLiNER2.5-Decide
  and entity models also need `gliner2` 2.x. `sie-server` itself asks for `gliner2<2`, which the default bundle's GLiNER2
  models need, so pip reports that conflict when the second command below installs 2.x; the transformers5
  bundle's GLiNER2 models are pinned to 2.0.0:

  ```bash
  pip install sie-server "transformers>=5.17,<6"
  pip install "gliner2==2.0.0"  # for the GLiNER2.5 and GLiNER2 PII models
  sie-server serve -b transformers5
  ```

  Privacy Filter also requires the official pinned runtime. The bundle installs
  this dependency; for a native installation, use:

  ```bash
  pip install "opf @ git+https://github.com/openai/privacy-filter@f7f00ca7fb869683eb732c010299d901457f19c3" "tiktoken==0.12.0"
  ```

## Quick Start

```bash
sie-server serve --port 8080 --device cuda:0
```

### Perplexity typed decisions

`perplexity-ai/pplx-decider-v1.1-27b` uses its native 255-option decision head
in the `transformers5` bundle. It accepts text through `extract` with typed
`output_schema` questions, `/v1/systemone`, and the OpenAI-compatible
`/v1/decisions` route. The upstream image input is not supported by this adapter.

The pinned checkpoint's attention mode, pooling and calibration temperature
are preserved. Every state/question pair runs as a separate row, with at most
8,192 tokens including the chat template and answer options. Oversized items
return `INPUT_TOO_LONG`; inputs are never truncated. The request limits are
32 questions, 255 options per question, 1,024 options total and 65,536 question
characters. An item may occupy at most 32,768 tokens across all its rows.
The decision compatibility routes also apply their existing limits, including
at most 10 levels for a score question.
Input usage counts every complete row processed; no output tokens are generated.

```python
from sie_sdk import SIEClient

client = SIEClient("http://localhost:8080")
result = client.extract(
    "perplexity-ai/pplx-decider-v1.1-27b",
    ["The customer was charged twice and wants a refund."],
    output_schema={
        "intent": {
            "type": "choice",
            "instructions": "Which team should handle this message?",
            "criteria": {"billing": "Payments and refunds", "support": "Technical problems"},
        },
    },
)
```

Use a GPU with enough memory for the approximately 52 GB BF16 checkpoint plus
inference activations. This model uses native forward passes, not generation,
and has no runtime LoRA or image support in this adapter.

### LightOnOCR-3 extraction

`sie_server.adapters.lighton_ocr.adapter:LightOnOCR3Adapter` supports the
Qwen3.5-based LightOnOCR-3 4B and 0.8B checkpoints in the `transformers5`
bundle. Pin the checkpoint revision when configuring the profile; it applies
to both the processor and model. CUDA execution defaults to BF16 and rejects
FP16.

Send exactly one image per item to `extract`. Omit `instruction` for plain OCR,
or set it to exactly `grounding` for text with normalized layout coordinates.
These are the checkpoint's two trained modes; arbitrary prompts and thinking
are unsupported. The processor receives the decoded RGB image and keeps its
complete tensor dictionary, including `image_grid_thw`.

Generation uses sampling with one beam, `temperature=0.1` and `top_p=1.0`,
following the [project's sampling recommendation](https://github.com/lightonai/LightOnOCR/blob/36755d461be079737860a5f03ae0c803501269e9/README.md).
Runtime options may set a positive finite temperature, `top_p` in `(0, 1]`,
and `max_new_tokens` from 1 through the profile's configured cap (4096 by
default). The returned `markdown` entity preserves raw decoded text, including
grounding markers and whitespace; margin filtering or Markdown cleanup must
be applied separately. Each image is metered as one page.

The catalog's `lightonai/LightOnOCR-3-4B` default profile serves the same
contract through `SGLangVisionExtractAdapter` in the `sglang-vision-extract`
bundle, with continuous batching. Its loadtime options set `temperature: 0.1`,
`top_p: 1.0`, `chat_template_kwargs: {enable_thinking: false}`, no system
prompt and `allowed_instructions: [grounding]`. That adapter trims leading and
trailing whitespace from the decoded text. The `transformers` profile keeps
`LightOnOCR3Adapter`. Other `SGLangVisionExtractAdapter` profiles keep greedy
decoding (`temperature: 0.0`) and accept any instruction.

See the [official LightOnOCR-3 model card](https://huggingface.co/lightonai/LightOnOCR-3-4B)
for the trained input and output formats. The existing `LightOnOCRAdapter`
continues to serve LightOnOCR-2 with its own processor and generation behavior.

### TensorRT-LLM generation

TensorRT-LLM buffers completion token IDs and emits the final text together only
after generation finishes and the terminal stream is verified, even with
`stream=True`. This preserves context-sensitive spacing and punctuation and
removes matched stop sequences consistently from returned text, completion-token
usage, and logprobs. Long completions delay the first visible text; existing
request timeouts still apply. Other adapters are unaffected.

### Native ZeRank 2 scoring

`zeroentropy/zerank-2-reranker` runs in the default bundle using its saved
SentenceTransformers CrossEncoder modules and publisher chat template: query
as SYSTEM, document as USER, followed by the plain assistant prefix. Scores are
raw logits for the publisher's positive token, 9454; higher values indicate
greater relevance. They can be negative or greater than one and are not
probabilities. Put retrieval conditions in the query; a separate `instruction`
parameter is unsupported.

The profile accepts complete encoded rows of at most 32,768 tokens, including
the query, document and template. An oversized row returns `INPUT_TOO_LONG`
before model execution. Runtime `max_seq_length` can lower this limit. The
adapter keeps the full source and never reports a score for a truncated prefix.
`usage.input_tokens` counts the complete native row for each pair, excluding
left padding and including template tokens. Native forward batches use at most
32,768 padded tokens regardless of the outer scheduling estimate.

### NLI zero-shot classification usage

The NLI zero-shot classifiers (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`,
`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`,
`MoritzLaurer/ModernBERT-base-zeroshot-v2.0`, `cross-encoder/nli-deberta-v3-base`
and `facebook/bart-large-mnli`) score each label as its own (text, hypothesis)
pair: the item's text next to the hypothesis template filled in with the label
(`This text is about {}.` by default). `usage.input_tokens` counts, for each
item, the tokens of all its pairs after truncation to the model's
`max_sequence_length`, special tokens included, as the cross-encoder rerankers
count each (query, document) pair. An item classified against n labels
therefore counts its text n times. A request may carry at most 1,000 labels.

### GLiClass usage

The shipped GLiClass and Opir profiles read a 1,024-token window
(`max_sequence_length`) that holds the label prompt, the instruction and
examples, and the document. That is the default window of the gliclass
pipeline and of its training script. The encoders use relative (DeBERTa) or
rotary (ModernBERT) positions, so the window is not a position-table limit.

For GLiClass classification, `usage.input_tokens` counts each item's document
tokens plus the instruction and few-shot example texts sent with the request,
because the model encodes that text again for every item. Label names,
including the labels attached to examples, are not counted. With an
instruction or examples, an item's count is capped at the model's
`max_sequence_length` minus the label prompt, unless the document count alone
is already higher. An item refused because its document pushes the labels out
of the window returns a per-item `INPUT_TOO_LONG` error and counts nothing. A
request that sends no instruction or examples is counted exactly as before.

The models that read the labels before the document (`prompt_first` in the
checkpoint's config: every shipped GLiClass model except `gliclass-small-v1.0`,
`gliclass-base-v1.0` and `gliclass-large-v1.0`) cut the document to the room
the label prompt and instruction leave, and count only the part of the document
they read, as `truncate_text` would. When the labels leave fewer than 8 tokens
for the document (a margin for tokenization at the boundary), each item returns
a per-item `INPUT_TOO_LONG` error and counts nothing, rather than being scored
with little or none of its document. With `overflow_policy` `truncate_text` or
`error`, the document is cut to that room or checked against it instead, and a
request whose labels leave no room at all is refused with `INPUT_TOO_LONG`, as
before.

With `options={"overflow_policy": "error"}`, an item whose document does not
fit whole next to the labels returns a per-item `INPUT_TOO_LONG` error and
counts nothing, while the other items succeed. Concurrent requests that share
labels and options are batched into one model call, so an over-long document
fails only its own item, never another request's.

The instruction and each example text may be at most 2,048 characters, and
together with the example labels at most 8,192 characters. Up to 32 examples
are accepted, and they must leave room for the document in the model window.
Label names are refused when their total length exceeds 16 characters per
token of the window (16,384 characters for a 1,024-token model), more than any
label prompt can fit.

### GLiClass CUDA graphs

A GLiClass forward on a DeBERTa encoder launches about a thousand small GPU
kernels, so at small batch sizes the CPU that launches them is the bottleneck.
A CUDA graph records the kernel launches of one input shape once; a replay
launches them all in one call. Graphs are an operator setting, fixed when the
model loads, in the model profile:

```yaml
profiles:
  default:
    adapter_options:
      loadtime:
        cuda_graphs: bucketed
```

A graph records the encoder. The scoring head (label features, pooling and
scorer) runs eagerly on the graph's output with each forward's own number of
labels, so one graph serves every label count.

| Value | Shapes recorded | Scores |
|--|--|--|
| `off` (default) | none | eager |
| `exact` | each (batch size, sequence length) seen twice; the 64 most recently used are kept | bit-identical to eager |
| `bucketed` | sequence lengths padded up to a multiple of 64 tokens at a 1,024-token window (32 at 512); batches of up to eight rows keep their size, larger ones pad up to a power of two: a fixed set of shapes, all kept | padding moves fp16 probabilities (see below) |

A request can send `options={"cuda_graphs": "off"}` to run eagerly on a model
loaded with graphs. It cannot turn graphs on: any other value is refused.

Padding is masked, and padded rows are dropped before scoring, but a longer
sequence or a larger batch rounds fp16 sums differently, the same kind of
change batching requests together makes. A small change can flip a near tie
between the top two labels, and eager execution flips some near ties too,
depending on which inputs share its batch.

**When a speed-up ships enabled.** A shipped GLiClass profile enables a
speed-up that changes scores, such as `bucketed` graphs, only when, on the
evaluation sets below, both of the first two conditions hold against eager
execution, or the third holds against float32:

1. no probability moves by more than 0.02;
2. every answer whose top label changes had an eager top-two margin smaller
   than eager's own regrouping noise, δ_eager: the largest probability change
   eager execution makes on the same inputs when they share a batch with other
   inputs;
3. against a float32 reference of the same checkpoint, the speed-up is at
   least as accurate as the current path: its largest change against float32
   is no larger than the current path's, and it changes no more top labels (or,
   for scores, reorders no more pairs) against float32 than the current path
   does.

A top label that changes under the second condition was a near tie that eager
fp16 execution already flips under batching. The third condition admits a
speed-up that rounds differently from the current path, by more than eager's
own batching noise, but lands no farther from float32. The ModernBERT
adapters below apply the same rule to their scores.

**Evaluation sets.** The main set is the 384 CVE descriptions from
`examples/typed-decisions`, three questions each, asked one at a time, as
separate groups and as joint groups, plus 65 long documents: 4,041 answers per
model. The instruction set asks the same descriptions the three questions with
short labels and the question as the instruction: 1,152 answers. The rule
applies to each set on its own. The eager reference sends one input per
request. δ_eager is the largest change among four eager runs that batch each
input with others: in pairs, and in groups of eight in dataset order and in two
shuffled orders. `bucketed` sent the inputs one per request three times (long
documents twice), so graphs were recorded and then replayed. Measured on an L4
in fp16 at each model's 1,024-token window; each cell gives the main set, then
the instruction set:

| Model | Largest probability change | δ_eager | Top labels changed (largest eager margin) | Shipped profile |
|--|--|--|--|--|
| `gliclass-small-v1.0` | 0.0059 / 0.0034 | 0.0066 / 0.0039 | 2 (0.0000) / none | `bucketed` |
| `gliclass-base-v1.0` | 0.0054 / 0.0056 | 0.0061 / 0.0061 | none / none | `bucketed` |
| `gliclass-large-v1.0` | 0.0059 / 0.0088 | 0.0078 / 0.0107 | none / none | `bucketed` |
| `gliclass-base-v3.0` | 0.0054 / 0.0034 | 0.0063 / 0.0054 | 1 (0.0005) / none | `bucketed` |
| `gliclass-large-v3.0` | 0.0093 / 0.0107 | 0.0088 / 0.0122 | 2 (0.0020) / none | `bucketed` |
| `gliclass-instruct-base-v1.0` | 0.0066 / 0.0063 | 0.0095 / 0.0063 | 4 (0.0056) / none | `bucketed` |
| `gliclass-instruct-large-v1.0` | 0.0088 / 0.0039 | 0.0125 / 0.0054 | 7 (0.0017) / 1 (0.0006) | `bucketed` |
| `opir-multitask-large-v1.0` | 0.0144 / 0.0146 | 0.0247 / 0.0225 | none / none | `bucketed` |
| `opir-multitask-multilang-v1.0` | 0.0756 / 0.0093 | 0.0848 / 0.0122 | 15 (0.0260) / none | `off` |
| `gliclass-multilang-mini` | 0.0347 / 0.0479 | 0.0537 / 0.0376 | 6 (0.0056) / 3 (0.0618) | `off` |

A top label that changed did so in every pass; the table counts each answer
once.

Requests of up to eight rows run at their own batch size, so only their length
is padded; a forward of nine rows or more pads up to a power of two (9 rows to
16) or to the largest batch the token bound allows at its length. A request of
three questions as separate groups is three rows. The table above was measured
when a batch padded up to a power of two (3 rows to 4, 5 to 8). With batches of
up to eight rows keeping their size, the 384 CVE descriptions were asked again
in 1-, 3-, 5- and 8-item requests with the weakness question's eight labels,
and as three questions in separate groups, and compared with eager execution of
the same requests: the eight models that ship with graphs moved no probability
by more than 0.0127 (`opir-multitask-large-v1.0`, three questions) and 0.0098
for every other model, and every top label that changed had an eager margin
below the model's δ_eager (the largest was 0.0056, on
`gliclass-instruct-base-v1.0`). Eight CVE descriptions in one batch are over
the token bound of the DeBERTa-v3-large models and run eagerly, so the
eight-row shapes of those four models were checked on requests of eight short
news texts with eight labels instead: against eager execution no probability
moved by more than 0.0083, and no top label changed. One-row forwards replay
the same shapes as before.

`exact` changed nothing. A forward whose batch size is already a bucket (one
item, for example) scores exactly as it did before batches were padded.

The DeBERTa-v3 models meet the rule and their shipped profiles load with
`bucketed` graphs. The two mDeBERTa models, `gliclass-multilang-mini` and
`opir-multitask-multilang-v1.0`, moved probabilities by more than 0.02 and
load with `off`. To run a model eagerly, set `cuda_graphs: off` in its
profile, or send `options={"cuda_graphs": "off"}` with a request.

Graphs apply on CUDA to the DeBERTa-based GLiClass models: the v1.0 models,
`gliclass-base-v3.0` and `gliclass-large-v3.0`, the base and large instruct
models, the Opir multitask models and `gliclass-multilang-mini`. The
ModernBERT-based models (the edge models, `gliclass-multilang-edge`, the Opir
edge models and `gliclass-modern-{base,large}-v3.0`), CPU and MPS run without
graphs with any value, and the load logs a warning. On CUDA, the
ModernBERT-based models can run their encoder on the flash-attention path
below instead.

**Shapes.** A graph holds at most 2,048 tokens (batch size times padded
length), or 1,024 for encoders wider than 768 such as DeBERTa-v3-large.
Larger forwards are bound by the GPU rather than by kernel launches and run
eagerly: on an L4, a `gliclass-large-v1.0` forward stops gaining from a graph
at about 1,000 tokens, a `gliclass-base-v1.0` forward at about 2,000. In
`bucketed` mode at the 1,024-token window that leaves a fixed set of shapes,
48 for the DeBERTa-v3-large models and 84 for the base and small ones (84 and
136 at a 512-token window), and the model keeps a graph for every one. Once
they are recorded, every forward under the token bound replays a graph,
whatever mix of label counts, batch sizes and lengths the traffic has, and no
request's shapes push out another's.

**Padding costs GPU time.** A replay is not bound by kernel launches, so it
costs the GPU time of its padded shape, which grows with the tokens in it. On
an L4, `gliclass-instruct-large-v1.0` replays three rows of 192 tokens in
19 ms and four rows in 24 ms; eight rows of 96 tokens in 21 ms and eight rows
of 128 tokens in 31 ms. So two paddings are kept small. A batch of up to eight
rows keeps its size instead of padding up to a power of two. And a forward of
eight rows or more, where short texts are bound by the GPU, buckets its length
at half the width (32 tokens at a 1,024-token window, 16 at 512), unless the
window's bucket pads fewer tokens. In process on an L4, eight short texts per
request went from 31.8 to 23.3 ms on `gliclass-instruct-large-v1.0` and from
12.4 to 11.2 ms on `gliclass-base-v3.0`, and three questions about one record
from 25.8 to 21.3 ms and from 11.2 to 8.9 ms; one text per request did not
change (7.8 and 3.6 ms).

Nothing is recorded at load. A shape is recorded the first time a request
needs it (the second time in `exact` mode), and the new graph's first replay
answers that request, which takes 52 to 66 ms against 43 ms for an eager
forward on `gliclass-large-v1.0` on an L4. Recording is rationed (see below),
so traffic that needs every shape has them all recorded within about two
minutes.

**Memory, and other models on the same GPU.** Graph memory counts as device
memory in use, but it is not attributed to the model: under memory pressure
the server evicts whole models, least recently used first, which may be
another model. A model's graphs share one memory pool and write their output
into one shared buffer, and the driver keeps a copy of each graph: about 7 MB
for the `gliclass-large-v1.0` encoder. The runner adds up the device memory
its graphs hold (what each recording took, plus the tables and buffers they
read) against 4% of the device's memory (900 MB on an L4). On an L4, at the
1,024-token window, all of a model's `bucketed` shapes, recorded largest
first, took 743 to 777 MB for the large GLiClass models, 777 MB for
`opir-multitask-large-v1.0`, 760 to 834 MB for the base models and 578 MB for
`gliclass-small-v1.0`, plus about 150 MB cached on the recording stream (its
cuBLAS workspace and one warm-up row). Recorded in another order, the same
shapes take more, because the shared pool grows in steps: recorded smallest
first, the budget fills with the largest shapes of each batch size left
unrecorded (9 of 48 on `gliclass-instruct-large-v1.0`, 19 of 84 on
`gliclass-base-v3.0`). Those are the shapes a graph gains least on, and they
run eagerly. On a smaller GPU, where the shapes do not
all fit, recording stops at the budget and the graphs already recorded keep
replaying; the other shapes run eagerly. In `exact` mode, whose shapes are
unbounded, a model past its budget drops every graph, returns their memory to
the device and records again.
Graphs are also released when the model unloads, and when one of its forwards
runs out of memory.

While a graph records, PyTorch's caching allocator does not free cached blocks
to satisfy other allocations, so another model on the same GPU that needs
memory in that window (about one forward) can run out of memory where it
otherwise would not. Recording is kept rare to limit this: one recording at a
time in the process, none while less than a tenth of the device's memory is
free, and per model a budget of 16 recordings that refills at one every 2
seconds, so a model records at most 16 graphs in quick succession, one after
another, then about one every 2 seconds. If recording a graph runs out of
memory, the request still gets its eager answer; the model drops its graphs and
records nothing for a minute. If the new graph's first replay runs out of
memory, the model also drops its graphs and records nothing for a minute, but
the request fails with that error, as an eager forward that runs out of memory
does.

Both limits are approximate. The free-memory check reads the device once,
before recording, so a model loading at the same moment can still meet one
recording. The memory budget is checked after each recording, so a model's
graphs can exceed it by one recording. On a GPU shared with other models,
leave memory headroom, or enable graphs only where the model has the GPU to
itself. Usage and billing do not change.

**Failures and counters.** A shape that fails to record for a reason other
than memory runs eagerly from then on, and the failure is logged as a warning
with its traceback. After three such shapes, the model runs eagerly for the
rest of the process, logged as an error. The scoring head is not part of that
count: it reads the request's own inputs, so an error there fails that request
as it would in an eager forward. Each model counts the forwards it
replays, records, and runs eagerly (by reason: past the token bound, recording
paused, budget full, and so on), and logs the counts every ten minutes while it
serves requests.

### GLiClass ModernBERT flash attention

The GLiClass models built on ModernBERT or mmBERT (`gliclass-edge-v3.0`,
`gliclass-instruct-edge-v1.0`, `gliclass-multilang-edge`, `opir-edge-v1.0`,
`opir-edge-multilang-v1.0`, `gliclass-modern-base-v3.0` and
`gliclass-modern-large-v3.0`) can run their encoder through the
flash-attention layer stack that SIE's ModernBERT embedding, late-interaction,
cross-encoder and Laya adapters share. The rows of a forward are packed into
one token stream without padding, each row attends only to itself through
`flash_attn_varlen_func`, and the RoPE tables are built once, at load. The
gliclass scoring head (label-token features, pooling, projections and scorer)
runs unchanged on the encoder output. Label groups, instructions, examples,
overflow policies, usage and per-item errors behave as before. It is an
operator setting:

```yaml
profiles:
  default:
    adapter_options:
      loadtime:
        modernbert_flash: true
```

`modernbert_flash: single-label` limits it to single-label requests; a
request with `"classification_type": "multi-label"` then runs the gliclass
forward. It applies to float16 and bfloat16 weights on CUDA GPUs with flash-attn
(Ampere or newer). On CPU, MPS, older GPUs or without flash-attn, the model
runs the gliclass forward as before, and the load logs why. DeBERTa-based
models ignore the setting. If the flash path fails with an error other than
running out of memory, the gliclass forward answers that request, and the
model logs the error and stays on the gliclass forward until it is reloaded.
The shipped profiles enable it where the scores
met the margin rule below: for every request on `gliclass-modern-base-v3.0`,
and for single-label requests on `gliclass-multilang-edge` and
`gliclass-modern-large-v3.0`. The other four models keep the gliclass forward.

On a GPU with flash-attn, the gliclass forward already runs the Hugging Face
ModernBERT flash-attention path. That path unpads and repads every batch,
rotates queries and keys with one fused kernel per layer, and runs its MLPs
through `torch.compile`. Classification forwards are small, so the host
launching kernels, not the GPU, bounds most of them, and the flash path needs
less host time per forward. On an L4, a one-item `gliclass-edge-v3.0` forward
keeps the GPU busy for 2.4 ms of its 12.7 ms on the gliclass forward, and for
0.85 ms of 8.8 ms on the flash path. The flash path runs the rotation and the
MLP activation as several separate kernels, though, so larger forwards, which
the GPU bounds, are faster on the gliclass forward. A forward with more packed
tokens than a bound therefore runs the gliclass forward: 4,096 tokens for
encoders up to 384 wide, 2,048 up to 768, and 1,024 wider. On an L4,
`gliclass-modern-large-v3.0` is 1.13x faster on the flash path at 1,024
packed tokens and 0.90x at 1,289; `gliclass-modern-base-v3.0` is 1.27x at
2,048 and 0.96x at 2,560.

Latency is the median of 400 one-item requests on the CVE descriptions from
`examples/typed-decisions`: one question with eight labels, or three questions
as separate label groups. Each throughput request holds 64 items, one in eight
of them a long document, and one question. Both paths ran in one process on an
L4, from the gliclass forward to the flash path:

| Model | One item, one question | One item, three separate groups | 64 items per request |
|--|--|--|--|
| `gliclass-edge-v3.0` | 13.6 -> 11.1 ms | 15.1 -> 12.6 ms | 351 -> 397 items/s |
| `gliclass-instruct-edge-v1.0` | 14.2 -> 11.9 ms | 15.4 -> 13.1 ms | 334 -> 376 items/s |
| `gliclass-multilang-edge` | 25.8 -> 20.6 ms | 26.8 -> 21.6 ms | 235 -> 279 items/s |
| `opir-edge-v1.0` | 13.8 -> 11.3 ms | 14.9 -> 12.4 ms | 287 -> 313 items/s |
| `opir-edge-multilang-v1.0` | 28.4 -> 22.6 ms | 29.7 -> 24.0 ms | 207 -> 241 items/s |
| `gliclass-modern-base-v3.0` | 25.3 -> 19.9 ms | 26.7 -> 21.3 ms | 229 -> 253 items/s |
| `gliclass-modern-large-v3.0` | 32.3 -> 25.1 ms | 34.2 -> 27.1 ms | 173 -> 173 items/s |

**Scores.** The two paths round float16 sums differently. We compared them on
the 384 CVE descriptions from `examples/typed-decisions`, three questions
each: asked one at a time, with an instruction, with a few-shot example, as
separate and as joint groups, and in requests of eight, plus 65 long documents
under `truncate_text`. That is 7,497 answers per model. A shipped profile
enables the flash path only when the model meets the margin rule against the
gliclass forward: (i) no probability moves by more than 0.02, and (ii) every
answer whose top label changes had a top-two margin, on the gliclass forward,
smaller than the gliclass forward's own batching noise. The batching noise is
the largest probability change the gliclass forward shows between an item sent
alone and the same item in a request of eight with the same options, over the
same kinds of requests.

| Model | Largest probability change | Top label changed | Largest margin of a changed answer | Batching noise | Shipped profile |
|--|--|--|--|--|--|
| `gliclass-multilang-edge` | 0.015 | 18 | 0.0046 | 0.016 | on (single-label) |
| `gliclass-modern-base-v3.0` | 0.010 | 22 | 0.0029 | 0.0061 | on |
| `gliclass-modern-large-v3.0` | 0.015 | 4 | 0.0039 | 0.014 | on (single-label) |
| `gliclass-edge-v3.0` | 0.016 | 24 | 0.0098 | 0.0066 | off (2 answers over the noise) |
| `gliclass-instruct-edge-v1.0` | 0.012 | 5 | 0.0077 | 0.0062 | off (1 answer over the noise) |
| `opir-edge-v1.0` | 0.018 | 12 | 0.019 | 0.011 | off |
| `opir-edge-multilang-v1.0` | 0.037 | 30 | 0.028 | 0.032 | off |

Multi-label scores (independent sigmoids) were checked the same way, with the
rule read for the 0.5 threshold at which a multi-label group selects its
labels: (i) no score moves by more than 0.02, and (ii) every score that
crosses 0.5 was closer to 0.5 than the gliclass forward's own batching noise.
Each request kind was sent one item at a time and eight at a time, 59,220
scores per model:

| Model | Largest score change | Scores crossing 0.5 (largest distance from 0.5) | Batching noise | Multi-label |
|--|--|--|--|--|
| `gliclass-modern-base-v3.0` | 0.0088 | 18 (0.0020) | 0.0059 | flash |
| `gliclass-modern-large-v3.0` | 0.024 (joint groups) | 23 (0.0044) | 0.039 | gliclass forward |
| `gliclass-multilang-edge` | 0.039 (few-shot examples) | 18 (0.0073) | 0.032 | gliclass forward |

Usage and per-item errors were identical in every answer. Against the same
checkpoints in float32, the flash path is as accurate as the gliclass forward:
over the seven models, the gliclass forward's top label differs from float32
in 128 answers, the flash path's in 126. To run a model on the other path, set
`modernbert_flash` in its profile.

### GLiNER2.5-Decide usage and limits

The GLiNER2.5-Decide models (`fastino/GLiNER2.5-Decide`, `GLiNER2.5-multi-Decide`,
`GLiNER2.5-Decide-1B`) run on `gliner2` 2.x, which the transformers5 bundle
pins (the `transformers5` image, or a native install as described above). Each
item is one encoder row: every question's (or label group's) name,
instruction, and labels, then the document. One forward pass answers them all,
so the questions of a request are not independent: adding or changing one can
change another's probabilities and score. `usage.input_tokens` counts the
document tokens the model reads plus the tokens of the instructions and label
descriptions (criteria) sent with the item, as Laya and GLiClass count
instructions and criteria. Question ids, group names, and label names are not
counted, and an item that returns an error counts nothing.

A request takes at most 64 questions or label groups, 64 options per question,
and 1,024 options in total. Question ids and group names may have 128
characters, labels 256, and each instruction or description 2,048, with 65,536
characters in all. Strings that contain one of the model's prompt markers
(`[L]`, `[P]`, `[DESCRIPTION]`, ...) are refused. The questions may take at most
512 tokens, or half the model's window when that is less: 256 of
`GLiNER2.5-Decide`'s 512 tokens, 512 of the others' 2,048. This bounds the
uncounted question and label tokens read with each item; a request needing more
fails with `INPUT_TOO_LONG`. The
document is read up to the whole words that fit in the rest of the window; a
word longer than 4,096 characters, text past 64 characters per token of the
window, or 4 words per token of the window also ends what is read. Words are
split as gliner2 splits them, in linear time. A conversation (a list state) is
read from its newest turn back; a run of more than 4,096 characters without a
space is read only in its last 4,096 characters, and reading stops there. An item none of whose words fits, or that does
not fit whole with `options={"overflow_policy": "error"}`, returns a per-item
`INPUT_TOO_LONG` error while the other items succeed.

### GLiNER2.5 entity extraction

`fastino/gliner2.5-base-v1`, `fastino/gliner2.5-small-v1`, and
`fastino/gliner2-privacy-filter-PII-multi` use the native `AutoExtractor` API in
the Transformers 5 bundle. The base and small checkpoints use boundary
extraction; the PII checkpoint uses span extraction. Existing GLiNER2 v1 and
Decide routes keep their own adapters.

```python
from sie_sdk import SIEClient

with SIEClient("http://localhost:8080") as client:
    result = client.extract(
        "fastino/gliner2.5-base-v1",
        {"text": "Ada joined Acme in London."},
        labels=["person", "organization", "location"],
        options={
            "threshold": 0.5,
            "entity_descriptions": {
                "person": "Named human, including surname aliases.",
                "organization": "Named company or public institution.",
            },
        },
    )
    print(result["entities"])
```

The profiles allow one complete 4,096-token encoder row, including the schema
prompt. The prompt may take up to 2,048 tokens, and each label at most 128
characters. Labels are literal names and are returned exactly. Optional
`options.entity_descriptions` maps any subset of those exact names to string
descriptions in the native schema's separate prompt channel. Unspecified
labels remain undescribed; omitting the option or using an empty mapping keeps
the existing label-list behavior. Unknown keys, non-string values and reserved
prompt markers such as `[SEP_TEXT]` in labels or descriptions are rejected.
Inputs that do not fit return per-item `INPUT_TOO_LONG` errors; split long
documents into deliberate smaller items instead of relying on prefix truncation. Entity
offsets index the original text. Usage counts the native document segment,
including the processor's sentence-end punctuation; schema labels, descriptions
and markers are not billed. Failed items count zero. These new profiles default
to float32; another precision is an explicit profile setting.

Before native preprocessing, the adapter also bounds each source to 64
characters and four native words per configured row token, with at most 4,096
characters per word. The label list may contain at most `max_prompt_tokens`
entries. Each description and the combined label/description text are bounded
to 32 times `max_prompt_tokens` characters, counting label names repeated in
description prefixes. Inputs beyond these character and word bounds fail
explicitly before collation or inference. Descriptions also count toward the
actual 2,048-token prompt and complete 4,096-token row; they are not truncated.

### Privacy Filter extraction

`openai/privacy-filter` uses the official OPF runtime and the pinned
checkpoint's `original/` artifacts in the Transformers 5 bundle.
Its `o200k_base` vocabulary uses tiktoken's cache, independently of the
Hugging Face cache. An empty cache requires HTTPS access to
`openaipublic.blob.core.windows.net`. For offline use, populate a persistent
`TIKTOKEN_CACHE_DIR` with `tiktoken.get_encoding("o200k_base")` beforehand and
make that same directory available when serving; Hugging Face offline settings
do not populate this cache.

The model
extracts `account_number`, `private_address`, `private_email`, `private_person`,
`private_phone`, `private_url`, `private_date`, and `secret`. Omit `labels` to
return all categories, or select a subset; this filters detections without
changing the model's trained policy.

```python
from sie_sdk import SIEClient

with SIEClient("http://localhost:8080") as client:
    result = client.extract(
        "openai/privacy-filter",
        {"text": "Email Ada at ada@example.com."},
        labels=["private_person", "private_email"],
        options={"span_entry_bias": 0.5},
    )
    print(result.entities)
```

The default profile reads one complete input of up to 128,000 native tokens
with the checkpoint's native BF16 weights and FP32 controls. A shorter
`max_seq_length` or float32 is an explicit profile setting.
Oversized items fail before inference, and text that the tokenizer cannot
preserve is rejected. Spans use original-source character offsets. Usage is
the exact native token count; failed items count zero.

The official Viterbi decoder starts from the checkpoint's six calibrated
transition biases. `span_entry_bias` adds a delta in [-10, 10] only to
`transition_bias_background_to_start`; zero preserves the checkpoint default.
Positive values favor entering a privacy span, negative values discourage it.
No confidence threshold, dynamic instruction or schema is supported. The
native runtime does not produce span confidence scores: response
`data.confidence_scores_available` is false, and the HTTP API's existing
`score=1.0` default denotes an unscored detection rather than a probability.

### PII redaction with GLiNER PII and NuNER Zero

`superlinked/pii-redact` serves `urchade/gliner_multi_pii-v1` and
`numind/NuNER_Zero` as one extract model. It composes them as the PII redaction
study in `examples/redact` does:

- A document of more than 300 words, counted as GLiNER splits words, is read
  as 300-word windows sharing 50 words.
- Both models read every window with the request's labels. Spans scoring below
  0.6 are dropped.
- The two models' spans are unioned: one span per start, end and label, at the
  higher score.
- Every other whole-word, case-sensitive occurrence of a token of three or more
  letters from a `person` span is returned as a `person` span too, unless it is
  already inside a span. It scores as the best `person` span it came from.

A request without `labels` uses the study's 36 personal-data types. Spans can
overlap, so mask their union:

```python
import re

from sie_sdk import SIEClient

text = "Ann Lee wrote from ann.lee@example.com. Call Lee back."
with SIEClient("http://localhost:8080") as client:
    result = client.extract("superlinked/pii-redact", {"text": text})
hidden = [False] * len(text)
for e in result["entities"]:
    hidden[e["start"] : e["end"]] = [True] * (e["end"] - e["start"])
print(re.sub("\0+", "[REDACTED]", "".join("\0" if h else c for c, h in zip(text, hidden))))
```

A request may override `score_floor`, `window_words`, `window_overlap`,
`propagate_label` (`null` turns name propagation off), `name_min_letters` and
`default_labels` in `options`. Both models always run at their own defaults.
Usage counts each model's input tokens for every window it reads, so a word
two windows share is counted in both.

### ModernBERT CUDA graphs

The ModernBERT flash-attention adapters, for dense embeddings
(`modernbert_flash`), late interaction (`colbert_modernbert_flash`) and
reranking (`modernbert_flash_cross_encoder`), run a packed, unpadded token
stream through hundreds of small kernels per forward (about 500 for a
22-layer encoder). At the sizes of a query
or a few short documents, the host that launches those kernels is the
bottleneck: on an L4, a one-query forward of `GTE-ModernColBERT-v1` keeps the
GPU busy for about 2 ms of its 12. A CUDA graph records the launches of one
shape once and replays them in one call. Graphs are an operator setting,
fixed when the model loads:

```yaml
profiles:
  default:
    adapter_options:
      loadtime:
        cuda_graphs: bucketed
```

`off` (the default) runs every forward eagerly. With `bucketed`, a graph
records the encoder (token embeddings, layers and final norm) for one packed
shape: the token count padded up to 64, 128, 192, 256, 384, 512, 768, 1,024
tokens and so on, a fixed number of sequence slots, and `max_seqlen` 512 (or
the bucket, when a row is longer). Padding tokens belong to no sequence, so
real rows attend only to their own tokens, as eagerly. The adapter's head
(pooling, the late-interaction projection or the reranking head) runs
eagerly on the real rows. Forwards past a token bound run eagerly: 2,048
tokens for encoders up to 384 wide, 1,024 up to 768, 512 wider. Past that,
the GPU rather than kernel launches bounds a forward, and padding to the
bucket costs more than a graph saves. A model with LoRA adapters loaded runs
eagerly, because its forward depends on which adapter is active.

Speed on an NVIDIA L4 with 8 vCPUs, eager and graphs in one process with
their requests interleaved (median latency, and items per second over the
run). SciFact queries and abstracts; documents are cut at the profile's
length (300 tokens for `GTE-ModernColBERT-v1`, 512 for
`mxbai-edge-colbert-v0-32m`):

| Model | 1 query | 8 queries | 1 document | Larger requests |
|--|--|--|--|--|
| `GTE-ModernColBERT-v1` | 18.1 → 2.6 ms | 360 → 1,857/s | 22.3 → 5.1 ms | within 2% |
| `gte-modernbert-base` | 17.5 → 2.4 ms | 378 → 2,414/s | 21.4 → 4.3 ms | within 1% |
| `mxbai-edge-colbert-v0-32m` | 9.2 → 1.1 ms | 658 → 3,345/s | 12.1 → 2.8 ms | within 2% |
| `granite-embedding-small-english-r2` | 10.4 → 1.2 ms | 581 → 4,551/s | 14.0 → 2.4 ms | within 2% |
| `gte-reranker-modernbert-base` | 1 pair: 17.3 → 4.3 ms | 8 short pairs: 351 → 1,307/s | | within 2% |

"Larger requests" are 8 and 64 abstracts, 32 items of mixed lengths, and 8
or 32 (query, abstract) pairs: about 2,000 to 16,000 packed tokens, mostly
past the token bound. The eager side depends on the host's CPU: on another L4
host with faster cores, a one-query `GTE-ModernColBERT-v1` forward took
12.2 ms eagerly and 2.5 ms from a graph.

Padding changes how many rows a matrix multiply sees, so outputs can move by
floating-point rounding, as batching requests together does. We compared
graphs with eager forwards on SciFact: the 300 test queries, 500 abstracts
(every one relevant to those queries, and random others) and, for the
reranker, 20 abstracts per query. Items were encoded one per request, on
both paths, and the eager path was also run with 8 queries or 4 abstracts
(3 pairs) per request, which is its own batching noise. Scores are query–
document dot products for dense models and MaxSim for late interaction:

| Model | Graphs: largest score change | Graphs: pairs reordered in a top 10 (largest eager margin of any reordered pair) | Eager batching: largest score change (largest margin of a reordered pair) |
|--|--|--|--|
| `GTE-ModernColBERT-v1` | none (bit-identical) | none | 0.013 (0.018) |
| `mxbai-edge-colbert-v0-32m` | none (bit-identical) | none | 0.13 (0.23) |
| `gte-modernbert-base` | 0.006 | 12 (0.008) | 0.008 (0.012) |
| `granite-embedding-small-english-r2` | 0.004 | 16 (0.005) | 0.015 (0.014) |
| `gte-reranker-modernbert-base` | 0.015 | 21 (0.006) | 0.025 (0.019) |
| `modernbert-embed-base` | 0.004 | 10 (0.004) | 0.005 (0.006) |
| `granite-embedding-97m-multilingual-r2` | 0.006 | 57 (0.008) | 0.015 (0.015) |
| `Reason-ModernColBERT` | 0.007 | 1 (0.007) | 0.022 (0.021) |
| `mLateOn` | 0.020 | 17 (0.029) | 0.037 (0.046) |
| `Iso-ModernColBERT` | none (bit-identical) | none | 0.20 (0.21) |

Graphs meet the rule a GLiClass speed-up ships under (see "When a speed-up
ships enabled" above), read for scores instead of label probabilities: no
score moves by more than 0.02 (for MaxSim, which sums over a query's tokens,
0.02 per query token), and every pair that graphs reorder had an eager margin
smaller than the eager path's own batching noise, the largest score change
eager execution makes when the same items share a request with others. For
embeddings, the rule reads on the retrieval order the vectors give, and the
vectors stay as close to eager as eager batching keeps them (cosine at least
0.9997 for the dense models). Retrieval quality on the
full SciFact corpus (5,183 abstracts, nDCG@10, the same requests on both
paths) did not change beyond that noise: `gte-modernbert-base` 0.7632 eager
and 0.7644 with graphs, `GTE-ModernColBERT-v1` 0.7573 and 0.7558, and
`gte-reranker-modernbert-base` reranking the top 20 of `gte-modernbert-base`
0.7760 and 0.7767.

The eager batching noise is largest for `mxbai-edge-colbert-v0-32m` and
`Iso-ModernColBERT` because their profiles compute in bfloat16, whose
significand is three bits shorter than float16's (the other late-interaction
models compute in float16). With more packed rows, cuBLAS picks a different
matrix-multiply kernel; in the forward we traced, the first outputs to differ,
by one unit in the last place, were those of the first layer's attention output
projection. A few document tokens amplify that rounding: punctuation, `[SEP]`
or the `[D] ` marker, close to where the encoder turns a token into an attention
sink. The token builds a large activation in one forward and not in the other,
so its vector turns. Per-token cosine to the one-item forward went as low as
0.82 for `Iso-ModernColBERT` on the abstracts above, while more than 99.8% of
tokens stayed above 0.999 on a sample of mixed lengths. The reference
implementation behaves the same way. PyLate in bfloat16 moves such tokens as
much between a document encoded alone and in a batch; in float32 neither PyLate
nor the Hugging Face forward moves them. In float32 on the CPU, this adapter's
packed forward gives each item the same vectors alone and in any batch, equal
to the Hugging Face forward (`tests/adapters/test_colbert_modernbert_flash_batch_invariance.py`).

`Iso-ModernColBERT` serves the recipe its checkpoint publishes for PyLate, as
`GTE-ModernColBERT-v1` does:

- `[Q] ` and `[D] ` markers;
- queries cut at 32 tokens and documents at 300 (its `long_context` profile
  keeps 8,192);
- the punctuation skiplist, which drops the document vectors that training
  never scores.

Its row above was measured with that recipe. With it, SciFact nDCG@10 rose from
0.7326 to 0.7573 (PyLate: 0.7574 in bfloat16, 0.7584 in float32).

The shipped profiles of every model on these adapters load with `bucketed`
graphs: `GTE-ModernColBERT-v1`, `Reason-ModernColBERT`, `mLateOn`,
`Iso-ModernColBERT`, `mxbai-edge-colbert-v0-32m`, `gte-modernbert-base`,
`modernbert-embed-base`, `granite-embedding-small-english-r2`,
`granite-embedding-97m-multilingual-r2` and `gte-reranker-modernbert-base`.
To run one of them eagerly, set `cuda_graphs: off` in its profile. A Candle
profile of these models sets its own load-time options, so it does not
inherit the setting.

A graph is recorded the first time a request needs its shape (the dense
adapter's warm-up records the smallest at load), and that request is answered
by its first replay. Recording follows the rules of the GLiClass graphs above,
and shares their process-wide limits: one recording at a time in the process,
none while less than a tenth of the device's memory is free, a budget of 16
recordings per model that refills at one every 2 seconds (16 in quick
succession, one after another, then about one every 2 seconds), and a
model's graphs within 4% of the device's memory (900 MB on an L4). A model's graphs, which share
one memory pool and one output buffer, held 40 to 105 MB on an L4 once every
shape its traffic needed was recorded. A recording that runs out of memory
drops the model's graphs and pauses recording for a minute; a shape that
fails to record for another reason runs eagerly from then on, and after
three such shapes the model runs eagerly for the rest of the process. Each
model counts the forwards it replays, records and runs eagerly (by reason),
and logs the counts every ten minutes while it serves requests.

**Fused rotary embedding.** The dense and late-interaction adapters rotate
queries and keys with about ten small PyTorch kernels per layer. With
`adapter_options.loadtime.fused_rope: true`, one Triton kernel per layer does
it instead, with flash-attn's rotary arithmetic (float32, rounded once, as
the Hugging Face flash-attention forward rotates): on an L4, 26.5 µs instead
of 178 µs per layer at 2,048 tokens. For the models below that is 1.13-1.18x
the speed of one-query requests (with CUDA graphs on both sides) and
1.17-1.22x the throughput of 64-abstract requests. Outputs change by rounding, more than the eager path's own batching
noise, so the option ships under the third condition of the rule above.
Against a float32 reference of each checkpoint, on the SciFact sets of the
graphs comparison, it is at least as accurate as the unfused rotation for
`modernbert-embed-base`, `Reason-ModernColBERT` and `mLateOn`, and their
profiles enable it. The other models load without it.

### Laya CUDA graphs

The Laya adapter answers typed questions with one encoder row per (state,
question) pair. One state with a few questions is a forward of a few rows, and
kernel launches, not the GPU, bound it: on an L4, a request of four rows and
710 tokens launches 680 to 860 kernels and keeps the GPU busy for 7 ms of a
22 ms forward on `laya-multilingual`, and for 17 ms of 28 on `laya`. The
adapter takes the load-time option the ModernBERT flash adapters have, and
records its encoder with the same runner:

```yaml
profiles:
  graphs:
    extends: default
    adapter_options:
      loadtime:
        cuda_graphs: bucketed
```

With it, a request whose rows hold at most 2,048 packed tokens replays the
encoder as one graph, and the decision head runs eagerly on the result. On
encoders wider than 768 the bound is 1,024 tokens. A larger request runs
eagerly as before: the GPU bounds it, and padding it to a graph's bucket costs
more than the launches save.

The three shipped checkpoints offer graphs as a second profile. Request
`convaiinnovations/laya:graphs`, `convaiinnovations/laya-typed-decisions:graphs`
or `convaiinnovations/laya-multilingual:graphs` in place of the model; the
default profile runs eagerly.

Speed on an NVIDIA L4 with 8 vCPUs: the forward of rows that are already
tokenized, eager and from a graph in one process, median of 15. One state is
a 118-word email with four questions:

| Model | 1 state: 4 rows, 710 tokens | 2 states: 8 rows, 1,420 tokens | 3 states: 12 rows, 2,130 tokens |
|--|--|--|--|
| `laya` | 29.0 → 23.2 ms | eager (39.6 ms) | eager |
| `laya-typed-decisions` | 28.3 → 23.3 ms | eager (39.7 ms) | eager |
| `laya-multilingual` | 21.8 → 9.3 ms | 23.1 → 17.8 ms | eager (24.4 ms) |

On the 1,024-wide checkpoints a graph at 1,416 tokens took 40.0 ms against
39.6 ms eagerly, which is where their bound comes from.

**Why graphs are not the default profile.** Padding to a bucket changes how
many rows a matrix multiply sees, so probabilities move by rounding. We sent
240 public texts with six questions each, one state per request, through a
graph and eagerly (1,440 answers per model), and measured the eager path's own
batching noise the same way: each state alone against the same state in a
request of eight.

| Model | Graphs: largest probability change | Graphs: top answers changed (largest eager margin) | Eager batching: largest probability change | Eager batching: top answers changed |
|--|--|--|--|--|
| `laya` | 0.091 | 4 (0.008) | 0.130 | 11 |
| `laya-typed-decisions` | 0.050 | 7 (0.008) | 0.079 | 13 |
| `laya-multilingual` | 0.034 | 3 (0.014) | 0.052 | 5 |

Graphs meet the second condition of the rule a speed-up ships enabled under
(see "When a speed-up ships enabled" above): every answer they change had an
eager margin far below the eager path's batching noise, and they move
probabilities less than sharing a request does. They do not meet the first, a
change of at most 0.02. Laya computes in bfloat16 through 22 or 28 layers, and
the adapter's eager answers already move by more than that with what else is
in the forward. So the default profiles stay
eager, and the `graphs` profile is there for callers who want the one-state
latency. Against a float32 forward of the reference package, the top answer
agrees on 1,427, 1,427 and 1,430 of the 1,440 rows from a graph, on 1,429,
1,431 and 1,433 eagerly with all rows in one call, and on 1,426, 1,428 and
1,433 from the package's own bfloat16 autocast.

**Forward size.** Rows run longest first in forwards of at most
`max_forward_tokens` padded tokens, a load-time option that stays at 16,384.
Called directly with 128 rows of up to 233 tokens (32 states with four
questions), the adapter takes 353, 353 and 139 ms at 4,096 on the three
models above, against 400, 399 and 154 ms at 16,384. Behind the server that
gain does not show: the server hands the adapter a 32-state request in
smaller calls already. Behind the sidecar, with 4,096, the same request took
371.5 against 372.5 ms and 370.6 against 374.5 ms on the 1,024-wide
checkpoints, and 162.6 against 155.7 ms on `laya-multilingual`.

**Fused rotary embedding.** The adapter also takes `fused_rope` (above). It
took 7 to 10% off 128-row requests, and no shipped profile enables it: with
it, agreement with the float32 reference was lower on two checkpoints with
all rows in one call (1,424 and 1,426 of 1,440, against 1,429 and 1,431).

The measurements, with their scripts, are in
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/main/blog-laya-on-sie/2026-10-13b).

### Qwen3 reranker serving options

`Qwen/Qwen3-Reranker-4B` and `Qwen/Qwen3-Reranker-0.6B` run on
`qwen2_flash_cross_encoder`: flash attention over unpadded rows, and an output
head cut down to the `yes` and `no` rows. A request's documents are tokenized
in one call. Three load-time options take out what that leaves. The default
profile turns on the one that leaves scores unchanged, and a profile named
`fast` (`Qwen/Qwen3-Reranker-4B:fast`) turns on all three:

```yaml
profiles:
  default:
    adapter_options:
      loadtime:
        fused_kernels: true
  fast:
    extends: default
    adapter_options:
      loadtime:
        fused_kernels: true
        prefix_reuse: true
        prefix_cache_mb: 512
        cuda_graphs: bucketed
        cuda_graph_tokens: 512
```

**`fused_kernels`.** Each layer's residual add and RMSNorm, q/k-norm, rotary
embedding and `silu(gate) * up` run as the Triton kernels of the Qwen3
embedding adapter (`qwen2_flash/_fused_layers.py`). They round where the
eager operations round, so scores are the same bit for bit. A model the
kernels cannot run (no Triton, or an MLP that is not SiLU-gated) logs a
warning and keeps the eager layers.

**`prefix_reuse`.** Every row a reranker scores starts with the same tokens
for one query: the chat template, the instruction and the query. Attention is
causal, so those tokens have the same states in every row. With the option,
a forward encodes each distinct prefix once and each row's document once, and
a row attends to its prefix's keys and values (`flash_attn_varlen_func` takes
keys with their own lengths, and aligns the causal mask to the bottom right).
For a question with 20 candidates that removes 19 copies of the prefix: 39%
of the tokens on the requests measured below. The rows are metered whole:
token counts do not change.

A prefix that comes back is kept across requests. That is the shape of a
yes-or-no decision: the question and its rules as the query, one message as
the document, the same prefix on every request. The second request with a
prefix stores its keys and values per layer, and later requests encode only
the message. The cache holds at most `prefix_cache_mb` of device memory (512
by default; `0` shares a prefix inside a forward only) and at most 64
prefixes, and drops the least recently used. One cached token takes 147 KB
for the 4B and 115 KB for the 0.6B, so a 145-token question takes 21 MB and
17 MB. A forward that runs out of memory drops the cache. The option applies
to the `qwen3` input format.

**`cuda_graphs: bucketed`.** A one-message request is small enough that the
GPU waits on the host launching 1,500 to 2,000 kernels. With the option, one row
over a cached prefix replays as a CUDA graph through the runner of the
ModernBERT graphs above: the tail (document and closing tokens) padded to
64, 128 and so on up to `cuda_graph_tokens`, with the graph reading the
cached keys and values in place. Without `cuda_graph_tokens` the bound is 512
tokens for a model up to 1,024 wide and 128 wider, sized for an L4; the
`fast` profile sets 512 for both models, which an H100 or an RTX PRO 6000
gains from on requests of 130 to 500 tokens (mean latency 11% to 24% lower on
a set with a median of 98 tokens) and which cost an L4 under 2% on the same
set. Longer tails, requests with several rows, and prefixes not yet cached
run eagerly. Up to 8 prefixes hold graphs at once,
within the memory one model's graphs may take (4% of the device), and a
prefix's graphs go when the cache drops it. Graphs need `prefix_reuse`.

Median latency of one request, one in flight, through the Rust sidecar with
the client in the server's container (three runs each, main next to the
change in one container). Rerank is one question with 20 candidates, about
4,600 tokens; a decision is one message after a 145-token yes-or-no question:

| Model | GPU | Rerank: before → default → `fast` options | Decision: before → default → `fast` options |
|--|--|--|--|
| `Qwen3-Reranker-4B` | H100 | 103.8 → 77.7 → 54.2 ms | 23.6 → 24.8 → 8.0 ms |
| `Qwen3-Reranker-4B` | RTX PRO 6000 | 140.0 → 120.1 → 88.2 ms | 23.5 → 23.2 → 11.1 ms |
| `Qwen3-Reranker-4B` | L4 | 952 → 761 → 480 ms | 56.7 → 53.6 → 40.5 ms |
| `Qwen3-Reranker-0.6B` | H100 | 36.1 → 24.1 → 21.2 ms | 15.9 → 14.2 → 4.3 ms |
| `Qwen3-Reranker-0.6B` | L4 | 179 → 131 → 85 ms | 28.6 → 28.8 → 8.4 ms |

Rerank throughput at 16 requests in flight goes from 44,600 to 60,500 to
90,100 tokens per second for the 4B on an H100, and from 4,800 to 6,000 to
9,800 on an L4. The fused kernels pay where the GPU is the bottleneck
(rerank), graphs where the host is (one message), and the shared prefix on
both. For the 4B on an L4 a one-message request cannot go below about 30 ms:
one forward reads the model's 8 GB of weights.

**Why `fast` is a profile and not the default.** A shared prefix and a graph
change how many rows a matrix multiply sees, so scores move by floating-point
rounding. On the same requests, against the default profile:

| Model | GPU | Rerank: questions of 499 with the same top candidate (largest score change) | One message: largest score change |
|--|--|--|--|
| `Qwen3-Reranker-4B` | H100 | 499, every score bit-equal | 0.036 |
| `Qwen3-Reranker-0.6B` | H100 | 499, every score bit-equal | 0.039 |
| `Qwen3-Reranker-4B` | RTX PRO 6000 | 498 (0.064); the other had a margin of 0.00009 | 0.041 |
| `Qwen3-Reranker-4B` | L4 | 60 of 60 (0.074) | 0.055 |
| `Qwen3-Reranker-0.6B` | L4 | 196 of 200 (0.120); margins up to 0.0036 | 0.066 |

The default profile moves too. Scoring a pair alone or among its question's
20 candidates changes its score by up to 0.019 for the 4B on an H100
(0.000002 for the 0.6B), and by up to 0.097 and 0.144 on an L4, where the
0.6B's top candidate changes in 3 of 60 questions. Against a float32 forward
of each checkpoint the default profile is off by up to 0.167 on rerank and
0.083 on one-message requests, 0.0005 to 0.0044 on average, and the `fast`
options by about as much: an average within a tenth of the default profile's
on rerank and the decision questions on every GPU (up to half again as large
on longer one-message requests on an L4), and a largest difference that is
larger on some request sets and smaller on others. So `fast` is as close
to the model as the default profile is, without meeting the rule above ("When
a speed-up ships enabled") on every request set and GPU: some scores move by
more than 0.02, and the comparison against float32 goes either way. It ships
as a profile an operator chooses.

## Configuration

`sie-server` reads its config from `SIE_*` environment variables (Pydantic
`BaseSettings`). Common knobs:

### Authentication and exposure

| Env var | Default | Effect |
|--|--|--|
| `SIE_SERVER_HOST` | `0.0.0.0` | Address `sie-server serve` binds, same as `--host`. `127.0.0.1` accepts connections from this machine only. The default becomes `127.0.0.1` in 0.13.0. |
| `SIE_AUTH_MODE` | unset | `token` (alias `static`) requires a token on every route of the HTTP port except `/healthz`, `/livez` and `/readyz`, including `/ws/status`. `none` turns authentication off. `auto` requires a token when the bind address is not loopback; it is a `sie-server` value, which the gateway refuses. Unset behaves as `none`, or as `token` when a token is configured. Values are lowercase and case-sensitive, as on the gateway; any other value stops startup. |
| `SIE_AUTH_TOKENS` | unset | Comma-separated accepted tokens. `SIE_AUTH_TOKEN` is read when it is unset. Setting a token with `SIE_AUTH_MODE=none` stops startup, and so does a token variable that is set but holds no token, unless `SIE_AUTH_MODE=none`. |
| `SIE_DATA_DIR` | `$XDG_DATA_HOME/sie`, else `~/.local/share/sie` | Holds the generated token (`auth-token`, mode `0600`) and the telemetry worker id. Images set `/app/.cache/sie`. |

Clients send the token as `Authorization: Bearer <token>` or `x-api-key: <token>`;
when both carry a token, `Authorization` is used and `x-api-key` is ignored, as on
the gateway. The SDKs send `SIE_API_KEY`
as a Bearer token. Repeated headers differ from the gateway until 0.13.0: the
server reads every value of a repeated header, skips empty ones, and refuses
values that carry different tokens, while the gateway reads only the first
`Authorization` value and uses `x-api-key` when that value is empty or blank.

Token auth covers the HTTP port. The other listeners are local: the IPC socket
that a worker sidecar uses (`SIE_IPC_SOCKET_PATH`, mode `0600`) and the
loopback ports of the engines the server launches. On macOS a socket can
inherit an entry that allows access from its directory's access control list,
and its mode does not remove that access, so the server refuses to create the
IPC socket in a directory whose access control list has an entry that allows
access (`ls -led <directory>` lists the entries).

When token auth is on and no token is configured, the server generates one on
first start and saves it in `SIE_DATA_DIR`. When stderr is a terminal it prints
the token once; otherwise, as in a container without a terminal or under a
service manager whose output is kept in logs, it prints only the file's path
and the command that reads it. Later starts reuse the token and print only its
path. They only read it, so a read-only directory or volume works once it holds
the token. A container started with a terminal (`docker run -t`, Compose
`tty: true`, or a Kubernetes container with `tty: true`) counts as a terminal,
and the container runtime keeps that one print in the container's logs. In
containers, set `SIE_AUTH_TOKEN` yourself instead. The server refuses a token
directory that another user owns or can write to, and a token file that is a
symbolic link, not a regular file, hard-linked, owned by another user, or
readable by others. On macOS it also refuses a token directory or file whose
access control list has an entry that allows access (`ls -led <path>` lists
the entries); deny entries, such as the one macOS puts on the home folder, are
accepted. To rotate a generated token, delete `auth-token` from
`SIE_DATA_DIR` and restart the server. In Docker, keep the token across
containers with a volume:

```bash
docker run -p 127.0.0.1:8080:8080 -e SIE_AUTH_MODE=token \
  -v sie-data:/app/.cache/sie \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cpu-default
```

A server that listens beyond loopback without authentication logs a warning on
every start. From 0.13.0 such a server requires a token unless
`SIE_AUTH_MODE=none` is set explicitly.

A server bound to loopback (`127.0.0.0/8`, `::1` or `localhost`) with
authentication off checks that the `Host` header names this machine:
`localhost`, `127.0.0.1`, `[::1]` or the bind address, with any port. A web
page that rebinds its own domain to `127.0.0.1` sends another `Host`. Until
0.13.0 such a request is still answered, and the server logs a warning once for
each such `Host`. From 0.13.0 it gets `400`. That also affects a reverse proxy
on the same machine that passes the client's `Host` on, and a container that
reaches the server as `host.docker.internal`: have them send one of the names
above, or turn on token authentication.

### Memory & OOM resilience

SGLang generation, embedding and vision-extraction adapters size memory per
model when `adapter_options.loadtime.mem_fraction_static` is omitted. SIE reads
the pinned local safetensors headers at the serving precision and the model's
attention geometry. It reserves the weights (with a 5% allowance), the KV pool,
1 GiB for startup and 2 GiB for runtime work such as CUDA graphs. No weights are
loaded onto a GPU to compute this estimate.

The KV pool uses the resolved profile's `kv_budget_tokens`; embedding and
extraction profiles without that field use `max_seq_length`. Capacity rounds
up to 512 tokens, with another 512 tokens accounted for as allocation padding.
SIE passes `--max-total-tokens` to bound the pool. SGLang's static fraction is
derived from free memory immediately before launch, so loading a second model
does not reduce its requested capacity. The load-staging check uses the same
reservation and evicts idle eligible models when it cannot fit. Resident-model
accounting reports the reservation; the pressure monitor continues to use
actual device usage, including every SGLang subprocess.

To dedicate most of a GPU to one model, explicitly set a fraction:

```yaml
profiles:
  default:
    # Other required profile fields omitted here.
    kv_budget_tokens: 32768
    adapter_options:
      loadtime:
        mem_fraction_static: 0.90
```

An explicit fraction keeps the previous SGLang launch and admission behavior:
SIE requires that fraction of total GPU memory plus 1 GiB of load headroom,
and does not cap the engine's KV pool to `kv_budget_tokens`. Explicit fractions
are required for tensor-parallel engines, speculative or LoRA weights,
quantized checkpoints and nonstandard KV layouts such as MLA or linear
attention. Calibrated profiles for these configurations retain their overrides.
Automatic sizing is an estimate; unusually large runtime workspaces can still
invoke the existing OOM recovery path.

| Env var | Default | Effect |
|--|--|--|
| `SIE_MEMORY_PRESSURE_THRESHOLD_PERCENT` | `95` | VRAM utilisation that triggers reactive LRU eviction by the pressure monitor. |
| `SIE_OOM_RECOVERY__ENABLED` | `true` | Master switch for reactive OOM recovery in the worker dispatch path (`cache_clear → evict_lru → split_batch`). |
| `SIE_OOM_RECOVERY__STRATEGY` | `cache_clear,evict_lru,split_batch` | Ordered recovery actions. Earlier actions tried first. |
| `SIE_OOM_RECOVERY__MAX_SPLIT_DEPTH` | `4` | Cap on recursive batch halving (≤16 sub-batches). |
| `SIE_OOM_RECOVERY__EVICTION_LOCK_TIMEOUT_S` | `5.0` | Soft timeout when waiting for the registry's load-lock during recovery eviction. |
| `SIE_OOM_RECOVERY__RETRY_AFTER_S` | `5` | `Retry-After` header value on `RESOURCE_EXHAUSTED` responses. |
| `SIE_DISABLE_OOM_RECOVERY` | unset | Convenience kill switch (`1`/`true`/`yes`) for incident triage. Wins over `SIE_OOM_RECOVERY__ENABLED=true`. |
| `SIE_IDLE_EVICT_S` | unset (disabled) | Unload models that have been idle longer than this (seconds). Additive to the pressure monitor; helps free cold weights before pressure builds. |
| `SIE_OOM_NAK_DELAY_S` | `10.0` | Queue-mode only. NAK delay (seconds) for `RESOURCE_EXHAUSTED` work items so JetStream redelivers them after memory pressure has had a chance to clear. |

When OOM recovery is exhausted on a request, the server returns
`HTTP 503 RESOURCE_EXHAUSTED` with `Retry-After`. The Python SDK
auto-retries; see `packages/sie_sdk/README.md` for client-side controls.

### Batching & request handling

| Env var | Default | Effect |
|--|--|--|
| `SIE_MAX_BATCH_REQUESTS` | `64` | Maximum number of items per batched inference call. |
| `SIE_MAX_BATCH_WAIT_MS` | `15.0` | Initial value for the adaptive first-request batch timeout. At runtime the PI batching controller steers it between `SIE_ADAPTIVE_BATCHING__MIN_WAIT_MS` and `SIE_ADAPTIVE_BATCHING__MAX_WAIT_MS`, so it is a starting point rather than a fixed wait. |
| `SIE_MAX_CONCURRENT_REQUESTS` | `512` | Per-worker queue size; admission control returns `QUEUE_FULL` above this. |
| `SIE_MAX_LORAS_PER_MODEL` | `10` | Maximum concurrent LoRA adapters per base model. A request loads or selects only the adapters its selected profile declares. `options.lora` or `options.lora_id` naming any other is rejected with HTTP 400: `INVALID_INPUT` from this server, and `INVALID_REQUEST` from the gateway (code `unknown_lora_adapter` where it answers in the OpenAI error envelope). |
| `SIE_MAX_ITEM_TEXT_BYTES` | `2097152` (2 MiB) | Most bytes of UTF-8 one encode, score, or extract item may carry in its `text` and `metadata` together. A larger item is rejected with `INVALID_INPUT` (HTTP 400) before it is tokenized. Generation prompts are not items and are not affected. |

SGLang generation and embedding profiles can defer declared LoRAs until a
request selects them. Keep one source allowlist in `lora_paths` and use
`preload_loras` to select startup adapters:

```yaml
adapter_options:
  loadtime:
    mem_fraction_static: 0.80
    lora_paths:
      support:
        id: acme/support-lora
        revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa # replace with the repository's immutable commit
      legal: acme/legal-lora
    preload_loras: []
    max_lora_rank: 16
    lora_target_modules: [q_proj, v_proj]
    max_loras_per_batch: 2
```

Omitting `preload_loras` keeps the existing behavior: every declared source
loads at startup and remains pinned in SIE. An empty list defers every source;
a nonempty list preloads and pins only those served names. Names must be unique
members of the selected profile's map. Profile `loadtime` blocks use full
replacement when a child supplies one, so restate the source map and its other
options when changing only a child's preload selection.

Deferred adapters require the maximum rank and target modules at engine launch
because SGLang allocates its LoRA buffers then. LoRA profiles require an explicit
memory fraction. For profiles with deferred adapters, the resolved
`max_loras_per_model` engine setting (normally `SIE_MAX_LORAS_PER_MODEL`)
bounds resident adapters, including pinned preloads
and in-flight loads; `max_loras_per_batch` and the preload count cannot exceed
it. When full, SIE drains and evicts the least-recently-used unpinned adapter. An
adapter being loaded or drained retains its slot, and all-pinned/full capacity
returns a retry response. Unconfirmed attach rollback or unload retains the
slot and requires a model reload rather than risking unbounded residency.
Startup-only profiles retain every declared adapter: their effective capacity
is at least the pinned source count and `max_loras_per_batch`, even when the
configured runtime limit is lower.

Generation and chat requests select a selected-profile declared name with
`lora_adapter` (a served alias or its declared source id);
embedding requests use `options.lora` (or a declared source `options.lora_id`).
The first request for a cold adapter returns HTTP 503 `LORA_LOADING` with
`Retry-After`; retry after loading finishes. Source files are downloaded outside
the inference mutation lock, while attachment may briefly pause SGLang's
scheduler. Requests cannot supply a new repository or override its configured
revision. Runtime adapters must contain ordinary LoRA A/B safetensors, without
base-weight transforms, saved base modules, variable rank/alpha patterns,
rsLoRA, or DoRA.

Generation streams keep independent adapter leases and can run concurrently
through SGLang's continuous batcher. Eviction refuses new leases and waits for
active streams to finish. Embedding dispatch remains serialized by selected
LoRA; this change does not introduce a mixed-adapter embedding forward. Deferred
LoRA profiles require one tokenizer worker and one data-parallel worker;
LoRA generation on those profiles supports `n=1` and `best_of=1` because pinned
upstream versions have unsafe multi-candidate unload accounting. SGLang
0.5.10.post1 uses SIE's scoped GPU-slot release compatibility hook; 0.5.20
already implements that cleanup natively.

The single-server `/v1/models` response includes top-level `lora_adapters` with
ready served names, excluding adapters being loaded or evicted. Gateways omit
this worker-local snapshot. Their `capabilities.lora_adapters` continues to
advertise the declared allowlist, including cold adapters.

### Compute & precision

| Env var | Default | Effect |
|--|--|--|
| `SIE_DEFAULT_COMPUTE_PRECISION` | `float16` | One of `float16`, `bfloat16`, `float32`. |
| `SIE_ATTENTION_BACKEND` | `auto` | One of `auto`, `flash_attention_2`, `sdpa`, `eager`. |
| `SIE_DISABLE_CUDNN_SDP` | unset (off) | Set `1` to disable cuDNN for Torch scaled dot-product attention at Python worker startup. Unset or `0` leaves the current backend policy alone. |

This startup policy applies throughout the Python worker process, across all
models that use Torch SDPA. Other eligible SDPA backends remain available; it
does not guarantee Flash attention or better performance and does not disable
cuDNN generally. The policy is applied before model loading and readiness and
is not restored at shutdown. Start a fresh worker process with the variable
unset or `0` to restore Torch's defaults; unset or `0` does not reset a backend
already changed in the same interpreter.

### Diagnostics

| Env var | Default | Effect |
|--|--|--|
| `SIE_GRAMMAR_PREFLIGHT_DEBUG` | unset (off) | Enables the legacy worker-side Outlines preflight compile before each structured-output request. Off by default because SGLang is the production grammar authority. Use for diagnosing schema-rejection problems or slow compiles in a controlled environment; not recommended for production traffic. |

For nested settings (any field with `__`), the env-var format is
`SIE_<TOP>__<NESTED>=value`. The complete schema is in
`packages/sie_server/src/sie_server/config/engine.py`.

## Observability

The server emits the checked-in worker telemetry contract once through
OpenTelemetry/OTLP. The regional collector owns Prometheus exposition and
optional remote OTLP routing; the application does not expose `/metrics` or
maintain a second Prometheus registry. See `telemetry/contract.yaml` for exact
instrument names, dimensions, ownership, and histogram bounds.

## API

See the [API documentation](https://sie.dev/docs) for details.

## License

Apache 2.0
