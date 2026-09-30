# SIE Server

GPU inference server for embeddings, reranking, and entity extraction.

## Features

- Multi-model serving with LRU eviction
- Token-based dynamic batching
- Hot reload model configs without restart
- Unified API: `encode()`, `score()`, `extract()`
- Prometheus metrics and OpenTelemetry tracing
- Reactive GPU OOM recovery + proactive idle eviction

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

- **Transformers 5 bundle** (LightOnOCR, GLM-OCR, GLiGuard, the GLiNER2.5-Decide models, and the TopK-Embed-V1
  multi-vector models) — requires `transformers` 5.x, and is served with `-b transformers5`. The GLiNER2.5-Decide
  models also need `gliner2` 2.x. `sie-server` itself asks for `gliner2<2`, which the default bundle's GLiNER2
  models need, so pip reports that conflict when the second command below installs 2.x; the transformers5
  bundle's GLiNER2 models are verified on 2.0.0:

  ```bash
  pip install sie-server "transformers>=5,<6"
  pip install "gliner2==2.0.0"  # only for the GLiNER2.5-Decide models
  sie-server serve -b transformers5
  ```

## Quick Start

```bash
sie-server serve --port 8080 --device cuda:0
```

### TensorRT-LLM generation

TensorRT-LLM buffers completion token IDs and emits the final text together only
after generation finishes and the terminal stream is verified, even with
`stream=True`. This preserves context-sensitive spacing and punctuation and
removes matched stop sequences consistently from returned text, completion-token
usage, and logprobs. Long completions delay the first visible text; existing
request timeouts still apply. Other adapters are unaffected.

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
| `bucketed` | sequence lengths padded up to a multiple of 64 tokens at a 1,024-token window (32 at 512), batch sizes up to a power of two: a fixed set of shapes, all kept | padding moves fp16 probabilities (see below) |

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

Requests of several items pad their batch as well (3 items to 4, 5 to 8). Asked
in 3- and 5-item requests and compared with eager execution of the same
requests, the eight models that ship with graphs moved no probability by more
than 0.0190 (`opir-multitask-large-v1.0`) and 0.0115 for every other model,
and every top label that changed had an eager margin below the model's
δ_eager.

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
33 for the DeBERTa-v3-large models and 52 for the base and small ones (52 and
71 at a 512-token window), and the model keeps a graph for every one. Once
they are recorded, every forward under the token bound replays a graph,
whatever mix of label counts, batch sizes and lengths the traffic has, and no
request's shapes push out another's.

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
1,024-token window, all of a model's `bucketed` shapes took 551 to 571 MB for
the large GLiClass models, 677 MB for `opir-multitask-large-v1.0`, 761 to
767 MB for the base models and 653 MB for `gliclass-small-v1.0`, plus about
150 MB cached on the recording stream (its cuBLAS workspace and one warm-up
row). Recorded while serving mixed traffic, the same shapes can take more, and
at the 1,024-token window the budget can fill with one to three shapes left
unrecorded; those shapes run eagerly. On a smaller GPU, where the shapes do not
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

## Configuration

`sie-server` reads its config from `SIE_*` environment variables (Pydantic
`BaseSettings`). Common knobs:

### Memory & OOM resilience

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
| `SIE_MAX_LORAS_PER_MODEL` | `10` | Maximum concurrent LoRA adapters per base model. |
| `SIE_MAX_ITEM_TEXT_BYTES` | `2097152` (2 MiB) | Most bytes of UTF-8 one encode, score, or extract item may carry in its `text` and `metadata` together. A larger item is rejected with `INVALID_INPUT` (HTTP 400) before it is tokenized. Generation prompts are not items and are not affected. |

### Compute & precision

| Env var | Default | Effect |
|--|--|--|
| `SIE_DEFAULT_COMPUTE_PRECISION` | `float16` | One of `float16`, `bfloat16`, `float32`. |
| `SIE_ATTENTION_BACKEND` | `auto` | One of `auto`, `flash_attention_2`, `sdpa`, `eager`. |

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
