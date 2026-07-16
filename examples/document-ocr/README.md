# document-ocr

**Swap an OCR model with one identifier change. See what changes in the output.**

Three different model architectures, one inference server, one SDK call. This
demo is a working browser UI that runs document images through a recognition
model, a fine-tuned document model, and a zero-shot NER, all behind the same
`client.extract(...)` API. Pick a model from any of the three dropdowns; watch
the pipeline run again with that one identifier swapped.

<!-- Drop screenshots or a GIF here. Suggested:
     docs/cover.png  (the browser UI mid-run)
     docs/run.gif    (a sample to typed fields end to end)        -->

![document-ocr](docs/cover.png)

> Built on [SIE](https://github.com/superlinked/sie), the open-source inference
> engine from Superlinked. Apache 2.0. Run it locally with `docker compose up`;
> a hosted Hugging Face Space is coming soon.

---

## What this demo actually shows

OCR is almost never a single-model problem. A real pipeline has three concerns:

1. **Recognition** (image to text). Modern VLM-OCRs like Florence-2 and
   LightOnOCR take a whole document and emit Markdown.
2. **Structured extraction** (image to JSON). End-to-end document models like
   Donut on CORD skip the text intermediate entirely and emit nested JSON
   directly: `{ "total": { "total_price": "28.52" }, ... }`.
3. **Zero-shot NER on the recognized text** (text to typed fields). When you
   want to declare entity labels at query time (e.g. `["merchant", "total",
   "date"]`) instead of fine-tuning a new model.

The SIE pitch this demo makes visceral: **all three are the same SDK call,
the same auth, the same rate-limit budget, the same Docker image.** Only the
model ID changes.

```python
# Recognition: VLM-OCR, returns Markdown
client.extract(
    "lightonai/LightOnOCR-2-1B",
    Item(images=[image_bytes]),
)

# Structured: end-to-end Donut, returns JSON tree
client.extract(
    "naver-clova-ix/donut-base-finetuned-cord-v2",
    Item(images=[image_bytes]),
)

# NER: zero-shot, returns typed entities
client.extract(
    "urchade/gliner_multi-v2.1",
    Item(text=recognized_markdown),
    labels=["merchant", "total", "date", "line_item"],
)
```

Each cell in the demo's UI has a **"See the SIE call"** disclosure that shows
the exact line of code that just ran. Swap a dropdown, the snippet updates with
the one parameter that changed.

---

## Run it

There are two ways to try this demo:

- **Hosted on Hugging Face Spaces** (zero install, just click):
  [superlinked/document-ocr](https://huggingface.co/spaces/superlinked/document-ocr)
- **Local Docker** (steps below):

```bash
git clone https://github.com/superlinked/sie
cd sie/examples/document-ocr
npm install
npm start
```

`npm start` runs `docker compose up -d` (boots `ghcr.io/superlinked/sie-server:latest-cpu-transformers5`,
preloads three small models), then starts a Node UI server and opens
http://localhost:3032.

- **First start**: ~5-8 minutes on a typical home connection (~5 GB of
  LightOnOCR + Donut + GLiNER weights download from Hugging Face into a
  Docker volume; LightOnOCR is the big one at ~4 GB).
- **Subsequent restarts**: 30-60 seconds. Weights are cached in
  `/app/.cache/huggingface` inside the `sie-cache` Docker volume.
- **Per-sample click**: ~20-30 s on Linux x86_64 native, ~60-90 s on Apple
  Silicon (Rosetta tax on the amd64-only SIE image).

```bash
docker compose down   # when done
```

**GPU variant** (Linux + NVIDIA, makes GLM-OCR / Paddle / LightOnOCR fast):

```bash
npm run start:gpu     # uses compose.gpu.yml + latest-cuda12-default
```

---

## Specific things to try in the UI

The UI itself surfaces these as a "Try these moments" strip above the panels,
so a visitor sees the prompts even without reading the README. The same four
moments, with a little more context:

1. **Click any sample → open the "See the SIE call" disclosure under
   "Recognition".** You'll see the actual `client.extract(...)` call with
   the LightOnOCR-2-1B model ID. The disclosure updates when any dropdown
   changes; only the model string changes between calls.
2. **Switch the Structured dropdown** from `donut-cord-v2` to
   `donut-rvlcdip`. Same Donut architecture, different fine-tuning. The
   output shape changes from a CORD-shaped JSON tree to a document-type
   classification. Same model class, same SDK call, completely different
   output category.
3. **Switch the NER dropdown** from `gliner_multi-v2.1` to `NuNER_Zero`.
   Different model family entirely (NuMind vs urchade), same SDK call, same
   labels. The architecture-swap-with-one-string-change pitch.
4. **Compare two samples back to back**: click `receipt.png` (Donut on CORD
   dominates because that's exactly its fine-tuning distribution), then
   click `letter.png` (the opposite: Donut produces garbage shaped like
   CORD; recognition + GLiNER carry the pipeline).

Each of these illustrates a concrete SIE pitch: model swap, output-shape
swap, quality swap, document-type fit.

---

## Why SIE specifically for OCR

You could build this demo with three SaaS APIs (one embedding/OCR provider,
one document AI provider, one NER provider). It would work. It would also be:

- **Three auth flows.** Three API keys, three rate-limit budgets, three
  outage stories.
- **Three SDKs.** Each with its own retry semantics, error model, input
  encoding.
- **Three deployment stories.** When you eventually want to run this in your
  own VPC (because, for instance, you can't send customer PDFs to a third
  party), you have three Helm charts, three sets of secrets to rotate.

SIE collapses that into one process:

- **One server, three primitives.** `encode`, `score`, `extract`. This demo
  uses `extract` for all three model classes; the other two cover semantic
  search and reranking.
- **One SDK call.** `client.extract(model_id, item)` works for VLM-OCR,
  end-to-end document AI, and zero-shot NER. Swap the model ID alone.
- **Open source, runs in your VPC.** Customer documents never leave the
  host running this compose. Compliance teams stop blocking you.
- **A full catalog of models, swappable.** Need a different OCR-VLM, a domain-tuned
  GLiNER, a layout model, a multilingual reranker, or a custom LoRA?
  All live in the same catalog. Swap a model ID in `src/config.ts`.
- **Same code laptop to Kubernetes.** SIE ships a Helm chart, KEDA
  autoscaling, and Terraform modules for GKE/EKS. The code in this demo
  runs unchanged against a production cluster; only the URL changes.

---

## Architecture in one diagram

```
                   ┌───────────────────────────┐
                   │  one document image       │
                   └─────────────┬─────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │   one SIE server            │
                  │   client.extract(model_id)  │
                  └──┬───────────┬───────────┬──┘
                     │           │           │
              ┌──────▼──┐  ┌─────▼─────┐  ┌──▼──────┐
              │  VLM-   │  │  Donut    │  │ GLiNER  │
              │  OCR    │  │  (E2E)    │  │  (NER)  │
              │ Markdown│  │   JSON    │  │ entities│
              └─────────┘  └───────────┘  └─────────┘
```

The Node server in `web/server.ts` chains the three calls and streams progress
via Server-Sent Events. The browser renders each panel as the corresponding
event lands.

---

## Model lineup

The dropdowns expose nine models across three categories. Models marked
"alternate" are listed in the UI but lazy-load on first click; the demo
auto-disables anything not in the running server's `/v1/models` catalog (so
GPU-only models show up disabled on the CPU compose).

| Stage | Default (preloaded) | Alternates |
|---|---|---|
| Recognition | `lightonai/LightOnOCR-2-1B` (2.1B, Markdown output) | `PaddlePaddle/PaddleOCR-VL-1.5` (GPU image); `zai-org/GLM-OCR` (9B, GPU only) |
| Structured | `naver-clova-ix/donut-base-finetuned-cord-v2` | `naver-clova-ix/donut-base-finetuned-docvqa`; `naver-clova-ix/donut-base-finetuned-rvlcdip` (16-class document classifier) |
| Zero-shot NER | `urchade/gliner_multi-v2.1` | `urchade/gliner_large-v2.1`; `urchade/gliner_multi_pii-v1`; `numind/NuNER_Zero` (different architecture, useful contrast against GLiNER) |

Defaults are pinned in [`src/config.ts`](src/config.ts). To add a new model,
add a `ModelOption` entry and a HuggingFace model ID; SIE handles the rest.
The demo runs on the `latest-cpu-transformers5` SIE image; this is the
bundle where `lighton_ocr` / `glm_ocr` / `paddle` adapters live. The
Florence-2 family ships in SIE's `default` bundle (which pins
`transformers<5`) and is not loadable from this image. See
[sie-internal#828](https://github.com/superlinked/sie-internal/issues/828)
for the bundle-composition story.

---

## What's in the box

```
src/
  config.ts          model lineup (defaults + alternates per category)
  types.ts           SampleDoc, ExtractedField, TriageResult
  events.ts          typed SSE events streamed to the browser
  ocr.ts             VLM-OCR caller (Florence-2, LightOnOCR, etc.)
  donut.ts           image-to-JSON caller (Donut variants)
  extract.ts         GLiNER caller for zero-shot NER on text
  pipeline.ts        recognition → structured + NER orchestrator

data/samples/        6 synthetic document images (receipt, invoice, business
                     card, event poster, presentation slide, business letter)
                     + index.json metadata (labels per document type)
scripts/
  generate_samples.py    regenerates the bundled images via Pillow

web/
  server.ts          Node http server, /api/run SSE endpoint, static assets
  public/            index.html, style.css, app.js (vanilla, no build step)

compose.yml          CPU compose (local docker compose up)
compose.gpu.yml      CUDA compose with NVIDIA device reservations
```

**~1,400 lines total.** No bundler, no React, no build step. The UI is vanilla
HTML + CSS + JavaScript driven by `EventSource` for the SSE stream from SIE.

---

## Extend it

- **Add a layout step.** Some VLM-OCR models return per-region text and
  bounding boxes (e.g. PaddleOCR-VL with `task: "layout"`). Render them as
  overlays on the source image to demonstrate layout-aware OCR.
- **Switch to multi-page PDFs.** Replace `Item.images` with
  `Item.document` and use SIE's Docling adapter. Docling parses entire PDFs
  with layout + table-structure detection in one call.
- **Embed the recognized text.** Pair `extract` with `client.encode` to
  build a semantic search index over your processed documents. The SDK
  call is the same; the model ID picks an embedding model from SIE's catalog of 100+
  catalog.
- **Fine-tune for your schema.** If your documents look like one of the
  Donut training corpora (CORD, RVL-CDIP, DocVQA), pick the matching
  variant and skip the GLiNER step entirely.
- **Add a reranker after retrieval.** `client.score(reranker_id, query,
  documents)` does cross-encoder reranking; many real-world OCR pipelines
  need a "did we extract the right field" verification pass.

---

## Honest scope and known limits

- **Apple Silicon performance is poor** because the SIE Docker image is
  `linux/amd64` only and runs through Rosetta. Per-call latency on a Mac
  is ~60-90 s for LightOnOCR; on Linux x86_64 it is ~20 s.
- **GLM-OCR and Paddle are GPU-only in practice.** They need CUDA, so the
  CPU compose lists them but auto-disables them in the dropdown. The CUDA
  image variant (used by `compose.gpu.yml`) loads them.
- **This is a demo, not a production OCR pipeline.** The bundled sample
  images are synthetic; production OCR needs real-world layout coverage,
  per-merchant tuning, and human review hooks.

---

## Built with

- [SIE](https://github.com/superlinked/sie) (Apache 2.0): the inference
  engine that hosts all three model classes
- [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B)
  (Apache 2.0): LightOn's Pixtral+Qwen3 OCR-VLM
- [Donut](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2)
  (MIT): NAVER Clova's end-to-end document understanding model
- [GLiNER](https://huggingface.co/urchade/gliner_multi-v2.1) (Apache 2.0):
  Urchade's zero-shot NER
- Sample images: programmatically generated with Pillow; no real customer
  data

Star [superlinked/sie](https://github.com/superlinked/sie) if this was useful.
