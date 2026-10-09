# document-ocr

**Swap an OCR model with one identifier change. See what changes in the output.**

Three different model architectures, one SDK call. This demo is a working
browser UI that runs document images through a recognition model, a
fine-tuned document model, and a zero-shot NER, all behind the same
`client.extract(...)` API. Pick a model from any of the three dropdowns; watch
the pipeline run again with that one identifier swapped. Recognition and the
Donut/GLiNER stages do not share an image: their adapters live in different
bundles, so the two compose files start different SIE images.

<!-- Drop screenshots or a GIF here. Suggested:
     docs/cover.png  (the browser UI mid-run)
     docs/run.gif    (a sample to typed fields end to end)        -->

![document-ocr](docs/cover.png)

> Built on [SIE](https://github.com/superlinked/sie), the open-source inference
> engine from Superlinked. Apache 2.0. Run it locally with `docker compose up`,
> or open the hosted Space linked below.

---

## What this demo actually shows

OCR is almost never a single-model problem. A real pipeline has three concerns:

1. **Recognition** (image to text). VLM-OCRs such as LightOnOCR, GLM-OCR, and
   PaddleOCR-VL take a whole document and emit Markdown. They load on the GPU
   image only.
2. **Structured extraction** (image to JSON). End-to-end document models like
   Donut on CORD skip the text intermediate entirely and emit nested JSON
   directly: `{ "total": { "total_price": "28.52" }, ... }`.
3. **Zero-shot NER on the recognized text** (text to typed fields). When you
   want to declare entity labels at query time (e.g. `["merchant", "total",
   "date"]`) instead of fine-tuning a new model.

The SIE pitch this demo makes visceral: **all three are the same SDK call,
the same auth, the same rate-limit budget.** Only the model ID changes.
One image does not serve Donut and LightOnOCR together. Donut and GLiNER
load on the CPU `default` bundle; LightOnOCR, GLM-OCR, and PaddleOCR-VL load
on the GPU `sglang-vision-extract` bundle.

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

The first call needs the GPU compose (`latest-cuda12-sglang-vision-extract`).
The Donut and GLiNER calls need the CPU compose (`latest-cpu-default`).

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

`npm start` runs `docker compose up -d` (boots `ghcr.io/superlinked/sie-server:latest-cpu-default`,
preloads Donut on CORD, Donut on DocVQA, and `urchade/gliner_multi-v2.1`), then starts a Node UI
server and opens http://localhost:3032. It does not preload LightOnOCR, GLM-OCR, or PaddleOCR-VL.

- **First start**: the image downloads the Donut and GLiNER weights from Hugging Face into a Docker volume.
- **Subsequent restarts**: weights are cached in `/app/.cache/huggingface` inside the `sie-cache` Docker volume.
- **Apple Silicon**: the image is `linux/amd64` and runs through Rosetta, so calls are slower than on native x86_64.

```bash
docker compose down   # when done
```

**GPU variant** (Linux + NVIDIA). Preloads LightOnOCR-2-1B, GLM-OCR, and PaddleOCR-VL-1.5. This image does not serve Donut or GLiNER.

```bash
npm run start:gpu     # uses compose.gpu.yml + latest-cuda12-sglang-vision-extract
```

---

## Specific things to try in the UI

The UI itself surfaces these as a "Try these moments" strip above the panels,
so a visitor sees the prompts even without reading the README. The same four
moments, with a little more context:

1. **On the CPU compose, click any sample → open the "See the SIE call"
   disclosure under "Structured".** You'll see `client.extract(...)` with
   the Donut CORD model id. Recognition stays disabled: those models are
   not on this image.
2. **Switch the Structured dropdown** from `donut-cord-v2` to
   `donut-docvqa`. Same Donut architecture, different fine-tuning. The
   output shape changes from a CORD-shaped JSON tree to a DocVQA answer.
   Same model class, same SDK call, different output.
3. **Switch the NER dropdown** from `gliner_multi-v2.1` to `NuNER_Zero`
   on the CPU image. Different model family (NuMind vs urchade), same SDK
   call, same labels. NER reads recognition Markdown, and this image has
   no recognition model, so the NER panel stays empty until you point the
   UI at text from the GPU image.
4. **On the GPU compose, open the Recognition disclosure.** The call uses
   `lightonai/LightOnOCR-2-1B` unless you switch to GLM-OCR or
   PaddleOCR-VL-1.5. Donut and GLiNER are not registered on that image.
   Compare `receipt.png` with `letter.png` on the CPU image instead: Donut
   on CORD fits the receipt and emits a CORD-shaped tree for the letter too.

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
                  │   client.extract(model_id)  │
                  │   (two images, one call)    │
                  └──┬───────────┬───────────┬──┘
                     │           │           │
              ┌──────▼──┐  ┌─────▼─────┐  ┌──▼──────┐
              │  VLM-   │  │  Donut    │  │ GLiNER  │
              │  OCR    │  │  (E2E)    │  │  (NER)  │
              │ Markdown│  │   JSON    │  │ entities│
              └─────────┘  └───────────┘  └─────────┘
```

The Node server in `web/server.ts` chains the calls the running image can
serve and streams progress via Server-Sent Events. The browser renders each
panel as the corresponding event lands. VLM-OCR is the GPU image; Donut and
GLiNER are the CPU image.

---

## Model lineup

The dropdowns expose nine models across three categories. The UI disables
anything the running server's `/v1/models` catalog does not list, and it
disables recognition models on the CPU compose (`gpuRequired`). No single
image serves both columns.

| Stage | CPU compose `latest-cpu-default` | GPU compose `latest-cuda12-sglang-vision-extract` |
|---|---|---|
| Recognition | not served (dropdown disabled) | preloaded: `lightonai/LightOnOCR-2-1B`, `zai-org/GLM-OCR`, `PaddlePaddle/PaddleOCR-VL-1.5` |
| Structured | preloaded: `naver-clova-ix/donut-base-finetuned-cord-v2` (default) and `naver-clova-ix/donut-base-finetuned-docvqa` | not served |
| Zero-shot NER | preloaded: `urchade/gliner_multi-v2.1`. Lazy-loaded alternates: `urchade/gliner_large-v2.1`, `urchade/gliner_multi_pii-v1`, `numind/NuNER_Zero` | not served |

Defaults for the CPU image are pinned in [`src/config.ts`](src/config.ts):
Donut on CORD and `urchade/gliner_multi-v2.1`. Recognition defaults are
marked `gpuRequired` so the CPU UI does not treat them as local. To add a
model, add a `ModelOption` whose default adapter is in the bundle of the
compose that should serve it.

---

## What's in the box

```
src/
  config.ts          model lineup (defaults + alternates per category)
  types.ts           SampleDoc, ExtractedField, TriageResult
  events.ts          typed SSE events streamed to the browser
  ocr.ts             VLM-OCR caller (LightOnOCR, GLM-OCR, PaddleOCR-VL)
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
  Donut checkpoints this demo serves (CORD, DocVQA), pick the matching
  variant on the CPU image and skip the GLiNER step entirely.
- **Add a reranker after retrieval.** `client.score(reranker_id, query,
  documents)` does cross-encoder reranking; many real-world OCR pipelines
  need a "did we extract the right field" verification pass.

---

## Honest scope and known limits

- **Apple Silicon runs the CPU image through Rosetta.** That image is
  `linux/amd64` only, so Donut and GLiNER are slower than on native x86_64.
- **LightOnOCR, GLM-OCR, and PaddleOCR-VL-1.5 are on the GPU image only.**
  Their default profile is `sglang_vision_extract`. The CPU compose lists
  them and disables them. `compose.gpu.yml` uses
  `latest-cuda12-sglang-vision-extract` and does not load Donut or GLiNER.
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
