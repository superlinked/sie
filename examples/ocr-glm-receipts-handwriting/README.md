# Read receipt photos and handwriting with an open model

Z.ai GLM-OCR takes a photo and returns text. Run it on your own SIE server so your images go to the endpoint you operate. The recorded comparison below tests recovered annotation words; it does not evaluate layout, tables, reading order or structured fields.

## Try your photo on your server

On a Linux machine with an NVIDIA GPU and NVIDIA Container Toolkit, start the published SIE vision-OCR image:

```bash
docker run --gpus all -p 8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:v0.9.0-cuda12-sglang-vision-extract \
  serve --host 0.0.0.0 --port 8080 --models-dir /app/models \
  --models zai-org/GLM-OCR --preload zai-org/GLM-OCR
```

In this directory:

```bash
uv run run.py --image receipt.jpg
```

The SDK defaults to `http://127.0.0.1:8080`; `--base-url` or `SIE_BASE_URL` selects your server. Set `SIE_API_KEY` only if your endpoint requires it. Responses are saved under `runs/`. The example stops if its output file already exists, so a second command does not silently replace a recorded run.

GLM-OCR is not currently offered on SIE Cloud. The release image supports the model with a 16-request default limit. The measured study used public source [7a56b3c](https://github.com/superlinked/sie/tree/7a56b3c6bbdeb2408f9a4392a567175acd16c93e), Transformers 5.3.0, and the main profile's 64-request limit. These source and runtime settings are pinned in the evidence. The proposed cost lane is not the unmodified release profile.

To build that public source's serving implementation, rather than the release image:

```bash
git clone https://github.com/superlinked/sie.git
cd sie
git checkout 7a56b3c6bbdeb2408f9a4392a567175acd16c93e
docker build -f packages/sie_server/Dockerfile.cuda12 \
  --build-arg BUNDLE=sglang-vision-extract \
  --build-arg SIE_SRC_REV=7a56b3c6bbdeb2408f9a4392a567175acd16c93e \
  -t sie-glm-ocr:recorded .
```

Use `sie-glm-ocr:recorded` in the server command. The recorded processor stack was Transformers 5.3.0; a newly resolved container is not guaranteed to select that exact dependency version. The offline rescore below reproduces the recorded figures without depending on a new inference environment.

## Reproduce the recorded comparison without inference spend

```bash
uv run fetch.py
uv run score.py
```

`fetch.py` anonymously downloads the count-only evidence from an immutable revision of [superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence). It verifies the pinned manifest and every file hash. `score.py` recalculates overall and domain word metrics, the seeded paired bootstrap, the frozen comparison gates and GPT-5.4 mini's token-priced cost. Raw document text is not redistributed in this package; reliability checks are recorded evidence, not a fresh audit of response text by this count-only rescore.

| Scope | Z.ai GLM-OCR mean image word F1 | GPT-5.4 mini mean image word F1 |
| --- | ---: | ---: |
| 96 photos, balanced receipts and handwriting | 86.84% | 84.25% |
| 48 receipt photos | 91.95% | 87.44% |
| 48 handwriting photos | 81.73% | 81.06% |

The plan was frozen locally before the requests. The primary rule tested noninferiority within 3 percentage points, rather than a superiority hypothesis. Its one-sided 95% lower bound was +1.03 points, passing that rule. The descriptive difference was +2.59 points, with two-sided 95% interval [+0.73,+4.45]. The precision, per-domain, value-token and reliability gates also passed. All 96 responses from each model were successful and nonempty, with no fixed repetition or token-cap warning.

CORD-v2 annotations can omit visible headers. Word F1 therefore measures agreement with the annotation words, rather than complete transcription accuracy. Exact words count as a multiset, including duplicate words; reading order does not enter this metric. These photos were not used by the preceding 600-image candidate screen, but their source splits do not establish that the models never saw them during training.

GPT-5.4 mini ran as `gpt-5.4-mini-2026-03-17`, with reasoning off, image detail high and a 4,096-token output cap. Its recorded token usage works out to $1.99 per 1,000 photos at the [synchronous API rates](https://developers.openai.com/api/docs/models/gpt-5.4-mini). GLM's proposed Cloud rate is $1.58 per 1,000, about 21% less on this token mix. It is not a live billable rate. It comes from twice the earlier worst-of-three RTX PRO6000 serving floor, rounded up to cents; that hardware workload is separate from this 96-image quality test. Self-host costs depend on hardware and utilization.

## Rebuild the same inputs for your server

```bash
uv run --extra recorded prepare.py
uv run run.py --recorded
```

This downloads the pinned source datasets, including the roughly 1 GB GNHK archive, and reconstructs all 96 photos. Every original pixel hash and final JPEG hash must match the frozen sample metadata. A changed source or different JPEG encoding fails instead of silently substituting inputs. Your new server responses are saved separately from the public recorded evidence; the two commands do not rerun the GPT comparison or reproduce its paid calls.

The sources are [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2), published by NAVER Clova, and [GNHK](https://github.com/GoodNotes/GNHK-dataset), published by GoodNotes, both under CC BY 4.0. The frozen selection uses CORD validation and GNHK training photos; attribution, immutable revisions, preprocessing and metric rules are in the downloaded evidence's manifest and PREREGISTRATION.md.
