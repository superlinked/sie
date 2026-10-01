# Find named objects without training a detector

Send a photograph and the labels to find, then receive pixel boxes from
`google/owlv2-base-patch16-ensemble` through the SIE SDK. Change the labels
when the inspection job changes. The model is also available to self-host.

This example reproduces the recorded comparison behind
[superlinked.com/detect](https://superlinked.com/detect). It uses 2,895 seeded
test images across the 100 RF100-VL datasets, with each dataset's class names
as the only prompt. The domains include equipment, retail inventory, aerial
imagery and scientific images. No arm is fine-tuned for this study.

## Reproduce the recorded results

The code lives here. Licensed inputs, annotations, recorded outputs and usage
live in the public
[task-evidence dataset](https://huggingface.co/datasets/superlinked/sie-task-evidence),
at the immutable revision and manifest digest pinned in `fetch.py` and
`score.py`. Fetching and scoring send no model requests and require no API key.

```sh
uv sync
uv run python fetch.py
uv run python score.py
```

The scorer replays COCO box AP separately for each dataset, then averages the
100 scores. It also replays paired dataset-bootstrap intervals, verifies the
recorded token usage, checks every displayed box against its original output,
and checks the annotated-object hit counts. Altered files, duplicate images,
missing rows or mismatched published figures cause a nonzero exit.

`--summary-only` verifies the immutable manifest, row identities, recorded
score means, displayed boxes and prices, while skipping COCO and bootstrap
replay. It is a quicker integrity check, rather than a replay of raw-box AP.

| Arm | Mean box AP | AP50 | Real-time cost per 1,000 images |
| --- | --- | --- | --- |
| SIE OWLv2 Base | 11.1% | 19.1% | $0.24 |
| GPT-6 Luna | 13.3% | 24.4% | $0.15 |
| GPT-5.4 mini | 5.4% | 13.0% | $1.33 |
| Claude Haiku 4.5 | 0.9% | 1.8% | $1.95 |
| OWLv2 Large | 12.2% | 21.2% | Self-hosted cost depends on GPUs |
| OWLv2 Base, transformers | 10.9% | 18.8% | Self-hosted cost depends on GPUs |
| Grounding DINO Base | 7.5% | 11.5% | Self-hosted cost depends on GPUs |
| LLMDet Large | 10.8% | 16.7% | Self-hosted cost depends on GPUs |

These are COCO AP percentages, not the percentage of objects found. SIE's
served OWLv2 beats GPT-5.4 mini and Haiku on both AP and AP50 with positive
paired 95% intervals. **GPT-6 Luna scores higher and costs less.** The study
does not establish superiority over general vision models or fixed-taxonomy
APIs. Different labels and images can change the result.

The price comparison uses each provider's recorded tokens and its real-time
list rates. The displayed SIE price rounds up; competitors round down.
OpenAI and Anthropic also offer asynchronous Batch rates at half their
real-time prices. Both are retained in the recorded evidence.

## Try a single detection request

Set `SIE_API_KEY` for hosted inference. To use your own SIE server, set
`SIE_BASE_URL` to its gateway URL. Inspect the request before sending it:

```sh
uv run python run.py --show countingpills-12
uv run python run.py --case countingpills-12
```

Only the second command sends a paid request. It runs the same OWLv2 checkpoint,
labels and 0.1 score threshold used by the recorded product check. The runner
checks the catalog checkpoint before sending the image and reports the distinct
execution bundle/config digest observed on the response.

## Data and protocol

RF100-VL source: `LibreYOLO/rf100-vl`, revision
`1987e22ed542539fb3d0b8a3456455c2725079a1`. Every dataset's recorded MIT licence
and original project attribution is in `licenses.json`. The export contains the
sampled test annotations, selected original images, outputs from all completed
main-set arms, and the sample identities. `protocol.json` records the decision
rules, coordinate handling, thresholds and price assumptions.

The 30-image cap per dataset and seeded sampling were fixed before the main
run. LLMs received the same resized images with pixel-coordinate prompts,
strict schemas and no reasoning effort. Outputs were mapped back to the
original image. Detector thresholds match their shipped settings. The full
run, including models that outperform the served model, remains in the
recorded evidence.

The paired latency check used 200 images after five discarded warmups, one
request in flight, and the same Modal us-east client location for all three
services. Median full-response time was 0.716 seconds for hosted OWLv2,
1.372 for GPT-6 Luna, and 1.245 for GPT-5.4 mini. The SIE/Luna median ratio
was 0.522 (paired 95% interval 0.492–0.566). The registered requirement for
a threefold speed claim failed; this example makes no such claim. Full raw
responses and the protocol are included in the immutable evidence export.
