# Tag product photos with your own types, colours and materials

## What this shows

A catalogue team fills in the same fields on every listing: what the product is, its colour, its material. Each team
has its own lists. This example measures how often each service fills in all three fields right from the product photo
alone, when the lists are the caller's.

- **938 held-out product photos** from [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
  (CC BY 4.0), with a list of 51 types, 14 colours and 11 materials. Another 311 photos (the dev set) pick the one
  setting each arm is allowed to tune.
- **SIE** runs `google/siglip-so400m-patch14-384`, an open model: one encode call per photo, and each field takes the
  value whose prompt is closest to the photo. Nothing is trained.
- **Vision LLMs** (GPT-6 Luna, GPT-5.4 mini, GPT-5.4 nano, Claude Haiku 4.5) get the three lists in the prompt and a
  strict JSON schema.
- **AWS Rekognition `DetectLabels`** returns its own fixed labels, so they are mapped onto the lists with a map fitted
  on the dev set in Rekognition's favour: a specific label ("Ottoman") beats its parent ("Furniture").

The figures are published at [superlinked.com/image-classify](https://superlinked.com/image-classify); that page's
sources are in its [SOURCES.md](https://superlinked.com/reference/image-classify/SOURCES.md).

The run is already recorded. Every answer, every SIE cosine and the Rekognition map live in the public HuggingFace
dataset [superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence) (folder
`catalogue-tagging/`), pinned to one revision by `fetch.py`. Download it and you can re-derive every published number
with **no API key and no inference spend**.

## Reproduce the figures

Both steps are standard library only:

```sh
python3 fetch.py
python3 score.py
```

`score.py` re-derives every arm from the recorded answers, prints the table below and checks each figure against the
page's `page-evidence.json`:

| Arm | All three fields right | Type | Colour | Material | $ per million photos |
|---|---|---|---|---|---|
| GPT-6 Luna | 84.2% | 94.6% | 92.9% | 96.0% | $162 |
| GPT-5.4 mini | 77.7% | 91.4% | 89.8% | 93.2% | $650 |
| **SIE SigLIP so400m-384** | **72.3%** | **95.8%** | 86.1% | 87.5% | **$24** |
| Claude Haiku 4.5 | 71.2% | 90.6% | 86.3% | 90.4% | $2,158 |
| GPT-5.4 nano | 64.3% | 86.2% | 86.8% | 85.8% | $174 |
| AWS Rekognition | 26.8% | 82.0% | 40.0% | 77.5% | $1,750 |

Prices are real-time list prices for one million photos a month: LLMs on the tokens they actually used (each at the
cheaper resolution within a point of its best), Rekognition at its first-million tier with the image-properties charge
its colour answer needs.

## Tag your own photo

```sh
export SIE_API_KEY=sk-sie-...
uv run run.py photo.jpg --types "sofa, armchair, ottoman" --colours "blue, grey, green" --materials "velvet, leather, fabric"
```

With no lists given (after `python3 fetch.py`), it uses the study's own 51 types, 14 colours and 11 materials.
