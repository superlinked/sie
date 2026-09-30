# Read receipt photos and handwriting, scored against GPT-5.4 mini

## What this shows

SIE's hosted `lightonai/LightOnOCR-2-1B` read 600 photos with one `extract` call each on
30 September 2026: 172 handwritten notes photographed on phones (GNHK), 100 receipt photos
(CORD v2) and 328 receipt scans (SROIE). GPT-5.4 mini, PaddleOCR PP-OCRv5, Tesseract 5 and
EasyOCR read the same images, and every output was scored by the same rule.

| System | Words read exactly | Handwriting (GNHK) | $ per 1,000 images |
| --- | --- | --- | --- |
| SIE LightOnOCR-2-1B | 92.4% | 83.9% | $1.16 |
| GPT-5.4 mini | 91.3% | 81.6% | $1.18 on the batch API, $2.36 standard |
| PaddleOCR PP-OCRv5, self-hosted | 81.9% | 54.6% | $0 plus your servers |
| Tesseract 5, self-hosted | 56.7% | 12.5% | $0 plus your servers |
| EasyOCR, self-hosted | 55.1% | 15.7% | $0 plus your servers |

LightOnOCR-2-1B reads 1.1 points more words than GPT-5.4 mini (95% interval 0.55 to 1.69,
paired over images). That is ahead, and under the 2-point margin the study set before the
run for calling a rival worse, so [superlinked.com/ocr](https://superlinked.com/ocr) says
"as accurately as". `score.py` re-derives every figure from the recorded run and checks it
against the page, whose sources are in its
[SOURCES.md](https://superlinked.com/reference/ocr/SOURCES.md).

The call is one image in and text out:

```python
from pathlib import Path
from sie_sdk import SIEClient

client = SIEClient(api_key="sk-sie-…", base_url="https://api.superlinked.com")
image = {"data": Path("receipt.jpg").read_bytes(), "format": "jpeg"}
result = client.extract("lightonai/LightOnOCR-2-1B", {"images": [image]})
print(result["entities"][0]["text"])
```

## The score

A photo's reference is the list of words its annotators transcribed. A word counts as read
when the output holds an identical token, in any order, so reading order and line breaks
never decide the score. Tokens are Unicode NFKC; Markdown and HTML markup is removed from
model output; text is split on whitespace; leading and trailing punctuation and a trailing
full stop are dropped. Case is kept, except on SROIE, whose reference text is upper-cased by
the dataset. GNHK's placeholder tokens (`%math%`, `%SC%`, `%NA%`) are left out of its
references. "Digits" below is the same rate over reference tokens that contain a digit:
amounts, dates, doses and IDs.

The rule, the arms, the sample and the margins were written down before the run. Recall
does not penalise extra text. LightOnOCR-2-1B's precision (70%, against GPT-5.4 mini's 91%)
is lower because a handful of its outputs repeat text over and over: 10 of its 600 outputs
hold 63% of its unmatched tokens. The evidence holds every arm's precision.

## Settings

| Arm | Configuration |
| --- | --- |
| SIE LightOnOCR-2-1B | SIE Cloud, default profile, one `extract` call per image |
| GPT-5.4 mini | `gpt-5.4-mini-2026-03-17`, Responses API, reasoning effort `none`, image detail `high`, the instruction "Transcribe all text in this image exactly as written, line by line. Output only the text." |
| PaddleOCR PP-OCRv5 | `paddleocr==3.2.0`, `PaddleOCR(lang="en")`, orientation and unwarping off, recognised lines joined |
| Tesseract 5 | Debian `tesseract-ocr`, `--oem 1 --psm 3`, `eng` |
| EasyOCR | `easyocr==1.7.2`, `Reader(["en"])`, detections joined top to bottom, left to right |

Every arm received the same bytes: each photo converted to RGB, longest side at most
2,048 px, JPEG quality 92. GPT-5.4 mini's price is its recorded tokens (1,898 input and
209 output per image on average) at $0.75 and $4.50 per million, halved on the batch API
([OpenAI pricing](https://developers.openai.com/api/docs/pricing), read 30 September 2026).

Google Cloud Vision, AWS Textract `DetectDocumentText` and Azure AI Document Intelligence
`prebuilt-read` are not in the recorded run: we had no credentials for them. The page makes
no accuracy claim about any of the three.

## Where the recorded run lives

The results, the per-image counts and the text every arm returned for the GNHK and CORD
photos are in the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
folder `ocr-receipts-handwriting/`, pinned to one revision by `fetch.py`:

```
ocr-receipts-handwriting/
  results.json     every arm's rates, overall and per set, the paired tests, and the prices
  per_image.json   per image and arm: reference words, words read, digit tokens, digits read
  outputs/         each arm's text for the 272 GNHK and CORD photos
  serving.json     SIE's measured serving throughput and cost per image on three GPUs
  images.jsonl     every image's source and SHA-256
  manifest.json    the sets, their licences, and the SHA-256 of every file
```

SROIE's receipts are not redistributed, so their text is not in the dataset; their
per-image counts are. `score.py` pins the manifest's SHA-256, and the manifest pins every
other file, so a missing or altered file stops the score instead of changing a number.

## Run it

Reproduce the published figures. Standard library only, no key:

```sh
python3 fetch.py
python3 score.py
```

Read photos yourself, then score them with the same rule:

```sh
export SIE_API_KEY=sk-sie-...
uv run run.py --images 20
python3 score.py --rescore runs
```

`--sets cord,gnhk --all` reads all 272 photos the page may show (GNHK is a 1 GB download).
A score over a few photos is much noisier than the recorded run.

## Credits

GNHK, the GoodNotes Handwriting Kollection (Lee, Chung and Lee, ICDAR 2021), is released
under CC BY 4.0. CORD v2 (NAVER Clova) is released under CC BY 4.0. SROIE is the ICDAR 2019
receipt dataset.
