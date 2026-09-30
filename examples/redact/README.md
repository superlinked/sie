# Mask personal data before it reaches a log, a prompt or a vendor

## What this shows

This example holds the recorded run behind
[superlinked.com/redact](https://superlinked.com/redact). Its sources are in
that page's [SOURCES.md](https://superlinked.com/reference/redact/SOURCES.md).

The study asks one question of each redaction option: how much of the personal
data in a document does it leave readable? It sends 660 synthetic financial
documents through five arms and counts, for each, the in-scope personal-data
spans it masks completely. The metric was fixed before the first request on
30 September 2026.

- **Documents.** 660 rows of the English test split of
  [`gretelai/synthetic_pii_finance_multilingual`](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual)
  (Apache 2.0, revision `7b844d16`), drawn with `random.Random(20260930)`:
  support logs, IT tickets, emails, loan, card and insurance forms, SWIFT and
  EDI messages. The personal data in them is synthetic, and every row carries
  Gretel's own gold spans.
- **Which spans count: 1,792.** The HIPAA Safe Harbor identifiers the set
  labels (45 CFR 164.514(b)(2)), plus credentials: names, street addresses,
  coordinates, dates of birth, phone numbers, emails, SSNs, bank and card
  numbers, driver's licence and passport numbers, employee and customer IDs,
  user names, IP addresses, passwords, PINs, card security codes and API keys.
  Company names, dates, times, routing numbers and SWIFT codes are out of
  scope.
- **The figure: coverage recall.** A span counts as masked only when every
  non-space character of it sits under a mask, whatever label the detector gave
  it. A redactor fails when personal data stays readable, and a mask does not
  care about labels; exact-span F1 would punish a detector for calling an
  account number a card number while it hides it perfectly. F1 and overlap
  precision are printed too.
- **A second figure, fixed before the results were read.** The same, except a
  span also counts when the only characters left readable are a U.S. state,
  a Canadian province, a country or punctuation. Gretel's address spans run
  through the state, and an arm that masks the street, city and postcode but
  not `MI` fails the first figure without leaking anything that identifies a
  person.
- **Intervals.** 95% paired document-cluster bootstrap, 2,000 resamples, seed
  20260930.

## Results

| Arm | Masked (of 1,792) | Coverage recall | 95% interval | States and countries excused | $ a month, 1M documents |
| --- | --- | --- | --- | --- | --- |
| SIE, two models composed | 1,591 | 88.8% | 86.7% to 90.9% | 90.5% | $32 |
| Claude Haiku 4.5 | 1,490 | 83.1% | 80.8% to 85.2% | 92.6% | $1,450 |
| GPT-6 Luna | 1,448 | 80.8% | 78.2% to 83.1% | 93.2% | $113 |
| OpenAI Privacy Filter | 1,214 | 67.7% | 64.3% to 71.0% | 69.8% | $23 (self-hosted, L4) |
| Microsoft Presidio | 859 | 47.9% | 45.0% to 50.8% | 49.6% | $3 (self-hosted, CPU) |
| AWS Comprehend | not measured | | | | $351 |

SIE minus each arm, paired: Presidio +37.5 to +44.3 points, Privacy Filter
+17.8 to +24.5, Claude Haiku 4.5 +2.8 to +8.6, GPT-6 Luna +5.3 to +11.0. With
states and countries excused SIE stays well ahead of Presidio (+37.6 to +44.3)
and Privacy Filter (+17.4 to +24.1), is level with Claude Haiku 4.5 (-4.5 to
+0.3) and behind GPT-6 Luna (-4.9 to -0.4). SIE's lead over the two LLM prompts
on the first figure is addresses whose state they left readable; the label list
has no `state`. So this study makes no detection claim against an LLM prompt.
It does say what they cost, and that redacting with an LLM API sends the
unredacted text to that API.

**AWS Comprehend, Azure AI Language and Google Cloud DLP were not measured.**
The study had no access to them. Comprehend documents an entity type for 96.2%
of the in-scope spans here (all but coordinates, employee IDs and customer
IDs), so nothing here places it above or below SIE. Its row is price only.

Overlap precision is low for every arm (0.35 to 0.60) because Gretel annotates
a subset of each document's personal data: a mask on an unannotated name or
email counts against it.

## The arms

| Arm | How it ran |
| --- | --- |
| SIE | `urchade/gliner_multi_pii-v1` and `numind/NuNER_Zero` on `https://api.superlinked.com`, the same 36 labels, each span kept at score 0.6 or above, the two unioned. Then every other whole-word, case-sensitive mention of a three-letter or longer token from a returned `person` span is masked, in the caller's code. A document over 300 words is sent as 300-word windows with a 50-word overlap, each window's offset added back to its spans. |
| OpenAI Privacy Filter | the `opf` package from [openai/privacy-filter](https://github.com/openai/privacy-filter) at `f7f00ca`, default checkpoint and decoding, every span masked, on one NVIDIA L4 |
| Microsoft Presidio | `presidio-analyzer` 2.2.364 `AnalyzerEngine()` defaults with spaCy `en_core_web_lg` 3.8.0, every entity masked |
| GPT-6 Luna | `reasoning_effort: "none"`, `temperature: 0`, strict JSON schema, the 36 labels in the prompt |
| Claude Haiku 4.5 | no extended thinking, `temperature: 0`, JSON schema, the same prompt |

The LLM prompt: "Find every piece of personally identifiable information in the
text. Use these types: {labels}. Copy each mention exactly as it appears in the
text. Return an empty list if there is none." An LLM returns strings, not
offsets, so every occurrence of a returned string is masked. Three of GPT-6
Luna's 660 replies did not parse and count as no masks.

The windows are there because each GLiNER-family model reads about 384 words
per request, and the server this run was recorded against read only the first
384 and said nothing when it stopped
([#446](https://github.com/superlinked/sie/issues/446)). So the study split
documents over 300 words in the client. A server with the fix for #446 reads a
long document whole, in overlapping windows of its own; `run.py` still sends
the client-side windows, so it reproduces the recorded requests. Each SIE
model alone, one request per document with no floor: GLiNER PII 82.1%, NuNER
Zero 62.7%.

### Price inputs

For 1,000,000 documents a month like these, list prices read on 30 September
2026. `score.py` holds each input as a named constant.

- **SIE:** $0.04 per million input tokens for GLiNER PII and $0.05 for NuNER
  Zero, on the 238,605 and 227,727 tokens their tokenizers count in the 660
  documents.
- **AWS Comprehend `DetectPiiEntities`:** $0.000025 per 100-character unit,
  its cheapest tier, 3-unit minimum per request, units counted as characters
  over 100 with no rounding up.
- **LLMs:** the tokens each provider reported for the run, at $0.10 and $0.50
  (GPT-6 Luna) and $1 and $5 (Claude Haiku 4.5) per million input and output
  tokens.
- **Presidio and Privacy Filter** are free to download, so their price is the
  compute: Modal list price per second (L4 $0.000222, a core $0.0000131, a GiB
  $0.00000222), divided by 75% utilisation and times 1.75 for region, at the
  throughput measured with the best worker count. Presidio: 8 cores and 16 GiB,
  98.5 documents a second. Privacy Filter: one L4, 8 cores and 32 GiB, 39.7
  documents a second.

## Run it

Download the recorded run, then score it. Neither step needs a key, and both
are standard library only:

```sh
python3 fetch.py
python3 score.py
```

`fetch.py` pulls the `redact/` folder of the public Hugging Face dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
pinned to one revision, about 4.4 MB. It checks every file against the id the
dataset lists and against the SHA-256 in `manifest.json`, whose own digest is
pinned in `fetch.py`. It holds:

- `inputs/gretel-main.jsonl`: the 660 documents and their gold spans (start
  and end in Unicode code points);
- `rows/`: one recorded row per document per arm. The SIE rows hold the API
  responses (`whole`, and `windows` for documents over 300 words); the LLM rows
  the raw JSON reply and token counts; Presidio and Privacy Filter their spans;
- `results/`: the study's own results file, token counts, and the throughput
  and price per arm;
- `manifest.json`: run date, endpoint, models, labels, prompt and arm settings.

`score.py` composes SIE's two recorded responses, turns each arm's output into
masks, and prints every figure above plus overlap precision, exact-span F1,
each SIE model alone, and how many currency amounts SIE's masks touch. It
exits nonzero unless every figure matches both the recorded results file and
the page. It takes about two seconds; `--no-bootstrap` skips the intervals.

```
660 documents, 1,792 in-scope personal-data spans

Arm                                                  Masked  Coverage  State excused  Precision  Exact F1
SIE (GLiNER PII + NuNER Zero, composed)      1,591 of 1,792     88.8%          90.5%      0.346     0.396
Claude Haiku 4.5                             1,490 of 1,792     83.1%          92.6%      0.595     0.484
GPT-6 Luna                                   1,448 of 1,792     80.8%          93.2%      0.590     0.474
OpenAI Privacy Filter                        1,214 of 1,792     67.7%          69.8%      0.590     0.504
Microsoft Presidio                             859 of 1,792     47.9%          49.6%      0.346     0.167
...
Every figure matches the recorded results and the published page.
```

Run SIE's composition yourself, with your own key:

```sh
uv sync
python3 run.py --show                           # the requests for one document, not sent
SIE_API_KEY=... uv run python run.py            # one document, masked and scored against its gold
SIE_API_KEY=... uv run python run.py --doc 851  # another, by index or uid
SIE_API_KEY=... uv run python run.py --all      # all 660 into run-output/, about 470,000 tokens per model
python3 score.py --sie-rows run-output          # your run beside the recorded arms
```

`run.py` sends through `sie_sdk.SIEClient`, with the same labels and windows as
the recorded run, and for one document prints the masked text, the gold spans
it left readable, and whether its spans match the recording.

## What this does not establish

- **Nothing about your documents.** Gretel's documents are synthetic and
  financial. Measure on your own data before relying on any of these figures.
- **Nothing about AWS Comprehend, Azure or Google Cloud DLP.** They were not
  run.
- **No lead over an LLM prompt.** It depends on whether a readable state counts
  as a leak.
- **Nothing about what stays readable.** Overlap precision is bounded by what
  Gretel chose to annotate, and SIE's name step masks every later mention of a
  word a model called a person, including `Insured` in some insurance forms.
- **Latency is not compared.** The recorded latency is in
  `results/e2_results.json`; the self-hosted arms run in-process and SIE over
  the network, so the numbers are not like for like.
