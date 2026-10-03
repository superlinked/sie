# Get the right values into a JSON Schema, measured against six hosted models

## What this shows

Every model here has a strict structured-output mode, so every answer parses and
fits the schema. Validity cannot tell them apart. This example measures what
can: of the values an answer should contain, the share the model got right.

The same requests went to SIE's Qwen3.8 27B and six hosted models, each through
its vendor's own strict structured-output API:

- **E1**: 500 records of the public
  [Structured Output Benchmark](https://github.com/JigsawStack/sob) (SOB) test
  split. Each record is a question, a few Wikipedia paragraphs and a JSON
  Schema; the gold answer fills the schema. Scored by SOB's own `evaluate.py`.
- **NHTSA**: 150 vehicle safety complaints filed with NHTSA from 1 September
  2026, after every model's training data, so none of them can have seen the
  answers. Gold is the agency's coded crash, fire and injury count, plus the
  make, model and year when the narrative names them.
- **Repeat**: the first 100 E1 records sent again, one request at a time, for
  repeatability and latency.

The comparison was pre-registered before any request was sent: the arms, the
settings, the sample, the metric and the bars a claim had to clear.
[superlinked.com/structured-output](https://superlinked.com/structured-output)
publishes the result, and its
[SOURCES.md](https://superlinked.com/reference/structured-output/SOURCES.md)
lists every source behind it.

## Headline figures

On exact values, SIE's Qwen3.8 27B got 83.5% of the E1 gold values right, the
highest of the seven:

| Model | Values exactly right (E1) | SIE minus model, 95% interval | Lenient (E1) | NHTSA fields right |
| --- | --- | --- | --- | --- |
| SIE Qwen3.8 27B | 83.5% | | 93.0% | 91.8% |
| Claude Haiku 4.5 | 80.4% | +3.1 (+1.1 to +5.0) | 91.5% | 92.7% |
| Claude Sonnet 5 | 79.3% | +4.2 (+2.3 to +6.2) | 92.8% | 91.8% |
| GPT-6 Sol | 77.7% | +5.8 (+4.1 to +7.5) | 92.3% | 91.8% |
| Claude Sonnet 5.5 | 77.2% | +6.3 (+4.4 to +8.2) | 93.1% | 92.3% |
| Claude Opus 5.5 | 76.9% | +6.6 (+4.7 to +8.6) | 92.5% | 91.9% |
| GPT-6 Luna | 76.6% | +6.9 (+5.0 to +8.9) | 92.2% | 91.4% |

The gap is about exact wording. On the lenient reading every model lands
between 91.5% and 93.1%, and on the fresh NHTSA complaints between 91.4% and
92.7%: there, all seven are level. So the page claims two things, both on exact
values only: parity with Claude Sonnet 5, and a lead over GPT-6 Luna.

The Anthropic models' NHTSA figures come from the `sdk-transform` run, because
`output_config.format` rejects the schema as sent; as sent, each scores 0 on
all 150. On E1 each Anthropic model had 2 schemas rejected, scored 0.

## Models and settings

| Model | Called as | Structured output | Settings |
| --- | --- | --- | --- |
| SIE Qwen3.8 27B | `Qwen/Qwen3.8-27B-FP8` on `https://api.superlinked.com/v1/chat/completions` | `response_format` json_schema, `strict: true` | temperature 0 |
| GPT-6 Luna | `gpt-6-luna` | `response_format` json_schema, `strict: true` | reasoning effort none, temperature 0 |
| GPT-6 Sol | `gpt-6-sol` | `response_format` json_schema, `strict: true` | reasoning effort none, temperature 0 |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | `output_config.format` json_schema | temperature 0 |
| Claude Sonnet 5 | `claude-sonnet-5` | `output_config.format` json_schema | thinking disabled; API-default temperature, because the API rejects `temperature` |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | `output_config.format` json_schema | thinking `between_tools`, the setting the API names as thinking off (it rejects `disabled`); API-default temperature |
| Claude Opus 5.5 | `claude-opus-5-5` | `output_config.format` json_schema | effort low, since its thinking cannot be disabled |

Every model got the same system prompt, user message and schema, with a
2,048-token output cap (SOB's setting). For E1 those are SOB's
`SYSTEM_PROMPT`, its `build_user_message`, and the record's schema through its
`normalize_schema_strict` (every property required, no extra properties), the
only form OpenAI's strict mode accepts. For NHTSA they are the vehicle-complaint
instruction and schema from the task page, with the complaint narrative as the
document. All calls streamed, on 30 September 2026.

A transport failure (429, 5xx, timeout) was retried up to six times; a record
that still failed scores 0. A schema the vendor's API refuses also scores 0, and
the tables count it separately. Anthropic's `output_config.format` refuses the
NHTSA schema outright (`minimum` on an integer), so the NHTSA set also has a
second run of each Anthropic model with the schema passed through the anthropic
SDK's own `transform_schema`, the conversion its `messages.parse` applies. Those
show as extra rows marked `sdk-transform`.

## Scoring

- **Value accuracy** (E1) is SOB's `leaf_value_em`, averaged over the 500
  records: the share of the gold answer's leaf values the model returned
  exactly, and zero for an answer that does not parse or validate against the
  record's schema. `score.py` imports SOB's `evaluate.py` unmodified.
- **Lenient accuracy** relaxes the match: equal after lower-casing and removing
  punctuation and articles, or one value containing the other. **Token F1** is
  SOB's own softer measure. Both are secondary.
- **NHTSA** scores each record as the share of its gold fields right. Make and
  model match on letters and digits only, so `F-150` equals `F150`.
- **SIE minus model** is the paired difference over records, with a 95%
  interval from a 10,000-draw bootstrap. Each comparison draws from its own
  generator seeded with 1, so the intervals come out the same everywhere and
  do not depend on which other models are scored.
- **$/1M docs** prices a million documents at each model's mean recorded
  tokens per document, at list price and at the vendor's batch price (half, at
  both OpenAI and Anthropic). SIE is $0.25 in, $2.00 out and $0.025 for cached
  input per million tokens, with no batch discount. `score.py` holds the rates
  and where they were read.

The E1 sample is 500 records drawn with seed `20260930`, in proportion to the
split's difficulty by schema-complexity strata. `score.py` redraws it from the
SOB parquet and stops if it does not come out the same.

## Where the recorded run lives

The inputs and every call are in the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
folder `structured-output-accuracy/`, pinned to one revision by `fetch.py`:

```
structured-output-accuracy/
  inputs/sob_sample.json       the 500 SOB record ids, the seed and the pinned SOB file
  inputs/nhtsa_records.json    the 150 complaints: prompt, schema, the agency's coded fields and the gold
  inputs/nhtsa_pool.json.gz    the 3,215 complaints the 150 were drawn from
  calls/<set>/<model>.jsonl    one row per request: reply, tokens, timings, any error
  manifest.json                arms, settings, run dates, sources and the SHA-256 of every file
```

A calls file keeps every attempt, including rate-limited ones that were sent
again, and `score.py` scores the first answer each record got. `fetch.py` also
downloads the SOB test split from `interfaze-ai/sob` and SOB's scorer from
`JigsawStack/sob` at pinned commits, and checks every file against its
published digest.

## Run it

Reproduce the figures. No key, no inference spend:

```sh
python3 fetch.py
uv run score.py
```

`score.py` prints one table per set. `--json report.json` also writes every
figure to a file.

Look at a request without sending it. The id can be any unique prefix:

```sh
uv run run.py --show e1 bf462440
uv run run.py --show nhtsa nhtsa-11763784 --model claude-sonnet-5
```

Send requests yourself. `--limit` keeps it to a few cents; a rerun resumes
where it stopped:

```sh
SIE_API_KEY=... uv run run.py --record --set e1 --limit 20
OPENAI_API_KEY=... uv run run.py --record --set e1 --limit 20 --model gpt-6-luna
uv run score.py --calls runs
```

With `--calls`, `score.py` scores only the records your calls cover.
`SIE_BASE_URL` points the SIE arm at another SIE server, such as a self-hosted
one, and `--provider openai --base-url` sends the same requests to any
OpenAI-compatible endpoint. A rerun will not match the recording word for
word: the hosted models are not deterministic at temperature 0, and SIE's
answers move a little between serving configurations.

## Licences

SOB's code and dataset are MIT-licensed (JigsawStack). Its text records are
built from [HotpotQA](https://hotpotqa.github.io/), whose Wikipedia-derived
passages are CC BY-SA 4.0; the SOB records and the model answers quoting them
carry that licence. The NHTSA complaints are US government records from
`api.nhtsa.gov`.

## Limits

One run per model, on one day. SOB's gold is exact-match, so a longer
phrasing of a right answer counts as wrong for every model alike; the lenient
column shows how much that moves each one. The NHTSA set is too small to rank
models on its own: it checks that E1's result holds on documents no model
could have trained on.
