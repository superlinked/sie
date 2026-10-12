# Send each request to a cheap or a strong LLM

## What this shows

Cost routing picks a model for each request before you call one. A request
that a cheap model answers as well as a strong one goes to the cheap model;
the rest go to the strong one. The router has to be much cheaper and faster
than the calls it saves.

This router embeds the request with `BAAI/bge-m3` on SIE Serverless and finds
the 20 most similar requests in a bank of 729 whose answers from both models
were graded. It weights each neighbour by its cosine similarity and averages
whether the cheap model held up on it. If that score reaches a threshold, the
request goes to the cheap model. The threshold was chosen on the bank to keep
98% of the strong model's quality at the lowest cost.

On 728 held-out requests, routing between GPT-6 Luna and GPT-6.1 Sol cost
**$3.19 per 1,000 requests, 38.6% less than always using GPT-6.1 Sol at 95.8%
of its quality**. It sent 77.2% of requests to Luna. Between Claude Haiku 5.5
and Claude Sonnet 5.5 it cost 11.7% less than always using Sonnet, at 99.0% of
its quality. The router itself costs $0.013 per 1,000 requests, included in
those figures.

| GPT-6 Luna → GPT-6.1 Sol | Cost per 1,000 requests | Saving vs always-Sol | Quality vs always-Sol | Sent to Luna |
| --- | ---: | ---: | ---: | ---: |
| Always GPT-6.1 Sol | $5.19 | — | 100% | 0% |
| **SIE router** | **$3.19** | **38.6% (33.6–44.2)** | **95.8% (93.5–98.0)** | **77.2%** |
| Random routing at 77.2% to Luna | $1.51 | 71.0% | 85.2% | 77.2% |
| Always GPT-6 Luna | $0.42 | 92.0% | 80.9% | 100% |

| Claude Haiku 5.5 → Claude Sonnet 5.5 | Cost per 1,000 requests | Saving vs always-Sonnet | Quality vs always-Sonnet | Sent to Haiku |
| --- | ---: | ---: | ---: | ---: |
| Always Claude Sonnet 5.5 | $10.36 | — | 100% | 0% |
| **SIE router** | **$9.15** | **11.7% (9.4–14.5)** | **99.0% (97.7–100.3)** | **34.9%** |
| Random routing at 34.9% to Haiku | $6.99 | 32.5% | 93.8% | 34.9% |
| Always Claude Haiku 5.5 | $0.70 | 93.3% | 82.2% | 100% |

Quality is routed quality as a share of always-strong quality on the same
requests. Bracketed ranges are the study's 95% intervals from a stratified
paired bootstrap. Costs are list prices accessed 2026-10-11, with no caching or
batch discount.

The OpenAI pair did not hold the 98% target on held-out: the threshold that
kept 98% of Sol's quality on the bank kept 95.8% on new requests. Random routing
at the same share to the cheap model keeps 10.6 fewer points of quality for the
OpenAI pair and 5.2 fewer for the Anthropic pair.

## The study

The study was pre-registered before any request was sent
([pre-registration](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/456601521d3eccee3830c29f0f5ee2e97920c212/evaluations/2026-10-11/request-routing-cost-prereg-v1),
[amendment 1](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/a003af2e392964468a107cc6a13c9d8b933f50d4/evaluations/2026-10-11/request-routing-cost-prereg-v1/amendment-1),
published after the pilot and before the main run). Its 1,457 requests come
from SuperGPQA (hard), MATH-500, MBPP+ and Arena-Hard v2.0 chat, alternating
in hash order between 729 calibration and 728 held-out requests. Every model answered every
request at its default reasoning setting. Benchmark answers were graded
automatically; chat answers were judged by a model from the other provider.

Each calibration request is labelled `y = 1 − max(0, q_strong − q_cheap)`: 1
when the cheap answer is as good, lower by the quality it loses. Calibration
chose the threshold for each pair and chose `knn20` from six SIE router
variants. Every figure above is on the held-out requests. The
[results](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/95d34b775b61bf1d584cada099bfed053710fc2b/evaluations/2026-10-11/request-routing-cost-results-v1)
hold every arm, router and record, and the study's own
[`analyze.py`](https://huggingface.co/datasets/superlinked/sie-task-evidence/blob/95d34b775b61bf1d584cada099bfed053710fc2b/evaluations/2026-10-11/request-routing-cost-results-v1/code/analyze.py).

## Verify the result

From this directory:

```sh
uv sync --frozen
uv run --frozen python fetch.py
uv run --frozen python score.py
```

Neither step needs a key. `fetch.py` downloads about 9 MB from the public
Hugging Face dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence)
at revision `95d34b775b61bf1d584cada099bfed053710fc2b` into `evidence/`. It
checks `MANIFEST.json` against the SHA-256 pinned in [`study.py`](./study.py),
and every other file against the size and SHA-256 the manifest lists:

- `inputs/items.jsonl`: the 1,457 requests and their split;
- `records/routers/bge.jsonl.gz`: the bge-m3 vector of each request, with the
  credits SIE Serverless billed for it;
- `items-openai.jsonl` and `items-anthropic.jsonl`: each request's graded
  quality and list cost for both models of the pair, and its recorded router
  score and route;
- `results.json`: the thresholds, the selected router and every strategy on
  held-out.

`score.py` checks the files again, then recomputes every knn20 score from the
vectors, re-chooses both thresholds on calibration, and rebuilds the held-out
rows from the recorded quality and costs. It uses the rules of `analyze.py`
(`router_scores`, `routed`, `choose_tau` and `summarize`) and makes no model
calls. It exits non-zero unless every score, route, threshold and figure
matches `results.json` and the headline above. It takes under a second:

```
GPT-6 Luna (cheap) -> GPT-6.1 Sol (strong), 728 held-out requests
knn20 threshold 0.9457; router $0.013 per 1,000, included

  Strategy                   per 1,000   Saving  Quality  To cheap
  Always GPT-6.1 Sol             $5.19        -   100.0%      0.0%
  SIE router (knn20)             $3.19    38.6%    95.8%     77.2%
  Random at the same 77.2%       $1.51    71.0%    85.2%     77.2%
  Always GPT-6 Luna              $0.42    92.0%    80.9%    100.0%

Claude Haiku 5.5 (cheap) -> Claude Sonnet 5.5 (strong), 728 held-out requests
knn20 threshold 0.9516; router $0.013 per 1,000, included

  Strategy                   per 1,000   Saving  Quality  To cheap
  Always Claude Sonnet 5.5      $10.36        -   100.0%      0.0%
  SIE router (knn20)             $9.15    11.7%    99.0%     34.9%
  Random at the same 34.9%       $6.99    32.5%    93.8%     34.9%
  Always Claude Haiku 5.5        $0.70    93.3%    82.2%    100.0%

Quality is routed quality as a share of always-strong quality on the same requests.
Cost is list price, with no caching or batch discount.

Recomputed 1,457 knn20 scores per pair and both thresholds. Every figure matches results.json.
```

The intervals come from the study's bootstrap in `results.json`; `score.py`
does not resample them.

## Route your own requests

`run.py` embeds a request on SIE Serverless, scores it against the study's 729
calibration requests and their labels for the pair, and prints the score, the
threshold from `results.json` and the model to call. Run `fetch.py` first.

```sh
export SIE_API_KEY=your-key
uv run --frozen python run.py "Solve for x: x/2 + x/3 = 5"
uv run --frozen python run.py --pair anthropic "Write a limerick about a lighthouse keeper"
```

The embedding is one call:

```python
from sie_sdk import SIEClient
from sie_sdk.types import Item

client = SIEClient()  # SIE Serverless when SIE_API_KEY is set
vector = client.encode("BAAI/bge-m3", Item(text=request))["dense"]
```

`SIEClient()` reads `SIE_API_KEY` and connects to SIE Serverless. Set
`SIE_BASE_URL` as well to use your own SIE server instead. That default is
newer than the `sie-sdk` release on PyPI, so `pyproject.toml` pins the SDK to
the [superlinked/sie](https://github.com/superlinked/sie) commit that added it.
In the study a call was billed 1.3 credits on average, and 1 for most requests;
longer requests cost more. A credit is $0.00001. From the study's client host
the router took 345 ms at the median, mostly network: SIE reported a median
inference time of 20 ms.

`--item` routes one of the study's requests by id and prints its recorded
score beside the new one. `--recorded` uses its recorded vector instead of
calling SIE, so it needs no key:

```sh
uv run --frozen python run.py --item mbpp_plus/161 --recorded
```

```
pair   openai: GPT-6 Luna (cheap) or GPT-6.1 Sol (strong)
score  0.9016 < tau 0.9457
route  gpt-6.1-sol (GPT-6.1 Sol)
study  mbpp_plus/161 (held-out)
       recorded score 0.9016, routed strong
```

`--answer` also sends the request to the chosen OpenAI model with the study's
settings and prints the reply. It needs `OPENAI_API_KEY`, and the call is
billed by OpenAI.

### Build a bank from your own traffic

The bank here is the study's: benchmark requests, labelled by how the two
models answered them. It learns partly from the task type the benchmark
templates reveal, so it is not a router for your traffic. To route your own
requests, you need your own sample of requests answered by both models:

1. Answer each request in the sample with both models and grade both answers
   from 0 to 1.
2. Embed each request with `BAAI/bge-m3` and label it with
   `study.labels(q_cheap, q_strong)`.
3. Hold part of the sample back. Choose the threshold on the rest with
   `score.leave_one_out_scores` and `score.choose_tau`, and check it on the part
   you held back.
4. Route new requests with `study.Bank(pair, ids, study.normalise(vectors), labels, tau).route(vector)`.

## Test without inference

```sh
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

The tests check the kNN score and the routing rule on a tiny synthetic bank,
the threshold rule, the SDK request through a mock transport, and that
`fetch.py` rejects a manifest or file that does not match. They need no
network and no evidence.

## What this does not establish

- **Nothing about your traffic.** The requests are benchmark items, and the
  saving depends on the mix: the chat stratum is about a sixth of the requests
  but most of the always-strong spend, and the strong model wins most of it.
  The oracle saving on this mix is 41.6% for the OpenAI pair and 37.5% for the
  Anthropic pair.
- **Not the same quality for the OpenAI pair.** It kept 95.8% of GPT-6.1 Sol's
  quality, below the 98% target. Only the Anthropic pair passed the
  pre-registered rule for "same quality".
- **Grading limits.** Claude Haiku 5.5 often left out the `\boxed{}` the MATH
  prompt asked for; the pre-registered grader marked 35 of those replies wrong,
  about 23 of them correct in another form. The chat judges seldom called ties.
- **Latency elsewhere.** Router latency was measured from one client host.
- **Prices other than list prices.** Caching and batch discounts were not
  applied.
