# Route requests without a generation call on every message

Choose a support queue, banking intent, voice-assistant action, or agent route
with an embedding classifier. Send requests below its fixed confidence
threshold to Qwen3.8 27B for a constrained answer. Your application uses the
returned route to decide what happens next.

The cascade uses two models through `sie_sdk.SIEClient`:

1. **Qwen3 Embedding 4B** embeds the request. Frozen classifier coefficients
   rank the routes. A score of at least `0.29` returns the top route directly.
2. **Qwen3.8 27B** handles the remaining requests. Its prompt includes every
   route name and up to ten training examples for each of the ten highest-ranked
   routes. A JSON schema restricts the answer to the route list.

The classifier score is a routing threshold, not a calibrated probability of
being correct. CLINC150 includes `none of these` for out-of-scope requests;
the generation model can choose it. Application handoff or human review is
outside this example.

## Verify the recorded comparison

The recorded evaluation uses a fixed 3,000-case subset selected before
inference, with the same cases paired against archived GPT-6 Luna and Claude
Sonnet 5 answers. Those baselines see the full route list and the available
training examples, capped at ten per route. The verifier uses the supplied frozen classifier directly.

From this directory, install the locked environment and download the immutable
evidence revision for the recorded comparison:

```sh
uv sync --frozen
uv run --frozen python fetch.py --revision 2b1185ff8be4cfb6e5f1926ff85452441f9d176b
uv run --frozen python score.py
```

Both steps run without an API key. The evidence lives in
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/2b1185ff8be4cfb6e5f1926ff85452441f9d176b/request-routing/20261001-primary3000-v1).
The downloader verifies every file's size and SHA-256 before transferring it
to a new directory, and writes `manifest.json` last as the completion marker.
It never overwrites an existing destination. Use `--output` and `--evidence`
to download and score another directory.

`score.py` validates the manifest and all files before reading the evaluation
data. It checks ordered case coverage, reconstructs classifier decisions and
the conditional cascade, and recomputes accuracy, paired confidence intervals,
per-dataset results, out-of-scope precision/recall, generation avoidance and
listed-price cost bounds. An inconsistent recording returns a non-zero exit.
See the downloaded `METHOD.md` for model revisions, selection, weights, prompts,
price assumptions, and collection limitations.

On these paired cases, the cascade avoids generation for **76.8%** of requests
under the original population weights. Its conservative modeled cost is
**97.3% lower than Sonnet 5's cheapest theoretical listed-cost bound**.

| Recorded system | Equal-dataset accuracy | Modeled listed cost per 1,000 requests |
| --- | ---: | ---: |
| SIE embedding + conditional Qwen3.8 27B | 89.6% | At most $0.101 |
| GPT-6 Luna | 90.6% | At least $0.1307 |
| Claude Sonnet 5 | 89.0% | At least $3.697 |

The paired accuracy difference against Sonnet is +0.57 percentage points
(95% interval: -0.33 to +1.43). Against Luna it is -1.00 points
(-1.80 to -0.20). The registered one-point non-inferiority criterion across
both rivals does not pass; this result supports a cost comparison, not a claim
of parity with both systems. The cost bounds charge the cascade without cache
discounts and allow each rival the cheaper of ideal cache reads or Batch prices.
They do not establish a bill, serving margin or rival-service latency.

## Try the cascade

After downloading the assets, choose a SIE deployment that exposes both frozen
models above. Set its endpoint and your API key explicitly, then run a small
check:

```sh
export SIE_API_KEY=your-key
export SIE_BASE_URL=http://localhost:8000
uv run --frozen python run.py
uv run --frozen python run.py --limit 100 --output run-output/another.jsonl
```

The default is **20 inputs**, processed sequentially. Each input sends one
embedding request and at most one generation request. `--limit` explicitly
allows 1–3,000 inputs. These calls can spend API credits; there are no retries
or rival-provider calls. The endpoint determines actual availability and charges;
the recorded comparison does not establish current hosted availability.

Output retains answers, available usage and fixed failure codes. Missing usage
and failed-call charges remain unknown. Existing output is never overwritten.
Fresh calls are serving observations; the offline scorer verifies the recorded
comparison and does not score fresh `run-output` files.

## Data and interpretation

| Dataset | Routes | Selected test cases | Attribution and licence |
| --- | --- | --- | --- |
| [CLINC150](https://github.com/clinc/oos-eval) | 150 plus out of scope | 1,000: 818 in scope, 182 out of scope | Larson et al., 2019; [CC BY 3.0](https://github.com/clinc/oos-eval/blob/master/LICENSE) |
| [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) | 77 | 1,000 | Casanueva et al., 2020 / PolyAI; [CC BY 4.0](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE) |
| [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive), en-US | 60 | 1,000 | Amazon Science; CC BY 4.0 |

Each route's examples come from training data, capped at ten per route; MASSIVE's
`cooking query` route has four. The sample, coefficients,
threshold and prompts remain fixed during evaluation. Overall quality gives
each dataset equal weight; cost and generation avoidance use the original
test-population weights. Paired bootstrap intervals describe sampled-case
uncertainty conditional on the retained run. Repeated-request stability and
online latency are not established by this comparison.

Cost figures are dated, modeled listed-price bounds. Per-item embedding
rounding is a conservative individual-request model; it is not settlement of
the batched collection. Additional embedding attempt bounds include unknown
outcomes. Failed generation attempts
retain their registered bounds. These figures do not measure an invoice,
deployment throughput or self-hosted margin.
