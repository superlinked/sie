# SIE primitives in the agent action gate

A reference for `examples/agent-action-monitor/README.md`, shaped like
`superlinked/sie`'s existing `stripe-link-fraud` example: a model lineup,
where each primitive is actually used in this codebase, and an honest
account of what the deterministic core still does versus what SIE adds.

## Model lineup

| Model | Primitive | Role |
|---|---|---|
| `BAAI/bge-m3` | encode | Embeds an action's description for similarity search against past decisions. |
| `BAAI/bge-reranker-v2-m3` | score | Cross-encoder rerank of the top candidate matches, and of an agent's own history against a new action. |
| `urchade/gliner_multi-v2.1` | extract | Zero-shot extraction of privileged terms (role, privilege, resource, segment, port) from an action's target and change payload, with no training data. |

All three are verified against the live Superlinked model catalog
(`superlinked.com/models`), not assumed from a family name.

## Where each primitive is wired in

- **encode** -- `src/dusk/trace/vector.py`'s `sie_encode()` (wrapped by
  `embed_text()`), called live by `/v1/gate` (`src/dusk/api.py`): once per
  request to embed the incoming action, and once per verdict to record it
  for future lookups. `find_similar_cached()` compares the fresh query
  embedding against a bounded, pre-embedded history (capped at 200 entries)
  to populate the response's `similar_decision_ids`, without re-embedding
  that history on every call.
- **score** -- `src/dusk/trace/vector.py`'s `sie_score()`, used two ways:
  reranking the encode-shortlisted candidates for `similar_decision_ids`,
  and inside `src/dusk/actions/analyse.py`'s `_semantic_novelty()` to check
  a new action's rerank similarity against the acting agent's own raw
  baseline history. Its raw cross-encoder output is a logit with no fixed
  scale, so `sie_score()` bounds it into `[0, 1]` via sigmoid before a
  fixed threshold compares against it -- that bounding is monotonic, not a
  calibrated probability (see Known limits below).
- **extract** -- `src/dusk/trace/vector.py`'s `sie_extract()`, used inside
  `src/dusk/actions/analyse.py`'s `_extracted_sensitive_terms()` to flag
  privileged terms the static frozenset (`_SENSITIVE_TOKENS`/
  `_SENSITIVE_VALUES`) does not already cover. Each extraction keeps its
  GLiNER confidence score; terms below `_EXTRACT_CONFIDENCE_FLOOR` (0.5)
  are dropped rather than counted, so a low-confidence zero-shot guess
  doesn't carry the same weight as one the model was actually sure about.

## What happens without SIE

Every one of the three call sites above degrades to a no-op or a
deterministic fallback rather than failing: `sie_encode` falls back to a
hash-based n-gram embedding, `sie_score` and `sie_extract` return `None`/`[]`,
and every downstream signal that depends on them is additive-only, so the
gate's rule-based score is never reduced by their absence. `dusk gate` and
`/v1/gate` work identically without any SIE container running.

This degrades quickly, not just gracefully: all three calls pass
`wait_for_capacity=False` and a short `provision_timeout_s`, so a model that
isn't warm yet fails in ~1.5s rather than blocking the request while the
SDK's own retry loop waits for it. See the
[local run instructions](../README.md#run-it-locally) for the expected cold
start behavior.

## Validated against a real SIE cluster

`tests/test_sie_live_benchmark.py` skips until `DUSK_SIE_ENDPOINT` and, for
authenticated deployments, `SIE_API_KEY` point at a reachable cluster. Run
against Superlinked's hosted tester endpoint, both checks pass:

- `sie_encode` returns a real 1024-dimension dense vector from `BAAI/bge-m3`
  (confirming the model actually loaded and served, not just that the
  endpoint answered).
- Precision and recall on the labelled fixture stay at 1.0/1.0, matching the
  deterministic-only baseline exactly -- no regression from enabling SIE.
- At least one attack's `reasons` carries a real `SIE rerank` or
  `SIE extract` marker, confirming the primitives are actually contributing
  a signal over the network, not a no-op that happens to still pass.
- The full test suite passes unchanged with live SIE enabled, confirming
  nothing depends on the deterministic fallback path being taken. The exact
  test count is intentionally omitted because it changes as coverage grows.

This is evidence that SIE is load-bearing here ("removing SIE degrades the
result"), not just a claim.

## Known limits

- The baseline/attack fixtures used in the benchmark (`lab/actions/
  generate_actions.py`) are synthetic, not real production traffic.
- The deterministic feature checks in `actions/baseline.py` and
  `actions/analyse.py` still do the primary anomaly scoring; SIE's three
  primitives are an enrichment layer on top, not a replacement for it. This
  matches the project's own stance that the core detection logic is not
  dependent on any AI model at runtime.
- `sie_score`'s rerank pass only reorders a small shortlist (`top_k`,
  default 3) of candidates already retrieved by cosine similarity -- it does
  not rerank the full decision history.
- `_SEMANTIC_SIMILARITY_FLOOR` (0.3) is a heuristic cutoff on the
  sigmoid-bounded rerank score, not a value derived from an empirical
  calibration set. Sigmoid makes the score bounded and monotonic; it does
  not make it a calibrated probability that 0.3 has a principled meaning
  against.
- The live decision history behind `similar_decision_ids` is in-memory and
  capped at 200 entries per gate process -- a demo-scale audit trail, not a
  durable store. It resets on restart and is not shared across replicas.
- `sie_extract`'s privileged-term detection is zero-shot: it has not been
  evaluated against an adversarial corpus designed to evade GLiNER
  specifically, only against the same synthetic fixtures used elsewhere.
  The 0.5 confidence floor is a reasonable default, not an empirically
  tuned threshold.
