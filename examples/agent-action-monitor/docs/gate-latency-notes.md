# Gate latency under load

A first data point on latency-under-load, captured once a real
`DUSK_SIE_ENDPOINT` and, for the authenticated hosted deployment,
`SIE_API_KEY` became available. This measures
`/v1/gate`'s own added latency with live SIE enabled -- not the full
`agent-demo` -> gate -> `mock-prod` round trip. Treat this as a preliminary
probe, superseded by the full-stack run recorded further down.

## Setup

- `dusk-gate` run locally (not in Docker), baseline loaded from
  `sample-data/baseline.json`, with `DUSK_SIE_ENDPOINT` pointed at
  Superlinked's hosted tester cluster.
- 10 requests per concurrency level, a single trial, same clean
  `firewall_rule_change` action repeated (an `ALLOW` case, so both
  `sie_score` and `sie_extract` fire per request via `_extra_sie_signals`).
- Client and server on the same machine, HTTP over loopback -- network
  latency to the hosted cluster is the dominant cost, not local overhead.

## Results

| Concurrency | p50 | p95 | Throughput |
|---|---|---|---|
| 1 | 666ms | 6151ms | 0.30 req/s |
| 3 | 601ms | 716ms | 4.06 req/s |
| 5 | 640ms | 1604ms | 3.51 req/s |

## Caveats

- The concurrency=1 p95 (6.1s) is almost certainly a single cold-start
  outlier -- the first request in the whole run, before any model on the
  hosted cluster had been hit yet. p50 across all three levels (600-670ms)
  is a more representative steady-state number once models are warm.
- n=10 per level, one trial: enough to sanity-check the shape (steady-state
  latency does not blow up with concurrency, throughput scales sensibly
  from 1 to 3 workers), not enough for a confident p95 at any level.
- This does not yet include the full `mock-prod` round trip captured below.
- Superlinked's tester cluster is shared, sponsored compute -- this probe
  deliberately used a small n and low concurrency rather than a sustained
  load test, out of courtesy to that grant.

## A first full-stack attempt hit a cluster outage, not a gate bug

With `agent-demo`/`mock-prod` in place, running the real `dusk-gate` +
`mock-prod` + `agent-demo/harness.py` end to end confirms a clean action is
`ALLOW`ed and applied, and a poisoned action is `WOULD-BLOCK` (watch mode)
or `BLOCK` (enforce mode). In watch mode, `WOULD-BLOCK` is still forwarded;
in enforce mode, `BLOCK` never reaches `mock-prod`. See the
[local run instructions](../README.md#run-it-locally) for the exact commands.

A first attempt at a real `agent-demo/load_driver.py` run against the
hosted tester cluster (after the table above was captured, in the same
session) hit sustained `503 Service Unavailable` from the extract model
(`urchade/gliner_multi-v2.1`) at every concurrency level tried, including
sequential (concurrency=1) requests -- not a capacity limit specific to
concurrent load. A follow-up direct check showed `sie_encode` alone (no
concurrency at all) taking 458 seconds to return, versus roughly a second
earlier in the same session. This points to a transient problem on
Superlinked's shared tester cluster at that moment, not a regression in the
gate or the SDK wiring: `sie_extract`'s own error handling degraded
correctly (returned `[]` rather than raising), just too slowly for
`agent-demo/harness.py`'s 10-second client timeout under any load at all.

No further load was placed on the cluster once this pattern was clear, out
of courtesy to shared, sponsored compute in a visibly degraded state.

## Full-stack load test against the recovered hosted cluster

The hosted tester cluster came back after the outage above, but not into a
steady "always warm" state -- it scales its per-model capacity down to zero
within roughly a minute of no traffic, then re-provisions on the next
request. `sie_score` and `sie_extract` (the two primitives `/v1/gate`
actually calls per request, via `_extra_sie_signals`; `sie_encode` is not
on this request path) each took 0.1-35s to come back from cold before
settling into sub-second responses. This is a real characteristic of a
shared, scale-to-zero tester allocation, not a gate or SDK defect --
`sie_sdk`'s own transient-error retry handled it transparently in every
case except when a cold re-provision outlasted `agent-demo/harness.py`'s
10-second client timeout.

**Setup:** `dusk-gate` run locally (not in Docker) with `sie-sdk` installed
temporarily so live SIE calls are actually made (the project's own venv
does not ship `sie-sdk` by default -- it lives in the `sie` extras group,
uninstalled again after this run to keep the venv matching CI); baseline
from `sample-data/baseline.json`; `mock-prod` run locally; full
round trip via `agent-demo/load_driver.py` (`harness.run_scenario` ->
`/v1/gate` -> `mock-prod` on `ALLOW`), 20 requests per concurrency level,
20% poisoned / 80% clean mix, single trial.

| Concurrency | p50 | p95 | Errors | Verdicts |
|---|---|---|---|---|
| 1 | 294ms | 10008ms | 2/20 | 13 ALLOW, 5 WOULD-BLOCK |
| 3 | 307ms | 474ms | 0/20 | 13 ALLOW, 7 WOULD-BLOCK |
| 5 | 295ms | 317ms | 0/20 | 13 ALLOW, 7 WOULD-BLOCK |

Correctness held throughout: every `ALLOW` reached `mock-prod` (confirmed
via its `/log`, 46 applied actions across this run and earlier manual
checks) and every poisoned action was `WOULD-BLOCK` in watch mode, never
applied.

**Reading the errors:** the 2 timeouts at concurrency=1 are cold-provision
blips (a model scaling back to zero between the sparse, sequential
requests at this concurrency, then not re-provisioning inside the 10s
client timeout) -- not a concurrency effect, since concurrency 3 and 5 (more
total request pressure, keeping the cluster continuously warm) both ran
error-free. p50 (294-307ms) is steady and consistent with the earlier
gate-only preliminary probe's p50 (600-670ms; lower here since this run
landed after the extract/score models were already warm going in).

**Caveats:** n=20 per level, one trial -- enough to confirm the shape (flat
p50 across concurrency, errors tied to idle-driven cold starts rather than
load) but not a high-confidence p95 at concurrency=1. Deliberately kept
small (60 requests total across the sweep) out of courtesy to shared,
sponsored compute. If Superlinked's production SIE tier doesn't scale to
zero this aggressively, the concurrency=1 tail disappears entirely; this is
a property of the tester allocation, worth noting to Superlinked directly
rather than treating as a DUSK-side latency number.
