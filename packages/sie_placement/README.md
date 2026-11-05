# sie-placement

Versioned documents for a measured-only placement planner: the service level
each served identity must meet, the measurements a placement may be chosen
from, the prices those placements cost, the demand they serve, and the plan
that results, together with the planner that turns them into a plan and a cost
report. The package uses only the Python standard library. It is a
workspace package and is not published to PyPI.

A served identity is a model id or a `model:profile` pair, the routing key a
request resolves to. A measurement cell is a machine profile together with the
GPU product the measurement actually ran on, such as the
`nvidia.com/gpu.product` node label value `NVIDIA-L4`. Recording the realized
product rather than the requested one keeps a measurement taken on different
silicon from standing in for the profile it was requested on.

## Documents

Every document is a JSON object with a `kind` and a `schemaVersion`. Readers
reject another kind, any version other than `1`, unknown fields, `null`
values, booleans where numbers are expected and non-finite numbers. Errors
name the offending field by its JSON path, for example
`$.lanes[1].envelope.floorMax: must not exceed maxContainers (2), got 3`.

| Kind | Type | Contents |
| --- | --- | --- |
| `placement-service-levels` | `ServiceLevels` | Optional named `tiers` and the `identities` that declare or inherit service levels |
| `placement-receipt` | `Receipt` | One throughput and latency measurement of one identity on one cell, container shape, concurrency and request shape |
| `placement-boot-evidence` | `BootEvidence` | One measured boot, snapshot restore or resident load of one identity on one cell and container shape |
| `placement-price-list` | `PriceList` | Accelerator, core and memory rates per machine profile, and the region multiplier per placement |
| `placement-demand` | `Demand` | Hourly request counts per identity and operation over a UTC window |
| `placement-plan` | `Plan` | Lanes with their cell, container shape, concurrency, ceiling and envelope, and the lane and warmth of every identity |

Each type parses with `from_dict(data)` and serializes with `to_dict()`.
Constructing a type in code applies the same rules as parsing it:

```python
import json

from sie_placement.schema import Plan

with open("plan.json", encoding="utf-8") as handle:
    plan = Plan.from_dict(json.load(handle))
assert Plan.from_dict(plan.to_dict()) == plan
```

## Operations

Receipts and demand name the primitive a lane runs: `encode`, `score`,
`extract` or `generate`. Gateway telemetry labels requests by route and
already reports the generation and transcription routes as `generate` and
`extract`. An adapter folds the remaining route labels the same way:
`embeddings`, the OpenAI-compatible route that the gateway rewrites into an
encode request, counts as `encode`; `moderations` answers 501 without serving
a model, so it carries no demand to place; `other` names no primitive and is
not assigned to one.

## Service levels

An identity declares the fields below, or inherits them from the tier it
names. Fields it declares override the tier's.

| Field | Meaning |
| --- | --- |
| `comeAliveP95S` | Seconds from a request for an idle identity to its first result, at the 95th percentile. `0` means never cold |
| `warmP99Ms` | Warm latency at the 99th percentile, at the receipt's request shape |
| `ttftP99Ms`, `itlP99Ms` | Time to first token and inter-token latency at the 99th percentile. A generation identity declares both instead of `warmP99Ms` |
| `queueWaitP99Ms` | Time a request may wait for capacity under load, at the 99th percentile |

## Warmth

A plan gives every identity one warmth: `pinned`, `cold-with-bound`,
`bridged`, `snapshot-cold`, `scheduled-warm` or `lazy-resident`. A `pinned` or
`lazy-resident` identity sits on a lane whose envelope keeps `floorMin` at 1
or more; every other warmth sits on a lane whose `floorMin` is 0. Boot
evidence records `cold-with-bound` boots, `snapshot-cold` restores and
`lazy-resident` loads, with outcome `succeeded`, `failed` or `timed-out`.
`borrowed: true` marks a boot measured in another environment whose
boot-shaping inputs match: the identity and its model revision, the image, the
realized GPU product and count, the container shape, the warmth and the
placement class. Whoever supplies the evidence decides the match.

## Lanes

A plan lane carries the three tokens that name a lane of the `sie-cluster`
Helm chart: `queuePool`, the `machineProfile` of its `cell`, and `bundle`.
`queuePool` is the physical queue pool the lane's workers consume:
`workers.common.queuePool` (`default`), or a worker group's `queuePool`
override. It is neither the `workers.pools` map key, which names a worker
group, nor an SDK logical pool. `bundle` is a key of that worker group's
`bundles` map. `name` is the lane's handle inside the plan, and each identity
refers to its lane by that name. Two plan lanes may carry the same three
tokens, and so name one chart lane, which cannot give them separate envelopes
or concurrency; the planner's report lists every such chart lane.

## Envelope

A plan lane's envelope bounds what an online tuner may change without a new
plan. `floorMin` and `floorMax` are required, with `floorMax` at most the
lane's `maxContainers`. `bufferMin` with `bufferMax`, and
`scaledownWindowMinS` with `scaledownWindowMaxS`, are optional pairs for
platforms that can move a warm buffer or a scale-down window per lane.

## Planner

```python
from sie_placement.planner import PlannerParameters, plan
from sie_placement.report import render

result = plan(
    service_levels,
    receipts,
    boot_evidence,
    price_list,
    demand,
    current_plan,
    PlannerParameters(placement="us-central1", max_containers=4, fallback_floor=2, cooldown_s=900),
)
print(render(result.plan.to_dict()), render(result.report.to_dict()))
```

`plan` is a pure function of its documents. The same inputs render to the
same bytes in whatever order the receipts, boot evidence, identities and demand
series arrive. An identity whose choice is its current placement keeps its
current lane and that lane's name, so a run whose `current_plan` already holds
every choice, such as its own previous plan, reports no change.

| Parameter | Meaning |
| --- | --- |
| `placement` | The price-list placement whose region multiplier applies |
| `max_containers` | The reviewed ceiling of every lane, its `maxContainers` |
| `fallback_floor` | The least containers an active zero-floor lane holds, the `sie-cluster` chart's `autoscaling.fallbackReplicas` |
| `cooldown_s` | How long an active lane keeps its containers after its last request, the chart's `autoscaling.cooldownPeriod` |
| `buffer_containers` | The declared warm buffer above a lane's floor, `0` by default |
| `hours_per_month` | The hours a monthly figure covers, `730` by default |
| `queue_pool`, `bundle`, `bundles` | Every lane's queue pool, and each identity's bundle: its `bundles` entry, otherwise `bundle` |

### Selection

The planner chooses a cell and warmth for each identity of the service-level
document:

1. It refuses a receipt or boot record whose machine profile has no price, or
   whose realized GPU product differs from the product the price list gives
   that profile, and a receipt whose latency cannot answer the identity's
   service level. The report lists every refusal.
2. Every other receipt is a cell. A cell whose p99 misses `warmP99Ms`, or
   `ttftP99Ms` or `itlP99Ms`, is blocked.
3. A cell offers `pinned`, and `cold-with-bound` or `snapshot-cold` where boot
   evidence for the identity, cell and container shape exists. The slowest
   successful boot stands for the come-alive percentile, a boot that did not
   succeed withdraws the warmth, and the environment's own evidence replaces
   borrowed evidence. A `comeAliveP95S` of `0` admits `pinned` only.
4. The cheapest unblocked option wins. Ties go to the current placement,
   then to the faster come-alive.

The current placement is the identity's current lane when that lane has the
option's cell, container shape and concurrency, the identity's warmth is the
option's, and the lane's `queuePool` and `bundle` are the ones the parameters
give the identity. An identity whose choice is its current placement stays on
that lane under its name, beside whoever else the lane hosts, and the lane
takes the ceiling and envelope of the choice. Every other placed identity gets
a new lane of its own, named from the identity and its machine profile, with a
digest of the identity added when that name is taken. The planner does not
select `lazy-resident`, `bridged` or `scheduled-warm`, and does not move an
identity onto another identity's lane. An identity with no unblocked option
keeps its current lane, capped by the reviewed ceiling, or stays out of the
plan when it has none.

### Pricing

A container costs `(acceleratorPerHour * gpuCount + corePerHour * cpuCores +
memoryGiBPerHour * memoryMiB / 1024) * regionMultiplier` an hour. An identity's
demand is its hourly requests summed over operations. A container serves
`requestsPerSecond * 3600` requests an hour at the receipt's concurrency, and
no hour runs more containers than the ceiling. Monthly figures scale the
demand window to `hours_per_month`.

- `pinned` pays one container all month, plus each container an hour needs
  above it, for that hour.
- A cold warmth pays, in each hour with requests, the work plus a tail after
  every request: the active floor for `cooldown_s`, at most the whole hour.
  The active floor is `fallback_floor` capped by the ceiling, and at least one.

A lane costs its floor all month once, whoever it hosts, plus what each
resident's demand needs beyond the floor: on a lane with a floor, the
containers an hour needs above it, and on a zero-floor lane, the work and the
tails a cold warmth pays. A placed identity's demand is priced at its chosen
receipt, and a kept identity's at the slowest receipt it has for its lane's
cell, container shape and concurrency, under that lane's own ceiling.

### Envelope

`floorMin` is 1 for `pinned` and 0 otherwise. No source measures queue wait
yet, so `floorMax` is `floorMin` plus `buffer_containers`, capped by the
ceiling. A floor raised on a zero-floor lane realizes as at least
`fallback_floor`, so a zero-floor lane's nonzero `floorMax` is at least
`fallback_floor` capped by the ceiling. The envelope carries the floor range
only.

### Report

The report renders as a `placement-report` document:

- `totals`: the monthly cost of every lane of the plan, the monthly cost of
  their `floorMax`, and `unpricedLanes`, the lanes whose cost the totals leave
  out in whole or in part.
- `identities`: each identity's status (`placed`, `kept` or `unplaced`),
  effective service level and demand, its lane and `monthlyCost`, what its
  demand costs there with the lane's floor, the chosen cell, warmth, monthly
  cost and come-alive with its basis (`warm-floor`, `boot-evidence` or
  `borrowed-boot-evidence`), the peak containers its demand needs, and
  `nextCheaper`: the nearest cheaper option a service level blocks, its saving
  and the bounds that block it. A kept identity's `blockedBy` names the bounds
  its kept placement is known to miss: its receipt's warm latencies, a measured
  come-alive over the bound, or a cold warmth under a bound of `0`, which needs
  no measurement.
- `lanes`: each lane's hourly cost, its monthly cost, its floor's monthly cost
  and the monthly cost of its `floorMax`, which a floor budget prices. A lane
  on a cell the price list cannot price carries `unpricedReason` and no cost,
  and `unpricedIdentities` lists residents whose demand has no receipt to price
  it on the lane.
- `sharedChartLanes`: every chart lane, by `queuePool`, `machineProfile` and
  `bundle`, that more than one plan lane names.
- `measurementRequests`: options that lack evidence, ranked by the saving they
  could unlock. A profile without a receipt is bounded by price alone: one
  accelerator at its rate, kept warm or, where the come-alive bound allows,
  paying burst tails only. A cell whose receipt has no successful boot asks for
  boot evidence and is bounded by its cold cost. A request that cannot save is
  dropped, and the requests of an identity without a placement carry no bound
  and rank first.
- `comeAliveLevers`: cold warmths whose measured come-alive misses the bound
  on a cell that meets the warm bounds for less than the choice. Each is priced
  under the identity's bound, where a bound of 0 admits no lever, and at
  `permittedBoundS`, the bound its measured come-alive would permit. Levers rank
  by the first saving, then by the second.
- `queueWaitUnattained`: every identity that declares `queueWaitP99Ms`. No
  plan claims that bound as met.
- `refusals`, and `changes`: the identities and lanes that differ from
  `current_plan`.

## Tests

```bash
mise run test -- packages/sie_placement/tests
```
