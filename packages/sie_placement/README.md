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
| `placement-co-residency-receipt` | `CoResidencyReceipt` | One measurement of two or more identities served from the same container on one cell, container shape and concurrency |
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

## Co-residency

A co-residency receipt measures a set of identities loaded together on one
container. `residents` lists two or more identities, each with the request
shape, throughput and latencies it reached while the others were loaded
beside it, in the same form a receipt records them. `gpuMemoryMiB` is the
memory of each of the container's GPUs and `peakGpuMemoryMiB` the most any
one of them held while the set ran, so their difference is the set's GPU
memory headroom. The receipt stands for that exact set only: it says nothing
about a subset or a larger set.

A plan identity carries `coPacked: true` when the planner placed it on
another identity's lane by co-packing. Only a `pinned` identity is co-packed.

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
    co_residency_receipts=co_residency_receipts,
)
print(render(result.plan.to_dict()), render(result.report.to_dict()))
```

`plan` is a pure function of its documents. The same inputs render to the
same bytes in whatever order the receipts, boot evidence, identities and demand
series arrive. An identity whose choice is its current placement keeps its
current lane and that lane's name, so a run whose `current_plan` already holds
every choice, such as its own previous plan with its `coPacked` markers,
reports no change.

| Parameter | Meaning |
| --- | --- |
| `placement` | The price-list placement whose region multiplier applies |
| `max_containers` | The reviewed ceiling of every lane, its `maxContainers` |
| `fallback_floor` | The least containers an active zero-floor lane holds, the `sie-cluster` chart's `autoscaling.fallbackReplicas` |
| `cooldown_s` | How long an active lane keeps its containers after its last request, the chart's `autoscaling.cooldownPeriod` |
| `buffer_containers` | The declared warm buffer above a lane's floor, `0` by default |
| `gpu_memory_headroom_mib` | The GPU memory a co-residency receipt must leave free on each GPU for a co-packing move, `0` by default |
| `hours_per_month` | The hours a monthly figure covers, `730` by default |
| `queue_pool`, `bundle`, `bundles` | Every lane's queue pool, and each identity's bundle: its `bundles` entry, otherwise `bundle` |
| `apart` | Pairs of identities that co-packing never puts on one lane, for rules the documents cannot carry, such as an operator's isolation decision. Empty by default |

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
select `lazy-resident`, `bridged` or `scheduled-warm`, and moves an identity
onto another identity's lane only by co-packing. An identity with no
unblocked option keeps its current lane, capped by the reviewed ceiling, or
stays out of the plan when it has none.

### Co-packing

After selection, one pass visits the placed identities in decreasing floor
cost, the monthly cost of the floor of the lane selection gave them, with ties
broken by identity. A visited identity that is the only resident of its lane
may move onto another lane of the plan, on a cell the price list prices,
whose residents are all `pinned` and include none that `apart` pairs with the
identity, whose `queuePool` is the identity's and whose `bundle` is the
identity's bundle. The move is kept only when:

1. a co-residency receipt measured exactly that lane's residents and the
   identity, on the lane's cell, container shape and concurrency;
2. the receipt leaves at least `gpu_memory_headroom_mib` free on each GPU;
3. every resident, the identity included, meets its warm bounds at the
   latencies the receipt records for it. The lane keeps a warm floor, so every
   come-alive bound holds;
4. the plan's monthly cost falls by at least a cent.

Of the moves an identity can make, the one that saves most wins, and ties go
to the lane whose residents sort first. The identity joins the lane as
`pinned` with `coPacked: true`, and its own lane leaves the plan. A lane that
received an identity hosts later moves only when a receipt covers the larger
set. A move onto a lane that no co-residency receipt covers is a measurement
request instead, and a move that a receipt measured but that fails 2 or 3 is
reported with the numbers that block it. The pass is order dependent by design,
and the visiting order pins it. A `coPacked` placement never counts as an
identity's current placement, so the next run makes the same move from the
identity's own lane and a `current_plan` that is the planner's previous plan
yields an empty diff. The marker is how the pass tells a co-packed identity
from a lane's own residents, so a `current_plan` rebuilt from the deployed
lanes, such as chart values, must restore it: without it, a co-packed identity
whose own choice matches its lane stays there as a resident from the start, and
the pass may then make moves the previous run did not. The planner refuses a
co-residency receipt whose machine profile has no price or whose realized GPU
product differs from the profile's, or one with a resident whose latencies
cannot answer its service level, and ignores one that names an identity
without a service level.

### Pricing

A container costs `(acceleratorPerHour * gpuCount + corePerHour * cpuCores +
memoryGiBPerHour * memoryMiB / 1024) * regionMultiplier` an hour. An identity's
demand is its hourly requests summed over operations. A container serves
`requestsPerSecond * 3600` requests an hour at the receipt's concurrency,
counted exactly at the decimal rate the receipt states, and no hour runs more
containers than the ceiling. Monthly figures scale the demand window to
`hours_per_month`.

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

A lane whose residents are all `pinned` and that a co-residency receipt
covers is priced at that receipt, the one that prices it lowest when several
admit the set: each resident's demand at its co-resident throughput, and the
lane's containers above the floor from the residents' combined load in each
hour. A resident's own monthly cost there is the floor plus what its demand
alone needs above it.

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
- `identities`: each identity's status (`placed`, `co-packed`, `kept` or
  `unplaced`), effective service level and demand, its lane and `monthlyCost`,
  what its demand costs there with the lane's floor, the chosen cell, warmth,
  monthly cost and come-alive with its basis (`warm-floor`, `boot-evidence` or
  `borrowed-boot-evidence`), the peak containers its demand needs, and
  `nextCheaper`: the nearest cheaper option a service level blocks, its saving
  and the bounds that block it. The peak is counted at the throughput its lane
  prices it at. A co-packed identity's lane is the lane it joined, its
  `chosen` option the one selection gave it, and `joined` the placement it
  runs in: that lane's cell, container shape and concurrency, `pinned` with a
  `warm-floor` come-alive, at the lane's co-residency receipt. An identity's
  `nextCheaper`, measurement requests and levers are measured against what the
  plan pays for it: its chosen cost less the savings of the co-packing moves
  whose co-residency receipt names it, since those moves need it on its lane.
  A kept identity's `blockedBy` names the bounds its kept placement is known
  to miss: its receipt's warm latencies, a measured come-alive over the bound,
  or a cold warmth under a bound of `0`, which needs no measurement.
- `lanes`: each lane's hourly cost, its monthly cost, its floor's monthly cost
  and the monthly cost of its `floorMax`, which a floor budget prices. A lane
  on a cell the price list cannot price carries `unpricedReason` and no cost,
  and `unpricedIdentities` lists residents whose demand has no receipt to price
  it on the lane. `coResidencyReceipt` names the receipt a co-packed lane is
  priced at.
- `sharedChartLanes`: every chart lane, by `queuePool`, `machineProfile` and
  `bundle`, that more than one plan lane names.
- `coPackingMoves`: the moves the co-packing pass kept, in the order it made
  them, each with the lane it left and its warmth there, the lane it joined and
  that lane's residents, the co-residency receipt, the GPU memory headroom the
  receipt leaves and the monthly saving.
- `blockedCoPackingMoves`: moves onto a lane that co-residency receipts
  measured but none admits, ranked by saving. Each names the receipt that would
  price the lane lowest, `blockedBy` with the resident `identity` and the warm
  bound its latency misses, or `gpuMemoryHeadroomMiB` with the required and the
  measured headroom, and `saving`, what the move would save beyond the plan.
  A move that would save nothing more is left out.
- `measurementRequests`: options that lack evidence, ranked by the saving they
  could unlock. A profile without a receipt is bounded by price alone: one
  accelerator at its rate, kept warm or, where the come-alive bound allows,
  paying burst tails only. A cell whose receipt has no successful boot asks for
  boot evidence and is bounded by its cold cost. A co-packing move that no
  co-residency receipt covers asks for one on the lane's cell, container shape
  and `concurrency`, naming the `lane` and its `coResidents` as the pass found
  them when it visited the identity. A later move may vacate that lane or add
  residents to it, but that set is the one the pass looks for at the visit, so
  it is the one a receipt must measure. The request is bounded by what the plan
  pays for the identity, and requests that need the same receipt are one
  request, bounded by the larger saving. A request that cannot save a cent is
  dropped, and the requests of an identity without a placement carry no bound
  and rank first.
- `comeAliveLevers`: cold warmths whose measured come-alive misses the bound
  on a cell that meets the warm bounds for less than what the plan pays for
  the identity. Each is priced under the identity's bound, where a bound of 0
  admits no lever, and at `permittedBoundS`, the bound its measured come-alive
  would permit. Levers rank by the first saving, then by the second.
- `queueWaitUnattained`: every identity that declares `queueWaitP99Ms`. No
  plan claims that bound as met.
- `refusals`, where a refused co-residency receipt names its `residents`, and
  `changes`: the identities and lanes that differ from `current_plan`.

## Tests

```bash
mise run test -- packages/sie_placement/tests
```
