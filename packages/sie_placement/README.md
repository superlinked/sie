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
| `placement-receipt` | `Receipt` | One throughput and latency measurement of one identity on one cell, container shape, concurrency and request shape, alone or beside named co-residents, with the GPU memory the identity holds |
| `placement-boot-evidence` | `BootEvidence` | One measured boot, snapshot restore or resident load of one identity on one cell and container shape, with the GPU memory a successful one left in use |
| `placement-price-list` | `PriceList` | Accelerator, core and memory rates and the card memory per machine profile, and the region multiplier per placement |
| `placement-demand` | `Demand` | Hourly request counts per identity and operation over a UTC window |
| `placement-capacity` | `Capacity` | The accelerators a plan may hold: in total, per GPU product, and in a quota it shares with other environments |
| `placement-plan` | `Plan` | Lanes with their cell, fallback cells, container shape, concurrency, ceiling and envelope, and the lane and warmth of every identity |

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
`lazy-resident` loads, one attempt per document, with outcome `succeeded`,
`failed`, `timed-out` or `unplaced`. `placementWaitS` is the time the attempt
waited for a container to be placed before its `bootSeconds` began. An
`unplaced` attempt never got a container: it records no `bootSeconds`, and
`placementWaitS`, when present, is how long it waited before giving up. A
`lazy-resident` load runs on a container that is already placed, so it records
neither. `borrowed: true` marks an attempt measured in another environment
whose boot-shaping inputs match: the identity and its model revision, the
image, the realized GPU product and count, the container shape, the warmth and
the placement class. Whoever supplies the evidence decides the match.

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

`fallbackCells` is an optional ordered candidate list: cells, after `cell`,
that the lane's containers may be placed on when `cell` cannot be placed. No
fallback cell repeats `cell` or another fallback cell. A realized card is
billed at its own price, so the planner prices such a lane at its dearest
member.

## Capacity

A `placement-capacity` document states the accelerators an environment can
obtain. Each field is optional, and at least one is declared:

| Field | Meaning |
| --- | --- |
| `total` | The accelerators every lane of the plan may hold together |
| `types` | Per `gpuProduct`, the `accelerators` the lanes that may be placed on that product may hold together |
| `sharedQuota` | A quota of `accelerators` that this environment shares with others, which hold `heldElsewhere` of it |

Every container holds the `gpuCount` accelerators of its lane's container
shape. `total` and each of `types` must hold every lane they cover at its
container ceiling, `maxContainers` containers, at once, and so at its envelope
maximum, `floorMax` containers, too. `sharedQuota` bounds the envelope maxima,
the most a tuner may hold warm, together with `heldElsewhere`. A lane with
fallback cells counts against the ceiling of every product it names, and once
against `total`.

## Several models on one GPU

A lane can hold several identities on one card: some `pinned`, always
resident, and the rest `lazy-resident`, loaded on demand and evicted least
recently used. The planner decides this from measurements only:

- `footprintMiB`, on a receipt or on a boot record that succeeded, is the GPU
  memory the identity holds once loaded: the memory in use after its load less
  the memory in use before it. The planner takes the largest an identity
  records on a cell.
- `workingMemoryMiB`, on a receipt, is the GPU memory its run needed at the
  receipt's concurrency beyond the footprint. The planner takes the largest an
  identity records at a cell, container shape and concurrency.
- `coResidents`, on a receipt, names the identities pinned and loaded on the
  same container while it was measured, which makes it a co-residency receipt.
  A receipt without `coResidents` measured the identity alone and is a cell;
  a co-residency receipt is not.
- `gpuMemoryMiB`, on a price-list machine profile, is the memory of one of its
  cards.

A lane fits its residents when the footprints of its pinned identities, the
largest footprint of its lazy residents and the largest working memory any
resident needs at the lane's concurrency come to at most 80% of `gpuMemoryMiB`.
The planner puts an identity beside another only on a lane that fits them, of
one card per container and without fallback cells, on a profile that states
`gpuMemoryMiB`. A pinned identity beside other pinned identities is judged by a
co-residency receipt at the lane's cell, container shape and concurrency rather
than by a receipt of it alone: one whose `coResidents` include every other
identity pinned on the lane, since a receipt measured beside more residents
stands for fewer. Its warm bounds are checked and its demand is priced at that
receipt. The planner never places a generation identity, one that declares
`ttftP99Ms` and `itlP99Ms` or has a generate receipt, or an identity that the
`unshared` parameter names, beside another identity, nor another identity on a
lane that hosts one. Such an identity keeps a current lane it already shares.

Every lane that an identity joins in a run must fit all of its residents in
the plan that run writes, not only each joining identity beside the residents
it had before. Where the identities that join a lane do not fit it together,
they are admitted first fit in decreasing order of footprint, ties broken by
identity, and the others are decided again without that lane.

Where a resident's footprint or working memory is not on file, the planner
asks that resident for it, under its own identity and warmth at the lane, in a
`footprint` measurement request. A lane on a profile without `gpuMemoryMiB`
hosts no identity beside another, and the report names that profile.

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
    capacity=capacity,
)
print(render(result.plan.to_dict()), render(result.report.to_dict()))
```

`capacity` is optional. A plan whose lanes exceed a capacity ceiling, as
[Capacity](#capacity) states, is not returned: `plan` raises `CapacityExceededError`, a `PlannerInputError`
whose message names each exceeded ceiling and the lanes that hold it, whose
`exceeded` lists those ceilings, and whose `result` carries the refused plan
and its report.

`plan` is a pure function of its documents. The same inputs render to the
same bytes in whatever order the receipts, boot evidence, identities and demand
series arrive. An identity that stays on its current placement keeps its
current lane as it is, so a run whose `current_plan` already holds every
choice, such as its own previous plan, reports no change.

| Parameter | Meaning |
| --- | --- |
| `placement` | The price-list placement whose region multiplier applies |
| `max_containers` | The reviewed ceiling of every lane, its `maxContainers` |
| `fallback_floor` | The least containers an active zero-floor lane holds, the `sie-cluster` chart's `autoscaling.fallbackReplicas` |
| `cooldown_s` | How long an active lane keeps its containers after its last request, the chart's `autoscaling.cooldownPeriod` |
| `buffer_containers` | The declared warm buffer above a lane's floor, `0` by default |
| `hold_budget_s` | The longest a request for a cold lane is held for it before it is refused, `0` by default. An identity whose `comeAliveP95S` is at most this is served by holding its requests, and avoids unreliable GPU products |
| `hours_per_month` | The hours a monthly figure covers, `730` by default |
| `queue_pool`, `bundle`, `bundles` | Every lane's queue pool, and each identity's bundle: its `bundles` entry, otherwise `bundle` |
| `unshared` | Identities that never share a lane: none is packed or made a lazy resident beside another identity, and no other identity joins a lane one of them is on. Empty by default |

### Selection

The planner chooses a cell and warmth for each identity of the service-level
document:

1. It refuses a receipt or boot record whose machine profile has no price, or
   whose realized GPU product differs from the product the price list gives
   that profile, and a receipt whose latency cannot answer the identity's
   service level. The report lists every refusal.
2. Every other receipt is a cell. A cell whose p99 misses `warmP99Ms`, or
   `ttftP99Ms` or `itlP99Ms`, is blocked. An identity that declares none of
   them has no bound to choose a class by: every cell outside its current
   class, the cell of its current lane, is blocked by the undeclared bound, and
   an identity without a current lane has no unblocked cell.
3. A cell offers `pinned`, and `cold-with-bound` or `snapshot-cold` where boot
   evidence for the identity, cell and container shape exists. A cold
   warmth's come-alive is its placement wait plus its boot, each taken at the
   same percentile. The boot is taken over the identity's attempts at that
   cell, container shape and warmth that got a container. The placement wait
   is taken over every cold attempt on the cell's GPU product, whatever the
   identity. An attempt that got a container without recording
   `placementWaitS` waited none, and a product none of whose cold attempts
   records `placementWaitS` or was `unplaced` adds no wait. Below 20 attempts
   the slowest stands for the percentile; from 20, their 95th percentile by
   nearest rank does, so one slow attempt stops deciding once enough exist. An
   attempt that did not succeed, a failed or timed-out boot or an unplaced
   placement, ranks above every success, so the warmth is withdrawn when the
   percentile falls on one.
   The environment's own attempts replace borrowed ones, per cell, container
   shape and warmth for the boot and per GPU product for the placement wait. A
   `comeAliveP95S` of `0` admits `pinned` only.
4. A cell also offers `lazy-resident` on each host lane with its cell,
   container shape and concurrency. A host lane is a lane of the current plan
   with a `floorMin` of at least 1, on the parameters' queue pool and the
   identity's bundle, on which another identity whose `comeAliveP95S` is `0`
   stays pinned in this run, on its current placement or kept there. Those
   identities are placed first: a bound of `0` admits no cold or lazy option,
   so no other choice moves them. The identity's own current lane is also a
   host when it is a lazy resident there already. The come-alive is the
   identity's `lazy-resident` boot evidence, its measured load onto that cell
   and container shape, under the same statistic, with no placement wait,
   since the host's container is already placed. The option costs what the
   identity's demand needs above the host lane's floor, plus the floor itself
   on a lane where no such identity stays pinned, since no other identity pays
   it. The host must fit the identity beside the residents of its current
   lane, as [Several models on one GPU](#several-models-on-one-gpu) states;
   where a footprint or a working memory is not on file the option is a
   measurement request for it, asked of each resident that lacks it. A lane
   with fallback cells, or one that hosts a generation or `unshared` identity,
   hosts no lazy resident, and neither identity is ever one.
5. A GPU product is unreliable when more than 5% of the cold attempts on it
   were `unplaced`, counting the environment's own attempts in place of
   borrowed ones. For an identity whose `comeAliveP95S` is at most
   `hold_budget_s`, every option on an unreliable product is blocked by
   `placementFailureRate`. Such an identity may still be pinned on an
   unreliable product through a candidate list: a lane whose fallback cell is
   on another product no cheaper than the first, where the identity has a
   receipt meeting its warm bounds at the same container shape and
   concurrency on both. The list is priced at the dearer product and at the
   slower of the two receipts.
6. The cheapest unblocked option wins. Ties go to the current placement,
   then to the faster come-alive, then to a lane without fallback cells.

The current placement is the identity's current lane when that lane has the
option's cell and fallback cells, container shape and concurrency, the
identity's warmth is the option's, and the lane's `queuePool` and `bundle` are
the ones the parameters give the identity; a `lazy-resident` option must also be
hosted by that lane, and a `pinned` option on a lane where other identities are
pinned must stand on a co-residency receipt beside them. A current placement
that no option matches lacks evidence: a receipt at its lane's cell, container
shape and concurrency, a co-residency receipt beside the identities pinned with
it, a boot for a cold warmth, or a load for a lazy resident. Unless it is known
to miss a bound, through a receipt at the lane that misses a warm bound, a
measured come-alive over the bound, a boot or load at the lane that did not
succeed, a cold or `lazy-resident` warmth under a bound of `0`, or an unreliable
product without a fallback cell on another product under a bound `hold_budget_s`
fits, the identity keeps that placement and leaves it only for an unblocked
option at least a cent cheaper than the placement. The placement is priced as an
option of its warmth at the lane's fastest receipt, or without one at its floor
alone: one container all month for `pinned`, for `lazy-resident` nothing beyond
the lane's floor, or the floor itself where no identity whose `comeAliveP95S` is
`0` stays pinned on the lane, and the burst tails for a cold warmth. What the
placement lacks becomes a measurement request. Such a placement on a lane the
price list cannot price has no cost to compare, so the identity keeps it.

An identity that stays on its current placement keeps that lane exactly as it
is, beside whoever else the lane hosts: its cell, container shape,
concurrency and envelope, capped by the reviewed ceiling. A lazy resident
joins its host lane the same way. Every other placed identity gets a new lane
of its own, named from the identity and its machine profile, with a digest of
the identity added when that name is taken. The planner does not select
`bridged` or `scheduled-warm`. An identity with no unblocked option keeps its
current lane, capped by the reviewed ceiling, or stays out of the plan when it
has none.

### Sharing a lane

Once every identity has an option, the planner prices the plan as a whole and
changes it in the steps below. A floor is paid once per lane, so an option an
identity is offered alone can look cheaper than the lane it shares while the
plan total does not fall.

1. A pinned identity whose co-pinned neighbours change in this run is decided
   again beside the neighbours it is left with, until no identity's
   neighbours change.
2. Identities that left a current lane return to it, one at a time and then
   together, wherever their return does not raise the plan total. An identity
   whose current placement is known to miss a bound does not return.
3. The identities that joined a lane are held to fitting it together, as
   [Several models on one GPU](#several-models-on-one-gpu) states, and step 1
   runs again for every identity decided again.
4. Packing. The planner visits every placed identity that is alone on its lane
   and is neither a generation nor an `unshared` identity, in decreasing order
   of its lane's floor cost, ties broken by identity. For each it tries the
   other lanes of the plan on its queue pool and bundle that keep a floor and
   host a pinned identity, where it has a receipt at the lane's cell, container
   shape and concurrency: first the lanes on which an identity whose
   `comeAliveP95S` is `0` is pinned, since nothing moves that identity, then in
   decreasing order of the footprint already on them, then by name. On the
   first lane where a move fits and lowers the plan total by at least a cent,
   the identity joins it, as a lazy resident when an identity whose
   `comeAliveP95S` is `0` is pinned on the lane, its bound admits its measured
   load onto the lane and selection did not already offer it the lane,
   otherwise pinned, which needs every pinned identity's co-residency receipt
   beside the others to meet its warm bounds. A move that would lower the total
   but lacks a footprint, a working memory, a load or a co-residency receipt is
   not made: it is a measurement request whose `savingBound` is that fall.
5. Steps 1 and 3 run again for the lanes packing changed.

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
cells, container shape and concurrency, under that lane's own ceiling. A lane
with fallback cells costs the hourly rate of its dearest cell, and a candidate
list's demand is priced at the slower of its two receipts.

### Envelope

A new lane's `floorMin` is 1 for `pinned` and 0 otherwise. No source measures
queue wait yet, so `floorMax` is `floorMin` plus `buffer_containers`, capped by
the ceiling. A floor raised on a zero-floor lane realizes as at least
`fallback_floor`, so a zero-floor lane's nonzero `floorMax` is at least
`fallback_floor` capped by the ceiling. The envelope carries the floor range
only. A lane carried from the current plan keeps its envelope, capped by the
ceiling. Given a `capacity` document, the lanes must fit each of its ceilings,
at their container ceilings and envelope maxima as [Capacity](#capacity)
states, or the plan is refused as described above. A
loosened service level whose plan the ceilings would refuse saves nothing in
the report.

### Report

The report renders as a `placement-report` document:

- `totals`: the monthly cost of every lane of the plan, the monthly cost of
  their `floorMax`, and `unpricedLanes`, the lanes whose cost the totals leave
  out in whole or in part.
- `identities`: each identity's status (`placed`, `kept` or `unplaced`),
  effective service level and demand, its lane and `monthlyCost`, what its
  demand costs there with the lane's floor, the chosen cell, warmth, monthly
  cost and come-alive with its basis (`warm-floor`, `boot-evidence` or
  `borrowed-boot-evidence`), the `placementWaitS` part of a cold come-alive
  where a placement wait was measured, the `fallbackCells` of a candidate list,
  the `coResidents` of the co-residency receipt a co-pinned identity is judged
  by and, for a resident that joins another identity's lane, its `lane`, the
  peak containers its demand
  needs, and `nextCheaper`: the nearest option cheaper than the choice, or than
  a kept identity's current placement, that a service level blocks, the bounds
  that block it, and `saving`, the fall in the plan total found by planning
  again from this plan with each of those bounds loosened to what the option
  measured, rounded up to a whole second or millisecond and never below 1. A
  bound the identity does not declare blocks with no `bound`, and loosening it
  declares it. A `placementFailureRate` blocker, whose `bound` is `0.05` and
  `measured` the product's failure rate, is loosened by raising `comeAliveP95S`
  to one second above `hold_budget_s`. The loosened bounds may admit a cheaper
  option still, and the saving counts it. The saving is 0 when the total would
  not fall, as for an identity whose lane keeps its floor for its other
  residents. An identity without a placement names its `unplacedReason`:
  `no-receipt`, `warm-bound-missed`, `warm-bound-undeclared` when no cell is on
  its current class, `placement-unreliable` when its options are on unreliable
  products only, `current-unmeasured` when it keeps a current placement that
  lacks evidence and no unblocked option is cheaper, or `current-unpriced` when
  it keeps a current placement on a lane the price list cannot price. A kept
  identity's `blockedBy` names the bounds its kept placement is known to miss:
  its receipt's warm latencies, a measured come-alive over the bound, an
  unreliable product, and with no `measured` value a boot or load at its lane
  that did not succeed, or a cold or `lazy-resident` warmth under a bound of
  `0`, which needs no measurement.
- `capacity`: for each ceiling of the `capacity` document, its `scope`
  (`total`, `type` with its `gpuProduct`, or `shared-quota` with
  `heldElsewhere`), the `ceiling`, the `accelerators` the plan's envelope
  maxima hold against it, for `total` and `type` the `peakAccelerators` its
  lanes hold at `maxContainers`, the `lanes` that hold them, and whether it is
  `exceeded`.
- `lanes`: each lane's hourly cost, its monthly cost, its floor's monthly cost
  and the monthly cost of its `floorMax`, which a floor budget prices. A lane
  on a cell the price list cannot price carries `unpricedReason` and no cost,
  and `unpricedIdentities` lists residents whose demand has no receipt to price
  it on the lane. A lane that hosts more than one identity names the GPU memory
  its residents need, `memoryMiB`, and the 80% of its card's memory they may
  use, `memoryLimitMiB`, where both are known.
- `sharedChartLanes`: every chart lane, by `queuePool`, `machineProfile` and
  `bundle`, that more than one plan lane names.
- `measurementRequests`: options that lack evidence, ranked by the saving they
  could unlock. A profile without a receipt is bounded by price alone: one
  accelerator at its rate, kept warm or, where the come-alive bound allows,
  paying burst tails only. A profile on an unreliable product is not requested
  for an identity whose bound `hold_budget_s` fits. A cell whose receipt has no
  successful boot asks for boot evidence and is bounded by its cold cost, with
  reason `failed`, or `unplaced` when no attempt got a container or the
  product's placement wait falls on an unplaced attempt, and a host lane without
  the identity's load asks for `lazy-resident` boot evidence on it, bounded by
  what that option would cost, with `footprint` asked of each resident whose
  footprint or working memory is not on file, under that resident's identity
  and warmth. A packing that lacks evidence asks at the lane it would join for
  `boot-evidence`, `footprint` or `co-residency-receipt`, with the
  `coResidents` the receipt is to be measured beside, bounded by the fall in
  the plan total the packing would bring, and asks a resident there for its
  own missing `footprint` the same way. A kept identity pinned beside others
  asks at its lane for a `co-residency-receipt` beside them, and a kept lazy
  resident asks each resident of its lane, itself included, for a missing
  `footprint`. A request repeated for the same identity, cell, container
  shape, concurrency, lane, measurements, warmth and `coResidents` is listed
  once, at its highest rank. Savings are measured against the choice, or
  against a kept identity's current placement. A request that cannot save is dropped, and the
  requests of an identity without a placement or a current one carry no bound
  and rank first. The evidence a kept current placement lacks is requested at
  its lane, named in `lane` with its `concurrency`, and also ranks first without
  a bound; a profile it asks for there is not asked for again.
- `comeAliveLevers`: cold and `lazy-resident` warmths whose measured come-alive
  misses the bound on a cell that meets the warm bounds for less than the
  choice, or than a kept identity's current placement, listed only when
  loosening the identity's bound to `permittedBoundS`, the bound the measured
  come-alive would permit, lowers the plan total. Both savings are found by
  planning again from this plan and taking the fall in its total:
  `savingAtPermittedBound` with the bound loosened, and `savingUnderBound` with
  the warmth's boot measured at the identity's bound less its placement wait,
  which is 0 under a bound of 0, since no boot comes under it, when the
  placement wait alone exceeds the bound, and when the total would not fall. A
  lever names the `placementWaitS` part of its come-alive where one was
  measured. An identity that shares its lane saves nothing of the lane's floor
  by leaving, since the other residents keep it. Levers rank by the saving under
  the bound, then by the saving at the permitted bound.
- `scarcity`: every option cheaper than an identity's choice, or than a kept
  identity's current placement, that scarcity alone blocks, so scarcity is
  part of why the identity stays warm or pays more: `placement-wait`, a cold
  warmth whose boot fits the come-alive bound and whose boot plus placement
  wait does not; `unplaced`, a cold warmth whose boot fits the bound and whose
  product's placement wait falls on an unplaced attempt; and
  `unreliable-type`, an option on an unreliable product under a bound
  `hold_budget_s` fits. Each names the option's cell, container shape, warmth
  and monthly cost, the bound, the boot and placement wait it measured, and
  the product's `placementFailureRate`.
- `queueWaitUnattained`: every identity that declares `queueWaitP99Ms`. No
  plan claims that bound as met.
- `warmBoundUndeclared`: every identity that has a cell and declares neither
  `warmP99Ms` nor `ttftP99Ms` and `itlP99Ms`, so keeps its current class.
- `gpuMemoryUndeclared`: every machine profile without `gpuMemoryMiB` on which
  the planner weighed an identity beside another, so its lanes host no
  identity beside another and a lane that already shares one is not checked.
- `refusals`, and `changes`: the identities and lanes that differ from
  `current_plan`.

## Tests

```bash
mise run test -- packages/sie_placement/tests
```
