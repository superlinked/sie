# sie-placement

Versioned documents for a measured-only placement planner: the service level
each served identity must meet, the measurements a placement may be chosen
from, the prices those placements cost, the demand they serve, and the plan
that results, together with the planner that turns them into a plan and a cost
report. The package uses only the Python standard library, except the KEDA
applier, whose `keda` extra adds the SIE Python SDK. It is a workspace package
and is not published to PyPI.

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

## Tuner

`sie_placement.tuner.step(envelope, observation, current)` decides one tick of
an online tuner for one lane. It is a pure function, so every applier calls the
same decision. The `Observation` gives the containers running on the lane, the
requests waiting for capacity, the requests in flight and an `attainment` of
`met`, `breached` or `unmeasured`. An attainment given as its wire string reads
as that value, and any other value is rejected. `current` is the `Settings` in
force on the lane: the floor, and optionally the buffer and the scale-down
window. The result is a `Step`, one action and the settings to apply.

An applier reads `current` on every tick from the platform that scales the
lane, or records it again whenever a deploy or another writer resets those
settings. It never feeds the previous `Step.settings` back in as `current`
across such a reset, because the step then steers from settings the lane no
longer has. It applies `Step.settings` whenever they differ from `current`,
whatever the action. The action labels the decision for the audit record.

| Action | When | Settings |
| --- | --- | --- |
| `raise_to_floor` | The current floor is below `floorMin` | Floor `floorMin`, and a buffer or window outside its range brought inside it |
| `hold_at_cap` | The current floor is above `floorMax`, or the lane is under pressure and every tuned setting is at its maximum | Floor `floorMax` if it was above it, and a buffer or window outside its range brought inside it. Otherwise unchanged |
| `scale_up` | Requests wait or the service level is breached, and the lane runs at least its current floor | Floor and buffer one container higher and window halfway to its maximum, each up to the top of its range |
| `scale_down` | Nothing waits, nothing is in flight and the service level is not breached | Floor and buffer one container lower and window halfway to its minimum, each down to the bottom of its range |
| `hold` | Otherwise, including a lane under pressure that has not yet started its current floor | Unchanged, except that a buffer or window outside its range is brought inside it |

Every returned setting lies inside the envelope. `hold` and `hold_at_cap`
return settings other than `current` only when `current` lay outside it. The
buffer and the window are tuned only when the envelope declares their range
and `current` carries a value, and otherwise pass through unchanged, so an
applier that moves only the floor passes `Settings(floor=n)`. The floor never
depends on the buffer or the window. An `unmeasured` attainment steers on the
queue alone, as `met` does.

A deploy that resets the applied settings to the envelope minimum is a fresh
start: read back as `current`, the settings climb one step per tick under
steady demand until the lane serves the demand or reaches the top of its
envelope.

## KEDA applier

`sie-placement tune` keeps each plan lane's warm floor inside its envelope on a
`sie-cluster` deployment that autoscales workers with KEDA. It needs the `keda`
extra.

```bash
sie-placement tune --plan plan.json \
  --prometheus-url http://prometheus:9090 \
  --gateway-url http://sie-gateway:8080 \
  --api-key-file /run/secrets/gateway-client-token
```

Every tick, 60 seconds apart by default or once with `--once`, it reads the
four lane series below from Prometheus, reads each lane's floor-carrier pool
from the gateway with `SIEClient.get_pool`, and calls `step` with the carrier's
floor as `current`. It prints one JSON audit row per lane to standard output. Without
`--apply` it runs in shadow mode: it decides and records and writes nothing.
With `--apply` it upserts every lane's carrier on every tick through
`SIEClient.create_pool`, which renews the carrier's lease and sets its floor to
the step's floor. Each write uses its own SDK client and closes it after the
write, so the SDK's background lease renewal never runs: a carrier's lease is
renewed only by a tick that writes it. A lane that has no carrier and whose
floor stays at 0 gets none.

Under KEDA only the floor moves per lane. The scale-down window and the queue
threshold are chart-wide settings, and there is no buffer to tune, so the
applier passes `Settings(floor=n)` and ignores the envelope's buffer and window
ranges.

### Floor carriers

A floor carrier is an API logical pool named
`placement-floor-<queuePool>-<machineProfile>-<bundle>`, with `queue_pool` and
`bundle` set to the lane's, `gpus` set to `{<machineProfile>: 0}`, the
gateway's default lease, no GPU caps, no pinned models, and
`minimum_worker_count` set to the step's floor. The gateway keeps each lane at
the largest floor among the logical pools on it, so the carrier can raise the
lane above its chart floor and never lowers another pool's floor. The applier
writes only its carriers: a pool that has a carrier's name and any other spec is
reported and left unchanged. A carrier carries no pins, because every floor
change is a spec change, which resets the pool's worker assignment; pinned
models stay in the chart values.

A tick that cannot read a lane's carrier, or finds a pool with its name that is
not its carrier, does not write it. A carrier that no tick writes, because those
reads keep failing or the applier has stopped, lapses 1200 seconds after its
last write by default, the gateway removes it, and the lane returns to its
chart floor. On a lane whose chart `minReplicas` is 0, a floor of 1 or more
activates the lane, which then holds at least the chart's fallback floor.

### Series

| Series | Query | Step input |
| --- | --- | --- |
| Queue depth | `max by (pool, machine_profile, bundle) (sie_gateway_lane_queue_depth{} and on(producer_instance, collector_generation, pool, machine_profile, bundle) (abs(time() - sie_gateway_lane_queue_snapshot_timestamp_seconds{}) < 20))` | `running` |
| Pending demand | `max by (pool, machine_profile, bundle) (sie_gateway_pending_demand{} and on(producer_instance, collector_generation) (abs(time() - sie_gateway_capacity_snapshot_timestamp_seconds{}) < 20))` | `backlog` |
| Warm floor | `max by (pool, machine_profile, bundle) (sie_gateway_pool_warm_floor{} and on(producer_instance, collector_generation) (abs(time() - sie_gateway_capacity_snapshot_timestamp_seconds{}) < 20))` | Reported only |
| Running workers | `count by (lane) (max by (lane, producer_instance) (sie_worker_gpu_slots{state="ready"}) > 0)` | `runners` |

`--series-selector` adds label matchers to every selector, timestamps included,
for example `namespace="sie"` when one Prometheus scrapes several releases.
Lane labels match the plan's lane tokens after trimming and lowering them, as
the chart names lanes.

The collector keeps a producer's last point for minutes after the producer
stops reporting, so each gateway series keeps only the producers whose snapshot
is at most 20 seconds old, as the chart's KEDA triggers do: the lane's own
queue snapshot for queue depth, and the producer's capacity snapshot for
pending demand and warm floor. A lane whose gateway values are all stale reads
as missing. The worker series has no snapshot timestamp, so a worker that
stopped reporting still counts among the runners until the collector drops it.

Pending demand is a 0 or 1 flag per lane, not a count. The gateway sets it
while a request that found no ready capacity is recent, while any request's
durable publish to the lane awaits its acknowledgement, and for 120 seconds
after a failed dispatch. The attainment is `unmeasured`, so a lane is under
pressure whenever the flag is set, and idle when neither the flag nor queued or
in-flight work is reported. Because a publish in flight sets the flag, any
traffic can read as pressure even when the lane has capacity to spare, and a
lane that serves steady traffic seldom reads idle, so its floor can climb
towards `floorMax` while traffic lasts and falls one step per idle tick
afterwards. Run the applier in shadow mode first to see how often each lane
reads as under pressure. A lane with no
ready worker has 0 runners. A lane whose queue depth, pending demand or worker
series is missing is held: the step is given a busy lane with nothing waiting,
so it only brings the carrier's floor inside the envelope.

### Audit rows

Each row carries `at`, `mode` (`shadow` or `apply`), the lane's `lane`,
`queuePool`, `machineProfile` and `bundle`, the `carrier` pool name, the
`telemetry` read for the lane (`pendingDemand` is the 0 or 1 flag),
`currentFloor`, the step's `action` and `floor`, whether the write was
`applied`, `laneFloorAboveEnvelope` and `error`.
`laneFloorAboveEnvelope` is true when the lane's warm floor exceeds `floorMax`.
The applier does not lower another pool's floor, so a lane held above its
envelope by another pool stays reported until that pool changes.

### Refusals

The command exits with status 2 before it calls Prometheus or the gateway when:

- a lane's `floorMax` exceeds `--max-pool-floor`, which must match the gateway's
  `SIE_GATEWAY_POOL_MAX_MINIMUM_WORKER_COUNT` (default 4);
- the plan has more lanes than `--max-pools`, which must match the gateway's
  `SIE_GATEWAY_MAX_POOLS` (default 64);
- a lane's `queuePool`, machine profile or bundle is not 1 to 63 letters,
  digits, `_` or `-`, two lanes need the same carrier, or a carrier name is
  longer than the gateway's 128 characters;
- a token file or `SIE_ADMIN_TOKEN` holds anything but one run of printable
  ASCII characters without spaces after trimming, or the Prometheus URL has an
  invalid port.

A failed query, pool read or pool write fails only its tick, which is recorded,
and the next tick runs as usual.

With `--once`, a tick whose query, pool read or pool write failed exits with
status 1.

### Credentials

| Call | Permission | Source |
| --- | --- | --- |
| Prometheus `/api/v1/query` | Read access to the query API | None sent |
| `GET /v1/pools/{name}` | A gateway client token, when gateway token auth is on | `--api-key-file`, or `SIE_API_KEY` when `SIE_BASE_URL` names the same origin, as the SDK resolves it |
| `POST /v1/pools`, which also renews the lease | The gateway admin token | `--admin-token-file`, or `SIE_ADMIN_TOKEN`, read only with `--apply` |

The gateway admin token authorizes every gateway admin route, not only pools.
The applier reads it only with `--apply`, sends it only on pool writes to
`--gateway-url`, never takes it as a command-line value and never prints or
logs it. Without it, `--apply` sends its writes with no credential, which a
gateway with token auth refuses. The client token is never sent on writes. A
refused token is named by its source, never by its value, and a failed gateway
call is reported by the server's error message or, for a transport failure, by
the exception type alone, so no reported error carries a header value.

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
series arrive. An identity that stays on its current placement keeps its
current lane as it is, so a run whose `current_plan` already holds every
choice, such as its own previous plan, reports no change. The one exception is
a lane that the previous run created for an identity whose `comeAliveP95S` is
`0`: it hosts lazy residents only from the run after it.

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
   `ttftP99Ms` or `itlP99Ms`, is blocked. An identity that declares none of
   them has no bound to choose a class by: every cell outside its current
   class, the cell of its current lane, is blocked by the undeclared bound, and
   an identity without a current lane has no unblocked cell.
3. A cell offers `pinned`, and `cold-with-bound` or `snapshot-cold` where boot
   evidence for the identity, cell and container shape exists. Below 20
   successful boots the slowest stands for the come-alive percentile; from 20,
   their 95th percentile by nearest rank does, so one slow boot stops deciding
   once enough boots exist. A boot that did not succeed withdraws the warmth,
   and the environment's own evidence replaces borrowed evidence. A
   `comeAliveP95S` of `0` admits `pinned` only.
4. A cell also offers `lazy-resident` on each host lane with its cell,
   container shape and concurrency. A host lane is a lane of the current plan
   with a `floorMin` of at least 1, on the parameters' queue pool and the
   identity's bundle, on which another identity whose `comeAliveP95S` is `0` is
   pinned. The identity's own current lane is a host only when it is a lazy
   resident there, since its floor is paid either way. The come-alive is the
   identity's `lazy-resident` boot evidence, its measured load onto that cell
   and container shape, under the same statistic. The option costs what the
   identity's demand needs above the host lane's floor.
5. The cheapest unblocked option wins. Ties go to the current placement,
   then to the faster come-alive.

The current placement is the identity's current lane when that lane has the
option's cell, container shape and concurrency, the identity's warmth is the
option's, and the lane's `queuePool` and `bundle` are the ones the parameters
give the identity; a `lazy-resident` option must also be hosted by that lane.
A current placement that no option matches lacks evidence: a receipt at its
lane's cell, container shape and concurrency, a successful boot for a cold
warmth, or a load for a lazy resident. Unless it is known to miss a bound,
through a receipt at the lane that misses a warm bound, a measured come-alive
over the bound, or a cold or `lazy-resident` warmth under a bound of `0`, the
identity keeps that placement and leaves it only for an unblocked option at
least a cent cheaper than the placement. The placement is priced as an option
of its warmth at the lane's fastest receipt, or without one at its floor alone:
one container all month for `pinned`, nothing beyond the host lane's floor for
`lazy-resident`, and the burst tails for a cold warmth. What the placement
lacks becomes a measurement request.

An identity that stays on its current placement keeps that lane exactly as it
is, beside whoever else the lane hosts: its cell, container shape,
concurrency and envelope, capped by the reviewed ceiling. A lazy resident
joins its host lane the same way. Every other placed identity gets a new lane
of its own, named from the identity and its machine profile, with a digest of
the identity added when that name is taken. The planner does not select
`bridged` or `scheduled-warm`. An identity with no unblocked option keeps its
current lane, capped by the reviewed ceiling, or stays out of the plan when it
has none.

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

A new lane's `floorMin` is 1 for `pinned` and 0 otherwise. No source measures
queue wait yet, so `floorMax` is `floorMin` plus `buffer_containers`, capped by
the ceiling. A floor raised on a zero-floor lane realizes as at least
`fallback_floor`, so a zero-floor lane's nonzero `floorMax` is at least
`fallback_floor` capped by the ceiling. The envelope carries the floor range
only. A lane carried from the current plan keeps its envelope, capped by the
ceiling.

### Report

The report renders as a `placement-report` document:

- `totals`: the monthly cost of every lane of the plan, the monthly cost of
  their `floorMax`, and `unpricedLanes`, the lanes whose cost the totals leave
  out in whole or in part.
- `identities`: each identity's status (`placed`, `kept` or `unplaced`),
  effective service level and demand, its lane and `monthlyCost`, what its
  demand costs there with the lane's floor, the chosen cell, warmth, monthly
  cost and come-alive with its basis (`warm-floor`, `boot-evidence` or
  `borrowed-boot-evidence`) and, for a lazy resident, its host `lane`, the peak
  containers its demand needs, and `nextCheaper`: the nearest option cheaper
  than the choice, or than a kept identity's current placement, that a
  service level blocks, the bounds that block it, and `saving`, the fall in
  the plan total found by planning again from this plan with each of those
  bounds loosened to what the option measured, rounded up to a whole second or
  millisecond and never below 1. A bound the identity does not declare blocks
  with no `bound`, and loosening it declares it. The loosened bounds may admit
  a cheaper option still, and the saving counts it. The saving is 0 when the
  total would not fall, as for an identity whose lane keeps its floor for its
  other residents. An identity without a placement names its
  `unplacedReason`: `no-receipt`, `warm-bound-missed`,
  `warm-bound-undeclared` when no cell is on its current class, or
  `current-unmeasured` when it keeps a current placement that lacks evidence
  and no unblocked option is cheaper. A kept identity's `blockedBy` names the
  bounds its kept placement is known to miss: its receipt's warm latencies, a
  measured come-alive over the bound, or a cold or `lazy-resident` warmth under
  a bound of `0`, which needs no measurement.
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
  boot evidence and is bounded by its cold cost, and a host lane without the
  identity's load asks for `lazy-resident` boot evidence on it, bounded by the
  demand above the host's floor. Savings are measured against the choice, or
  against a kept identity's current placement. A request that cannot save is
  dropped, and the requests of an identity without a placement or a current
  one carry no bound and rank first. The evidence a kept current placement
  lacks is requested at its lane, named in `lane` with its `concurrency`, and
  also ranks first without a bound; a profile it asks for there is not asked
  for again.
- `comeAliveLevers`: cold and `lazy-resident` warmths whose measured
  come-alive misses the bound on a cell that meets the warm bounds for less
  than the choice, or than a kept identity's current placement, listed only
  when loosening the identity's bound to `permittedBoundS`, the bound the
  measured come-alive would permit, lowers the plan total. Both savings are
  found by planning again from this plan and taking the fall in its total:
  `savingAtPermittedBound` with the bound loosened, and `savingUnderBound` with
  the warmth's boot measured at the identity's bound, which is 0 under a bound
  of 0, since no boot comes under it, and when the total would not fall. An
  identity that shares its lane saves nothing of the lane's floor by leaving,
  since the other residents keep it. Levers rank by the saving under the bound,
  then by the saving at the permitted bound.
- `queueWaitUnattained`: every identity that declares `queueWaitP99Ms`. No
  plan claims that bound as met.
- `warmBoundUndeclared`: every identity that has a cell and declares neither
  `warmP99Ms` nor `ttftP99Ms` and `itlP99Ms`, so keeps its current class.
- `refusals`, and `changes`: the identities and lanes that differ from
  `current_plan`.

## Helm values

`helm_values` turns a plan into a values overlay for the `sie-cluster` Helm
chart, to pass with `-f` after the release's own values:

```python
import json

from sie_placement.helm import ChartParameters, helm_values

with open("placement-values.json", "w", encoding="utf-8") as handle:
    json.dump(helm_values(plan, ChartParameters(fallback_replicas=2)), handle, indent=2)
```

| Parameter | Meaning |
| --- | --- |
| `fallback_replicas` | The chart's `autoscaling.fallbackReplicas`, `2` by default |
| `bundled_dependencies` | Whether the release installs KEDA or Prometheus itself (`keda.install` or `kube-prometheus-stack.install`), which lowers the chart's lane limit from 192 to 96 |
| `base_pools` | The `workers.pools` of the values beneath the overlay: the chart's values merged with the release's own values files. Only each worker group's `enabled` and its bundles' `enabled` are read. By default the chart's own worker groups, `chart_pools()` |
| `device_groups` | Lowercase machine profile to `true` when a container of more than one GPU on it is one device group, a single worker that owns every GPU as a model too large for one accelerator needs, or `false` when it is one single-GPU worker per GPU |

A plan lane names the chart lane of its `queuePool`, `machineProfile` and
`bundle`, canonicalized as the chart does: trimmed, at most 63 characters of
`[A-Za-z0-9_-]`, and lowercased. Each plan lane is a chart lane of its own. The
overlay sets:

- `workers.pools.<machineProfile>`: one enabled worker group per machine
  profile, with its `machineProfile` and `queuePool`, `gpu.count` and
  `gpu.product` from the cell and container shape, `gpu.deviceGroup`, and
  `resources` requests and limits equal to the container shape. Each chart
  lane is an enabled `bundles.<bundle>` entry whose `minReplicas` is the
  envelope's `floorMin` and whose `maxReplicas` is `maxContainers`. Every other
  bundle that `base_pools` enables under the group is set to
  `enabled: false`, so the release runs no lane the plan does not place.
  `gpu.deviceGroup` is `false` for a single GPU and the `device_groups` entry
  of the machine profile for more.
- `queueRouting.staticQueuePools.<queuePool>`: one static queue pool for every
  queue pool other than `default`, which the gateway needs before it admits
  that queue's workers. `gpus` lists every machine profile on the queue,
  `minimumWorkerCount` is the lanes' `floorMin`, `pinnedModels` are their
  `pinned` identities, and `bundle` names the lanes' bundle when they share
  one.

It refuses, with an `UnrealizablePlanError` that names the plan field, a plan
the chart and the gateway would not realize as planned:

- A lane token the chart rejects, or a machine profile or bundle that is not
  a lowercase DNS-1123 label, which the chart requires of worker group and
  bundle keys.
- A lane with `floorMin` 0 whose `floorMax` is above 0 and below the chart's
  fallback floor, `min(maxContainers, fallbackReplicas)`, which is the least
  KEDA keeps on an active zero-floor lane.
- Two plan lanes on one chart lane. The chart refuses two entries on one
  canonical lane, and a chart lane realizes one floor, one ceiling and one
  pinned set, so the two lanes would share what the plan prices separately.
- Lanes on one machine profile that differ in queue pool, GPU product or
  container shape.
- A container of more than one GPU on a machine profile that
  `device_groups` does not name.
- More chart lanes than the chart admits, counting the lanes of the worker
  groups `base_pools` enables and the overlay leaves out: 192 with external
  KEDA and Prometheus, 96 when the release installs either.
- A `pinned` identity on the `default` queue pool, for which the gateway keeps
  no static queue pool.
- Lanes on one named queue pool that differ in `floorMin` or in their pinned
  identities, or that carry a floor or pins across more than one bundle. A
  static queue pool applies one floor and one pinned set to every worker it
  admits.
- The queue pool `_default`, which the gateway reserves.

The chart's queue-depth threshold, cooldown and scale-down stabilization are
chart-wide, so a lane's `concurrency`, `floorMax` and buffer or scale-down
window ranges are not part of the overlay. Helm merges the overlay into the
values beneath it: a worker group that those values enable and the plan does
not use stays enabled, and the worker group keys the overlay leaves out, such
as `gpuType`, `nodeSelector`, `tolerations` and `platform`, keep their values
there. No chart template reads `gpu.product`, so the overlay does not
constrain where a worker group schedules. The values beneath it must select
nodes with the cell's GPU for every machine profile the plan places, as the
cloud values files do with each worker group's `nodeSelector`. Requests reach
a lane on a named queue pool through that pool, for example
`gpu="<queuePool>/<machineProfile>"`.

These rules bound which plans emit. A worker group carries one queue pool, a
named queue pool carries one floor and one pinned set, and the `default` queue
pool carries no pins, so a pinned lane needs a named queue pool and a machine
profile of its own. The planner places each identity on a lane of its own, all
on the queue pool its parameters name, `default` unless set. Its plan
therefore emits only when no two identities share a machine profile and
bundle, and it pins nothing, or its queue pool is named and it has one lane.

## Tests

```bash
mise run test -- packages/sie_placement/tests
```
