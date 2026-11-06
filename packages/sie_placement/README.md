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
refers to its lane by that name. The chart compares the three tokens without
regard to case. Two plan lanes may carry the same three tokens, and so name one
chart lane, which cannot give them separate envelopes or concurrency; the
planner's report lists every such chart lane.

The planner gives each lane that pins at least one identity a named queue pool
of its own, since the chart carries a lane's pinned set on a static queue pool
and the default queue pool carries no pins. A lane the current plan already
puts on a named queue pool keeps it, so planning again leaves the pool as it
is, unless the pool is `default`, the parameters' `queue_pool` or a pool
another lane keeps, compared without regard to case as the chart compares them.
Otherwise the pool is derived from the lane's `name`: its slug, which
lowercases the name, replaces each run of characters outside `a-z` and `0-9`
with a hyphen and drops leading and trailing hyphens, or `lane` when nothing
remains. The slug is cut to 63 characters with a digest of the name appended
when it is longer or when another lane of the plan, `default`, or the
parameters' `queue_pool` already has it. A lane on which every identity is
lazy-resident or cold is on the parameters' `queue_pool`. Identities that share
a lane share its queue pool.

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
container ceiling, `maxContainers` containers, at once, as a deployment
compiler that enforces the ceiling charges it, and so at its envelope maximum,
`floorMax` containers, too. `sharedQuota` must hold every lane at its envelope
maximum, the most a tuner may hold warm, together with its `heldElsewhere`
accelerators. Warm capacity is held continuously, so it must fit the quota the
environments share. A burst above the floors is bounded by each environment's
own `total` and `types`, environments that share a quota rarely burst
together, and a burst that meets the shared quota shows as a measured
placement wait, which the planner adds to a cold come-alive. A lane with
fallback cells counts against the ceiling of every product it names, and once
against `total` and `sharedQuota`. The planner never lowers a lane's
`maxContainers` or `floorMax` to fit a capacity document: a plan that does not
fit is refused whole.

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

Where an identity would join a lane and a resident's footprint or working
memory is not on file, the planner asks that resident for it, under its own
identity and warmth at the lane, in a `footprint` measurement request. For a
lane the plan keeps, it asks only when a lazy resident is placed there. A kept
lane whose pinned residents have no footprint on file gets no `footprint`
request, so the report does not name every unmeasured shared lane. A profile
without `gpuMemoryMiB` takes no new identity beside another, and the report
names that profile. A lane the plan already shares on such a profile stays
shared.

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
- the `sie-cluster` chart cannot run the plan: a lane has fallback cells, a lane
  on the `default` queue pool pins an identity, whose pins no static queue pool
  carries, or an identity is `bridged`, `snapshot-cold` or `scheduled-warm`,
  warmths the chart does not realize. These are `plan_refusals` of
  `sie_placement.helm`, the refusals of `helm_values` that hold whatever values
  lie beneath the overlay, reported in its words;
- a token file or `SIE_ADMIN_TOKEN` holds anything but one run of printable
  ASCII characters without spaces after trimming;
- `--prometheus-url` or `--gateway-url` is not a valid absolute http or https
  URL with a host, embeds credentials, carries a query or a fragment, or has an
  invalid port. The refusal does not repeat the URL.

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
logs it. Its gateway client reads no settings from the environment: it ignores
`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` and `NO_PROXY`, so a proxy configured
for other traffic never receives either token. It still verifies an https
gateway against the CA certificates that `SSL_CERT_FILE` or `SSL_CERT_DIR`
names, as httpx does by default, and refuses before any call a bundle it
cannot load. Prometheus queries, which carry no credential, follow the
usual proxy variables. Without it, `--apply` sends its writes with no credential, which a
gateway with token auth refuses. The client token is never sent on writes. A
refused token is named by its source, never by its value. A gateway call that
gets an error reply is reported by its HTTP status alone, and so is a reply
that is neither a success nor an error, such as a redirect; a success reply the
SDK cannot parse is reported as an unexpected response. No text a server
returns reaches an audit row, so an intermediary that quotes the request's
`Authorization` header in its error body cannot put the token there. A failed
connection is reported by the SDK's connection error message, sent before any
header, and any other transport failure by its exception type alone. A message
the SDK raises before any reply is cut to 512 characters, with the token the
client holds replaced by `[redacted]`.

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
| `queue_pool`, `bundle`, `bundles` | The queue pool of every lane that pins no identity, and each identity's bundle: its `bundles` entry, otherwise `bundle` |
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
   with a `floorMin` of at least 1, on the identity's bundle, on which another identity whose `comeAliveP95S` is `0`
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
identity's warmth is the option's, and the lane's `bundle` is the one the
parameters give the identity; a `lazy-resident` option must also be
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
   other lanes of the plan on its bundle that keep a floor and
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
   not made: it is a measurement request whose `savingBound` is that fall. A
   move after which the lane's residents together need more containers than
   its `maxContainers` in some hour is not made either.
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

A lane costs its floor all month once, whoever it hosts, plus what its
residents' demand needs beyond the floor. On a lane with a floor, that is the
containers an hour needs above it for the residents together: each resident's
requests in the hour take the share of a container that its priced receipt's
rate gives them, and the shares are summed, since residents scale together and
share the card under load. On a zero-floor lane, it is the work and the tails a
cold warmth pays for each resident. A placed identity's demand is priced at its
chosen receipt, and a kept identity's at the slowest receipt it has for its
lane's cells, container shape and concurrency, under that lane's own ceiling. A
lane with fallback cells costs the hourly rate of its dearest cell, and a
candidate list's demand is priced at the slower of its two receipts.

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
  maxima hold against it, with `heldElsewhere` for `shared-quota`, for `total`
  and `type` the `peakAccelerators` its lanes hold at `maxContainers`, the
  `lanes` that hold them, and whether it is `exceeded`. `total` and `type` are
  charged at `peakAccelerators`, and `shared-quota` at `accelerators`.
- `lanes`: each lane's hourly cost, its monthly cost, its floor's monthly cost
  and the monthly cost of its `floorMax`, which a floor budget prices. A lane
  on a cell the price list cannot price carries `unpricedReason` and no cost,
  and `unpricedIdentities` lists residents whose demand has no receipt to price
  it on the lane. A lane that hosts more than one identity names the GPU memory
  its residents need, `memoryMiB`, and the 80% of its card's memory they may
  use, `memoryLimitMiB`, where both are known.
- `sharedChartLanes`: every chart lane, by `queuePool`, `machineProfile` and
  `bundle` in lowercase, that more than one plan lane names.
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

## Helm values

`helm_values` turns a plan, such as the `result.plan` of the planner example
above, into a values overlay for the `sie-cluster` Helm chart, to pass with
`-f` after the release's own values:

```python
import json

from sie_placement.helm import ChartParameters, helm_values

with open("placement-values.json", "w", encoding="utf-8") as handle:
    json.dump(helm_values(result.plan, ChartParameters(fallback_replicas=2)), handle, indent=2)
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
  bundle that `base_pools` enables under the group is set to `enabled: false`,
  so a worker group the overlay emits runs no lane the plan does not place on
  it. A worker group the plan does not use is not in the overlay and keeps the
  values beneath it, so the lanes they enable there still run outside the plan.
  `gpu.deviceGroup` is `false` for a single GPU and the `device_groups` entry
  of the machine profile for more.
- `queueRouting.staticQueuePools.<queuePool>`: one static queue pool for every
  queue pool other than `default`, which the gateway needs before it admits
  that queue's workers. `gpus` lists every machine profile on the queue,
  `minimumWorkerCount` is the lanes' `floorMin`, `pinnedModels` are their
  `pinned` identities, however many share a lane, and `bundle` names the
  lanes' bundle when they share one. A `lazy-resident` identity is not pinned:
  the workers of its lane load it on demand.

It refuses, with an `UnrealizablePlanError` that names the plan field, a plan
the chart and the gateway would not realize as planned:

- A lane with `fallbackCells`. A worker group runs on one machine profile and
  GPU product and has no ordered candidate list, so the chart would place the
  lane on its `cell` only.
- An identity whose warmth is `bridged`, `snapshot-cold` or `scheduled-warm`.
  The overlay realizes `pinned` through a static queue pool's `pinnedModels`
  and the lane's floor, `lazy-resident` through a warm lane whose workers load
  the identity on demand, and `cold-with-bound` through a zero-floor lane that
  scales from zero. It configures no remote bridge, and the chart restores no
  snapshot and keeps no warm window.
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
profile of its own, and two pinned lanes on one GPU product need two machine
profiles. Such a lane may pin several identities and host lazy residents beside
them. The planner gives every lane that pins an identity a named queue pool of
its own, places every other lane on the queue pool its parameters name,
`default` unless set, and may place several identities on one lane. Its plan
therefore emits only when no two lanes share a machine profile and bundle, no
lane that pins an identity shares its machine profile with another lane, no
lane has fallback cells, no identity is `snapshot-cold`, no zero-floor lane's
nonzero `floorMax` is below the chart's fallback floor, which holds when the
chart's `fallback_replicas` is at most the planner's `fallback_floor`, and,
when the planner's `queue_pool` is not `default`, the lanes on it share one
`floorMin` and, with a nonzero `floorMin`, one bundle.

## Tests

```bash
mise run test -- packages/sie_placement/tests
```
