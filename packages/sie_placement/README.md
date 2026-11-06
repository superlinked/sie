# sie-placement

Versioned documents for a measured-only placement planner: the service level
each served identity must meet, the measurements a placement may be chosen
from, the prices those placements cost, the demand they serve, and the plan
that results. The package uses only the Python standard library, except the
KEDA applier, whose `keda` extra adds the SIE Python SDK. It is a workspace
package and is not published to PyPI.

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

## Lanes

A plan lane carries the three tokens that name a lane of the `sie-cluster`
Helm chart: `queuePool`, the `machineProfile` of its `cell`, and `bundle`.
`queuePool` is the physical queue pool the lane's workers consume:
`workers.common.queuePool` (`default`), or a worker group's `queuePool`
override. It is neither the `workers.pools` map key, which names a worker
group, nor an SDK logical pool. `bundle` is a key of that worker group's
`bundles` map. `name` is the lane's handle inside the plan, and each identity
refers to its lane by that name.

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
- the `sie-cluster` chart cannot run the plan: a lane on the `default` queue
  pool pins an identity, whose pins no static queue pool carries, or an
  identity is `bridged`, `snapshot-cold` or `scheduled-warm`, warmths the chart
  does not realize;
- a token file or `SIE_ADMIN_TOKEN` holds anything but one run of printable
  ASCII characters without spaces after trimming;
- `--prometheus-url` or `--gateway-url` is not an absolute http or https URL
  with a host, embeds credentials, carries a query or a fragment, or has an
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
logs it. Without it, `--apply` sends its writes with no credential, which a
gateway with token auth refuses. The client token is never sent on writes. A
refused token is named by its source, never by its value, and a failed gateway
call is reported by the server's error message or, for a transport failure, by
the exception type alone, so no reported error carries a header value.

## Tests

```bash
mise run test -- packages/sie_placement/tests
```
