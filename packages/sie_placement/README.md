# sie-placement

Versioned documents for a measured-only placement planner: the service level
each served identity must meet, the measurements a placement may be chosen
from, the prices those placements cost, the demand they serve, and the plan
that results. The package uses only the Python standard library. It is a
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

## Tests

```bash
mise run test -- packages/sie_placement/tests
```
