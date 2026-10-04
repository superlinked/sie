# Threshold demand coordinator

This is the shared-state prerequisite for the `threshold` policy in issue
[#415](https://github.com/superlinked/sie/issues/415). Request routing still
refuses `threshold`: the deployment flag, coordinator lifecycle, shared caller
validation and request-path integration are separate delivery steps.

`ThresholdCoordinator` accepts a JetStream context for an **isolated control
broker endpoint**. Do not pass the inference queue connection. The coordinator
has no upstream credentials and performs no inference calls. Inference workers
and config publishers must have no credentials for the control endpoint.
Subject permissions inside the inference broker's shared account do not prove
who produced a stored record. Keeping a separate endpoint also preserves the
existing inference broker's account, stream namespace and persisted work.

The credential-free integration fixture is
[`sie-threshold-nats.conf`](../../../tools/ci/fixtures/sie-threshold-nats.conf).
It accepts only gateway authentication, grants only the three exact threshold
stream APIs and key subjects, and denies consumer APIs. Integration tests start
this fixture independently from the unchanged inference-broker fixture.

## Shared demand and decisions

Every gateway counts validated ingress locally with per-model atomic counters.
`record_request` and `decision` are synchronous and perform no broker I/O. The
caller must count once after shared validation and before profile recursion.
Each gateway calls `sample` once per second; standby gateways flush their demand
before checking sampler ownership. Drained deltas are added to the shared
per-model counter with bounded compare-and-set retries. Requests racing a drain
stay in the following batch. An uncertain acknowledgement never restores a delta,
which could count a committed batch twice; the next successful flush breaks
shared sampling continuity. Long publication gaps discard delayed demand and
also reset evidence. A sampler elected by a five-second broker lease reads
aggregate requests per second on that cadence. `wake_above` and `sleep_below`
therefore use requests per second across all gateways, rather than per replica.
Windows and cooldowns have at least one-second resolution and are capped at a
day. Only consecutive samples beyond the relevant bound count; gaps longer than
2.5 seconds discard the accumulated evidence. Equality does not cross a bound.

The sampler starts undetermined. Demand above `wake_above` for `window_s`
selects `WakeLocal`. Demand below `sleep_below` for `cooldown_s` selects `Remote`.
Values between the bounds retain an established decision. An undetermined,
unavailable or incompatible decision must not authorize suppressing local
warm-up; the request integration must preserve the ordinary local/fallback path.

## Configuration and lifetime

All targets bind the same nonzero authoritative configuration generation, the
model name, exact execution fingerprint and all routing-policy fields. The
lease also binds the full target set, independent of listing order. A newer
generation can replace a lease by compare-and-set; an older generation or a
conflicting target set at the same generation is refused. On replacement,
old counters reset to zero without requiring traffic or adding demand.

Decisions bind that generation, execution contract, sampler owner and a UUID
term minted for each lease acquisition. Renewals preserve the term so an earlier
valid decision remains usable while the next sampling cycle publishes decisions.
Expiry, a configuration takeover or tombstone recreation mints a new term and
requires fresh timer evidence. DEL and PURGE retain their stream revision for
subsequent compare-and-set recovery. Counter recreation also mints an incarnation
UUID, so even a replacement total larger than the old total resets evidence.
Failed sampling clears local ownership,
term, cached decisions and timer evidence.

Background refresh verifies the complete lease authority before and after
reading decisions. A verified decision is cached for at most one sampling
interval, measured from the verification's start on the local monotonic clock.
Its deadline is also capped by a proven lower bound on the lease's expiry:
owners know when their renewal began; standby readers establish that bound by
observing a revision advance within one unchanged term. A newly attached standby
uses the ordinary local path until it observes a renewal. This avoids shared
wall clocks and prevents a nearly expired lease from gaining another cache
interval. Malformed, missing or changed authority clears the cache; unavailable
or expired cached decisions must allow ordinary local warm-up.
Removed sampler model state is pruned; broker counters expire after one day
without requests, so retired model keys are not retained forever. Expiry
rebuilds a conservative baseline and cannot count as fresh demand. There are at most 256 current targets, 2 KiB per
value, 1 MiB per bucket, 16 compare-and-set attempts and a two-second deadline
per background broker operation. Counter overflow, malformed records, incompatible
stream configurations and unavailable storage all return typed errors.

Coordination state uses memory streams. A broker restart loses decisions and
restarts the evidence window; it does not move or consume inference work.
One or three JetStream replicas are accepted, and the external endpoint must
supply the corresponding topology. These bounds are implementation limits,
not measured throughput or latency claims.
