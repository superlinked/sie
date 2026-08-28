# SIE Telemetry Contract

[`contract.yaml`](./contract.yaml) is the checked-in source of truth for SIE
metrics, managed log records, and the remote trace privacy boundary. It defines
ownership, names, units, attributes, histogram boundaries, Prometheus wire
names, export eligibility, and the KEDA control contract.

The contract covers every production metric, the explicitly safe managed log
schema, and every KEDA control signal. This includes generation, the Python and
Rust engine runtime diagnostics, realtime-sidecar operations, and the
conditionally owned queue/adaptive-scheduler surfaces. Only authoritative lane
lifecycle signals remain excluded until a deployed producer owns them;
undeclared new instruments are not permitted.

The current inventory is 122 application families: 120 may reach the remote
OTLP branch and two remain Prometheus-only controls. Nine additional,
exact collector-self families form a separate operational allowlist. The
Better Stack dashboard code therefore covers 129 remotely eligible families
without turning collector self-telemetry into a version-dependent wildcard.

`sie.gateway.pool.pinned_model.loaded` is intentionally Prometheus-only. Its
`pool` attribute is an API-defined logical pool name (`logical_pool`), which is
not a bounded remote dimension. This is distinct from the KEDA families' `pool`
attribute (`physical_queue_pool`), which comes from the release/deployment lane
catalog and remains bounded and remotely eligible. The producer exposes the
pinned-model state as an observable current snapshot rather than an ever-seen
series cache: a removed pool/model lane emits zero for one collection and is
then released from producer memory.

The global KEDA capacity-snapshot timestamp remains Prometheus-only. The
exact-lane JetStream queue timestamp is remotely eligible alongside
`sie.gateway.lane.queue.depth` solely as its bounded freshness companion; it is
not a managed business KPI. The Prometheus exporter naturally normalizes the
canonical depth name to `sie_gateway_lane_queue_depth` without an application
alias or collector rename.

Topology determines whether a durable-queue observation exists. The OSS
Kubernetes gateway has an exact JetStream consumer authority and emits queue
depth plus lane freshness. Managed Modal dispatch has no JetStream queue, so it
omits those two families without warning; it still emits the same canonical
pending-demand, active-lease, warm-floor, and rejection state over OTLP. An
absent managed queue series therefore means “not applicable,” never a measured
zero and never a broken duplicate metric path.

## One event, one producer call

Application code emits a typed semantic event through its runtime-local
telemetry facade:

```text
request path -> telemetry.request_completed(observation)
                         |
                         v
                canonical OTel instruments/log
                         |
                  OTLP-only export
                         |
               regional/local OTel collector
                    /                   \
       Prometheus exporter :9464      OTLP exporter
          Prometheus -> KEDA           Better Stack
          Prometheus -> Grafana
```

The facade may update the counter, duration histogram, admission counter, and
safe completion log that belong to that one semantic event. The request path
still calls the facade only once. It must never call a Prometheus client and an
OTel client for the same event.

Every deployment uses the same application path: OTel instruments and log
records leave the process through OTLP. Prometheus is a collector exporter, not
an application instrumentation API. The bundled OSS/Kubernetes collector
exposes `:9464/metrics`; a managed collector exports to Better Stack; and a
Kubernetes deployment with remote telemetry can do both. Fan-out happens only
after the collector receives the one application observation.

The application OTLP wire contract is explicit rather than backend-specific:
monotonic counters and histograms use DELTA temporality, up/down counters stay
cumulative, and gauges remain current values. This lets short-lived managed
workers export their first real observation without first establishing a
cumulative baseline. Better Stack receives that stream unchanged. The pinned
collector Prometheus exporter owns the only compatibility state and accumulates
DELTA sums/histograms into cumulative Prometheus families; KEDA's gauge values
and exact labels pass through as current state. Python exporters pin the exact
per-instrument map and Rust exporters use the SDK's `LowMemory` preference,
which has those same semantics for the production instrument inventory.
Ambient OTel temporality environment variables cannot change this policy. An
external collector must therefore accept OTLP DELTA input.

Collector self-metrics are the deliberate exception: they originate as a
cumulative Prometheus scrape and are routed to Better Stack on their isolated
self pipeline. DELTA intervals are additive and cannot be reconstructed after
an exporter exhausts its bounded retries, so such an outage may temporarily
undercount counters. KEDA's primary capacity signals are refreshed gauges; its
rejection-rate counter remains freshness-gated and falls back through the same
bounded control-path failure policy.

OTLP transport is an exporter setting, not a semantic-instrument choice. Every
signal uses its signal-specific endpoint before the generic endpoint and never
inherits from a sibling signal. Where transport is selectable, its
signal-specific protocol likewise precedes the generic protocol. Dual-transport
runtimes retain gRPC as the OSS/Kubernetes default and use HTTP protobuf for the
authenticated managed Modal collector; the managed-only dispatcher requires
that HTTP-protobuf protocol explicitly. A signal-specific endpoint is always
used exactly as written; when only the generic endpoint is present, an HTTP
exporter appends the standard `/v1/traces`, `/v1/metrics`, or `/v1/logs` path,
while gRPC keeps the shared endpoint unchanged. No call site changes when the
transport or collector destination changes.

The collector also binds each retained metric to the exact `service.name`
declared for its owner. This is a schema and data-quality boundary among trusted
producers, not producer authentication: a process holding the shared managed
collector credential could forge `service.name`. Unknown services and
accidental cross-owner observations are still dropped before either
destination. The conditionally owned queue and scheduler families admit only
`sie-worker` on a Modal-native topology or `sie-worker-sidecar` on a realtime
topology. In Helm, split receivers plus NetworkPolicy provide the stronger
identity boundary protecting the KEDA control branch.

When telemetry is disabled, the semantic facade resolves to a tiny local
no-op: it constructs no SDK instruments or point-attribute maps. Request-level
instrumentation middleware is omitted entirely rather than running clocks and
route normalization into an SDK no-op. Business call sites can remain
unconditional without paying exporter or label-construction cost.

The checked-in `performance-budgets.json` makes that boundary executable. Each
representative hot path records three independently warmed samples and gates
the median. Telemetry-off paths have explicit no-op ceilings, telemetry-on
paths retain absolute ceilings, and paired paths gate their incremental median;
the config facade also gates a per-invocation temporary-allocation peak sampled
without amortizing one high-water mark across a large loop. These are generous
regression tripwires on CI hardware, not a claim of literally zero cost or a
substitute for end-to-end production latency measurements.

Every resource includes a unique-per-process-start `service.instance.id`.
Kubernetes provides `SIE_TELEMETRY_INSTANCE_ID=$(POD_UID)/<container-name>` as
a stable substrate prefix; each runtime appends a process-start UUID before
building its OTel resource. Other substrates provide an equivalent prefix
through the same environment seam. Without one, the UUID is the entire value.
For collector self-metrics, the Kubernetes collector reads
`/proc/sys/kernel/random/uuid` once through Collector 0.119's file config
provider and strips the procfs trailing newline. The managed Modal launcher
instead generates `SIE_OTEL_COLLECTOR_INSTANCE_ID` immediately before starting
each collector process because Modal does not expose that procfs path. Both
paths change identity on every collector process restart.
The collector's trusted receiver branch derives `producer_service` from its
receiver boundary and `producer_instance` from the sanitized
`service.instance.id` resource field before Prometheus export. The
Prometheus-only branch also reads one process UUID at config load and adds it
as `collector_generation` to every exported point. That label is constant for
one collector process and changes on every restart, giving the in-memory DELTA
accumulator an explicit series boundary. It is never sent to Better Stack. The
ServiceMonitor makes scrape-target identity authoritative with
`honorLabels: false`; it never copies exporter-controlled `job`, `instance`, or
`exported_*` values into the producer labels. KEDA removes the producer replica
and collector-generation labels only after its freshness join. Environment and
region remain resource attributes rather than copied point labels.

Operated environments never share a telemetry sink: managed `dev`, `staging`,
and `prod` each have a separate collector, Better Stack source, dashboard state,
and secret set per region. Producers also stamp those two routing dimensions as
`deployment.environment` and `cloud.region` resource attributes. A local process
with no deployment metadata uses the explicit value `unknown`; an operated
deployment must inject real values and must not rely on that fallback.

The Better Stack trace branch is an allowlist boundary, not a verbatim trace
forwarder. It preserves trace/span/parent IDs, start/end timestamps, kind,
status code, flags, a bounded span name, and the five safe resource identity
fields declared in `contract.yaml`. It removes every span event and span/scope
attribute, clears status text, inbound trace-state text, and scope identity,
drops unknown resource attributes, and collapses unknown span names to `other`.
Because each link can carry arbitrary attributes and trace-state text, and the
pinned collector cannot reliably mutate links in place, the remote branch
drops the entire linked span. Unlinked sibling spans continue through the
allowlist; the unchanged local OSS branch may retain linked spans. When Helm is
configured with Tempo as well as Better Stack, those are separate collector
pipelines: the local OSS branch receives the producer trace unchanged, while
only the remote branch runs the privacy processors. Producers still export one
OTLP stream. The remote Helm branch also treats receiver identity as part of
that boundary: the gateway receiver overwrites `service.name` with
`sie-gateway`, the application receiver admits only the four declared
non-gateway service names, and both overwrite environment and region from the
collector deployment. Producer process identity and optional version remain
the only retained producer-authored resource identity. These processors never
run on the local Tempo pipeline.

The declared gateway completion log has the same receiver-authority rule. Only
the gateway receiver is connected to the log pipeline; the collector overwrites
service, environment, and region before either its structured local sink or
remote OTLP sink. A config or worker process cannot make an allowlisted-looking
gateway record reach that pipeline by forging resource attributes.

Each runtime owns a thin facade because Python and Rust have different SDKs and
process lifecycles. The semantic methods and their emitted instruments are
nevertheless governed by the one contract:

- `sie_gateway` owns HTTP completion, admission, KEDA capacity state, and the
  request span/log boundary. Downstream deployments can reuse that facade for
  their final dispatch result.
- The Modal dispatcher owns actual substrate invocation attempts.
- `sie_config` owns config HTTP and authoritative state changes.
- `sie_server_sidecar` owns realtime queueing and batch formation plus its
  operational config, NATS, payload, capacity, IPC, generation-model-loading,
  and shutdown-drain signals.
- `sie_server` and `sie_server_rust` implement the shared `sie-worker` engine
  contract for item completion, phases, units, model lifecycle, eviction, and
  OOM. Request and phase histograms are item-weighted: a semantic completion
  carrying `item_count=N` records N duration observations so histogram counts
  stay aligned with `sie.worker.requests`, regardless of whether the engine
  reports items individually or as a batch. The Rust engine additionally owns
  bounded forward duration, permit wait, concurrency, and limit diagnostics
  under `sie.worker.runtime.*`.
- The Python `sie-worker` engine owns generation TTFT/TPOT/tokens,
  admission/KV state, duplicate prevention, and grammar diagnostics. Generation
  is not a separate `sie-generation` telemetry service.
- TPOT has one cross-runtime definition: the non-negative wall time from the
  first to the last output-bearing event divided by the authoritative positive
  completion-token count, or by the observed output-event count when usage is
  absent. The denominator includes the first output, matching the existing
  `total latency = TTFT + TPOT * output tokens` convention. A stream with fewer
  than two timed output events has no measurable TPOT and emits no observation.
- Queue, batch, and adaptive-scheduler ownership is topology-selective: the
  realtime sidecar owns it when present; the Modal-native Python lane owns it
  otherwise. Python QueueExecutor batch fragmentation uses the transport-neutral
  `sie.worker.runtime.batch.size`, `sie.worker.runtime.batch.subgroups`, and
  `sie.worker.runtime.subgroup.size` metrics. The engine must not observe sidecar
  queue or batch events again.

The Python worker facade, Rust worker facade, and realtime sidecar each admit
at most 256 exact `(model, profile)` metric pairs for the lifetime of one
process. Materialized `model:profile` registry aliases are canonicalized back
to that semantic pair, and catalog duplicates are de-duplicated as pairs. An
unknown model or profile collapses atomically to `(other, other)`; later pairs
beyond the lifetime budget share that same single pair. Admission never
affects routing, config apply, or inference, and each producer emits its
representability warning at most once per process. Both engine facades perform
this collapse before their OTel SDK. The Python synchronous aggregators
otherwise have no default ceiling; the Rust SDK defaults to 2,000 series.
Removed pairs retain their lifetime admission, so sequential catalog
replacement cannot reopen the budget. To keep the Rust SDK's eagerly sized
tracker maps small, its high-product streams take the first-observed,
process-lifetime subsets of that catalog admission: 32 exact pairs for
request/phase/unit detail and four exact pairs for the 47-stage
forward-duration detail. Excess pairs collapse only in those telemetry
streams; model lifecycle, permit, concurrency, and limit streams still retain
the 256-pair catalog domain.

The shared worker completion outcome is one closed five-value domain:
`success`, `error`, `retry`, `cancelled`, and `other`. With seven operations,
the Python request stream retains at most `257 × 7 × 5 = 8,995` series. The
Rust request-detail subset bounds requests at `33 × 7 × 5 = 1,155`, phases at
3,465, and units at 693. Cancellation is therefore observable without
admitting arbitrary finish reasons or silently folding a contract-valid
terminal into `other`.

Every Rust-engine instrument has an explicit SDK view derived from its
checked-in admission tier and finite domains, so contract-valid series do not
become `otel.metric.overflow`. Forward output paths use one closed six-value
domain: dense, sparse, three multivector wire paths, and other. The largest
single-instrument ceiling is now 8,460 series for forward duration, and all 13
view limits sum to 29,577. For the pinned OTel SDK, each tracker map requests
capacity for one plus its view limit, so the checked-in total initial capacity
request is 29,590 entries—not the previous 362,370-series forward-map ceiling.
At full observation, explicit histograms retain 379,022 `u64` bucket-counter
cells (3,032,176 raw bytes, or 6,064,352 with one export snapshot). Hash-table
control bytes, tracker and attribute allocations, and allocator rounding are
additional but remain bounded by those per-instrument series ceilings; the
contract does not claim a portable exact byte total for allocator-dependent
metadata.

For the sidecar's six declared queue operations, the budget is
`6 × (256 + 1) = 1,542` queue series; including the defensive
`operation=other` stream is `7 × 257 = 1,799`, below the Rust SDK's default
2,000-series ceiling.

Every sidecar instrument nevertheless has an explicit SDK view derived from
its full checked-in attribute domains. Batch size/cost omit `flush.reason`
because it does not change their batch-shape semantics; fill ratio retains it.
`sie.worker.work_item.age` omits the catalog pair entirely and costs seven
series: transport-queue age is a property of the queue rather than of the
model on the far side of it, so the catalog factor would multiply the series
count without adding an answer. The resulting high-product ceilings are 14,392
fill-ratio series and 4,112 generation-loading series, while all other sidecar
ceilings are at or below 1,799. These are upper bounds on retained SDK series,
not expected steady-state usage or byte-size claims; the machine-checked
formulas live in `contract.yaml`.

## KEDA is a control API

The `keda` section is load-bearing. Its Prometheus family names and labels may
not drift independently of the Helm `ScaledObject` queries. The bundled
collector exposes `:9464/metrics`, a `ServiceMonitor` makes Prometheus scrape
that target, and KEDA queries Prometheus through `/api/v1/query`. KEDA never
queries the exposition endpoint or Better Stack and never depends on a vendor
credential. This makes the local collector and Prometheus part of the
autoscaling control path. In the Helm chart `autoscaling.enabled=true` implies
canonical metrics, the bundled collector Prometheus exporter, and its
ServiceMonitor; `keda.install` installs only the operator. The application
metrics export interval and Prometheus scrape interval are each capped at five
seconds, and the scrape timeout cannot exceed its interval. Collector
readiness, Prometheus target health, and query-API readiness gate the path.
Every worker-lane and gateway `ScaledObject` keeps a bounded
failure-threshold fallback. Autoscaling validates the gateway's declared
one-to-ten replica range and uses that floor for fallback, so loss of the
telemetry control path retains safe capacity without unbounded fail-open
scaling. Canonical metrics enforce the same one-to-ten reader bound even when
KEDA is disabled. Any lane with `minReplicas=0` also requires queue activation
threshold `0`: KEDA activates only when the value is strictly greater than the
threshold, so this is what makes the first durable queued item sufficient to
leave zero. Deployments whose every lane has a static warm replica may choose a
higher activation threshold.

When its Prometheus exporter is enabled, the bundled collector is a single
active accumulation point. In that mode its Deployment uses a recreate rollout
so an upgrade cannot split one producer's DELTA intervals across overlapping
exporter caches. Better-Stack-only, trace-only, and log-only collectors retain
the rolling Deployment default. The resulting short Prometheus-mode collector
gap is fail-closed by the freshness joins and remains inside KEDA's bounded
fallback behavior. On any process restart, `collector_generation` changes, so
Prometheus starts distinct counter/histogram series even when the first new
accumulated value is greater than the prior process's final value; refreshed
gauges repopulate under the same new generation on the next export. This adds
no active-series multiplier during normal operation, but each restart leaves
one intentional historical generation per active series until Prometheus
retention expires.

Pending demand remains set until dispatch durability transfers authority to
the broker backlog. Submission itself is nonblocking: one handler-owned task
awaits the transport-neutral durability future after beginning one
request-scoped handoff lease. Complete success releases only that exact lease
and retires a same-lane cold/backpressure marker that predates the handoff;
demand recorded after the handoff began survives it. Failure retains
request-scoped demand, notifies the request driver, and cleans up the handoff
when the request exits. The aggregate completion timeout is six seconds. Queue
APIs admit at most 4096 items per request, and the publisher
overlaps at most 64 initial sends at once; ACK ownership is therefore bounded
without changing the per-item durability contract. Submission metrics use
`outcome="submitted"`; durable results use the separate
`queue.events{event="publish_ack",outcome="success|ack_error"}` family.

The chart carries canonical ScaledObjects in ordinary Helm-owned ConfigMap
shards of at most 32 worker manifests plus one gateway shard. A digest-pinned
post-install/post-upgrade hook server-dry-runs and applies those manifests, then
prunes only same-release managed names absent from the current revision. It
refuses to adopt a same-name object without the canonical Helm release
identity. The telemetry runtime retains its 1,024-lane bound. Helm-managed
autoscaling admits 192 worker lanes with external KEDA and Prometheus, or 96
when either dependency is bundled, so the complete stored release remains
below its conservative payload budget without a second migration
representation. The manifests are not secrets; write access to ConfigMaps in
the effective workload namespace is therefore a trusted control-plane
capability, just like the pod/label authority described below.

ScaledObject names, HPA names, scale targets, and immutable workload selectors
remain unchanged across the `0.6.20` boundary. When autoscaling is enabled,
Helm uses `lookup` to render each existing Deployment or StatefulSet's live
`spec.replicas`; a fresh workload uses its configured floor. This prevents the
Helm resource patch from resetting the observed HPA-controlled replica count and leaves no
permanent replica-pin data in values, annotations, or release history.

The `0.6.20` boundary requires one supervised maintenance-window upgrade.
Stop external traffic and topology/config writes, keep the effective namespace,
lane catalog, names, scale targets, Prometheus backend, and KEDA ownership
unchanged, and run one normal
`helm upgrade --wait` with hooks. Helm renders each observed live workload
replica count through `lookup`; the Prometheus gate first proves the new
collector path, the apply hook updates the same ScaledObject names and scale
targets in place, and the KEDA/HPA gate proves their collector-backed queries
before Helm reports success. Resume traffic only after that success.

There is no extra controller, second telemetry service, or target-owned durable
migration state. Canonical `0.6.20` may leave one inert source hook ConfigMap;
the runbook removes it once after target health succeeds. Rollback,
`helm upgrade --atomic`, and `--no-hooks` are
unsupported at this boundary. If the upgrade fails, keep the maintenance
window, diagnose the live state, fix forward, and retry the same target. Make
topology changes only after the boundary upgrade has completed, except for the
runbook's required over-limit lane reduction. Fresh installs use an ordinary
Helm install.

The readiness hook selects exact-revision ScaledObjects plus all release-owned
HPAs and requires the expected ScaledObject count, one current HPA for every observed ScaledObject, reconciled external metric names and
health keys, every trigger `Happy` with zero failures, and no fallback condition
after a complete three-poll failure window. It reads both APIs in bounded
32-object pages. An autoscaling-disable upgrade and uninstall delete only the
exact release-managed ScaledObjects before the KEDA API can disappear.

The Helm collector has separate trust-scoped OTLP receivers. Release gateway
pods emit once to the KEDA-trusted receiver on `4317`/`4318`; config, worker,
and worker-sidecar processes emit once to the application receiver on `4327`.
An always-rendered ingress `NetworkPolicy` selects the exact Helm release and
allows only those producer classes on their respective ports. This prevents a
non-gateway SIE producer from claiming `service.name=sie-gateway` and forging a
capacity signal, completion log, or remote gateway trace. Remote application
traces are limited to `sie-config`, `sie-dispatcher`, `sie-worker`, and
`sie-worker-sidecar`; their environment and region are collector-authored.
The effective workload namespace (`global.namespace` when set, otherwise the
Helm release namespace) is a workload trust boundary: cluster
RBAC must not let untrusted principals create pods or spoof release/component
labels there. Prometheus exposition and kubelet health ports remain reachable
without weakening either OTLP receiver. Port `9464` admits pods in the
effective workload namespace plus pods in the explicitly configured scrape
namespace names. That list defaults to `[]`; an external Prometheus namespace
must be named explicitly. The policy has no `13133` rule: node-local kubelet
probe traffic is unaffected by pod ingress NetworkPolicy.

Every application-metric selector is scoped to the exact release collector
target and the collector-authored producer service:

```promql
{namespace="<effective workload namespace>", service="<collector service>", endpoint="prometheus", producer_service="sie-gateway"}
```

`namespace`, `service`, and `endpoint` are authoritative ServiceMonitor target
labels. The monitor sets `honorLabels: false`, so exporter-provided `job` and
`instance` values cannot replace scrape identity. `producer_service` and
`producer_instance` already come from the collector's trusted gateway branch;
the monitor has no relabeling rule that can derive them from exporter input.
The exact target match plus the trusted gateway-only collector receiver
prevents another release, scrape target, or application receiver from
contributing to an autoscaling decision.

Collector health alone cannot prove that gateway OTLP export is still moving.
Before reading independently locked registry, pool, and demand state, the
gateway captures reconciliation start time. It records that value in the Unix
timestamp gauge `sie.gateway.capacity.snapshot.timestamp` only after recording
the state values; the collector exposes it as
`sie_gateway_capacity_snapshot_timestamp_seconds`. KEDA requires both the exact
collector target to be up and at least one release-scoped gateway snapshot to
be less than 20 seconds old:

```promql
(max(up{namespace="<effective workload namespace>", service="<collector service>", endpoint="prometheus"}) == 1)
and on()
(max(abs(time() - sie_gateway_capacity_snapshot_timestamp_seconds{namespace="<effective workload namespace>", service="<collector service>", endpoint="prometheus", producer_service="sie-gateway"}) < 20))
```

Before HA aggregation, every business series is first joined on both
`producer_instance` and `collector_generation` to the capacity timestamp from
that same gateway process and collector accumulator generation:

```promql
sie_gateway_pending_demand{...}
and on(producer_instance,collector_generation)
(abs(time() - sie_gateway_capacity_snapshot_timestamp_seconds{...}) < 20)
```

The queue trigger uses the stricter same-label broker timestamp instead:

```promql
sie_gateway_lane_queue_depth{..., pool="<pool>", machine_profile="<profile>", bundle="<bundle>"}
and on(producer_instance,collector_generation)
(abs(time() - sie_gateway_lane_queue_snapshot_timestamp_seconds{..., pool="<pool>", machine_profile="<profile>", bundle="<bundle>"}) < 20)
```

Its readiness guard also requires `count(...) > 0`. That distinction makes a
fresh explicit queue value of zero valid while a missing queue point remains a
KEDA error.

This prevents retained points from a terminated/restarted gateway instance
from becoming valid merely because another instance is fresh, and prevents an
old counter range from joining freshness emitted after the collector's
in-memory accumulator restarted. Current-state gauges never synthesize a zero:
their producers emit explicit zeros, so a missing or stale series yields no
result. Request and scale-worthy rejection counters legitimately have no series
before their first event; only those two queries derive zero from the same
fresh capacity-snapshot series. An unconditional `or vector(0)` is forbidden.
A down collector, a broken gateway-to-collector OTLP path, or a stale/missing
reconciliation therefore still yields no result.
All Prometheus triggers set `ignoreNullValues: "false"`, so KEDA advances its
failure threshold and uses the ScaledObject's bounded fallback instead of
interpreting the failure as zero demand.

The four current-state families use synchronous last-value gauges. Broker,
pool, and demand reads are independently reconciled, and the OTel reader may
collect instruments while a reconciliation is recording, so this contract is
explicitly eventually consistent rather than cross-family export-atomic. Using
reconciliation *start* time makes a slow build age out instead of falsely
refreshing old state. Broker success is lane-scoped: one consumer failure
updates neither that lane's queue value nor its freshness timestamp, while
successful lanes and the independent global demand/lease/floor state keep
refreshing. The retained queue value becomes unusable when its timestamp ages
out; it is never rewritten to a false zero. The five-second loop skips missed
ticks rather than queueing catch-up work. Each process hashes its process-start
UUID to a stable offset within that interval, so rollout-coincident replicas
keep staggered first and subsequent broker scans. Enabled canonical telemetry
intentionally runs this reader even when KEDA itself is external because queue
depth is remotely eligible. The frozen deployment catalog hard-caps the scan
at 1,024 lanes and the chart caps it at ten gateway replicas; the opt-in
live-broker benchmark exercises all ten readers concurrently at that limit
(10,240 lookups per interval).

The state families are synchronous last-value gauges over current gateway
state. The lane catalog is frozen for the gateway process lifetime; successful
reads always publish explicit zero, while failed reads retain an aging value
that cannot satisfy the freshness join. Multi-gateway queries deliberately
distinguish replicated state from distributed event rates:

- state replicated by every gateway uses `max` after the per-instance join;
- each gateway reads `num_pending + num_ack_pending` from the exact durable
  JetStream consumer into `sie.gateway.lane.queue.depth`; Prometheus then uses
  `max` across the replicated gateway snapshots;
- request and scale-worthy rejection counters use `sum(rate(...))`;
- rejection scaling is a positive `scaling_action="scale_up"` classification,
  not a negative regex that lets a new validation/error reason scale GPUs.

The queue gauge has only the exact physical-lane tuple (`pool`,
`machine_profile`, `bundle`) and is eligible for remote export under its
canonical OTel name. Its lane-scoped freshness timestamp uses the same labels,
is recorded only for each successful lane read, and is exported remotely only
to qualify depth from the same producer instance. A failed lane updates neither
its queue value nor freshness: KEDA receives an empty result for that exact
lane and activates its bounded fallback, while the Better Stack query drops the
aging depth rather than presenting it as current or zero. Other lanes and the
global demand/lease/floor snapshot remain fresh. `pool` is the physical queue pool, while
`machine_profile` and `bundle` are rendered deployment values.

This queue paragraph applies only to the realtime JetStream composition. The
managed Modal composition runs the shared reconciliation facade in an explicit
no-queue mode: no broker scan or missing-source warning occurs, and the
independent demand/lease/floor observations continue to reach Better Stack.

## Compatibility and conversion

The dotted OTel name is canonical. `prometheus_name` records the exact expected
Prometheus base family produced by the collector exporter; histogram children
add `_bucket`, `_sum`, and `_count`. Existing names may be retained through
collector transforms or backend recording rules, but compatibility must never
add a second producer call.

Before either destination, the collector admits only exact dotted names owned
by that receiver and rewrites every point to the exact attribute set declared
for that metric. It also clears arbitrary scope identity and metric
descriptions, canonicalizes units, and keeps only the five declared resource
fields. Every producer disables exemplars: the pinned collector cannot reliably
remove exemplar filtered attributes, so CI captures its remote protobuf and
requires zero exemplars. Prometheus `producer_service`/`producer_instance`
labels and the Prometheus-only `collector_generation` are authored by the
collector from the trusted receiver/resource boundary and collector process
identity. The remote queue depth and its freshness companion receive only the
collector-authored `producer_instance` needed for their per-replica join; the
process-generation label never enters the remote pipeline. These labels are
never copied from exporter-controlled `job`, `instance`, or `exported_*`
labels.

[`legacy-prometheus-migrations.yaml`](./legacy-prometheus-migrations.yaml) is
the complete migration ledger for the 245 Prometheus families registered at
the audited pre-refactor revision. Every family appears exactly once as
`direct`, `collapsed`, `derived`, `narrowed`, or `retired`; the five pre-refactor KEDA
families retain their exact legacy-to-canonical label mapping. The ledger is a
compatibility and query-rewrite input only—none of its legacy names is a
producer alias.

Better Stack, Prometheus remote-write, or another backend is selected in the
collector/exporter configuration. No producer contains a Better Stack token or
vendor-specific metric call. In managed Modal deployments, a producer receives
only its regional collector endpoint and the paired Modal proxy-auth client
credential. The collector alone receives the per-source Better Stack ingest
host/token; the provisioning-only Better Stack management API key is not a
runtime producer or collector credential. Dev, staging, and production keep
those source tokens and collector secrets isolated per region.

For Helm, the runtime credential is specifically the Telemetry source ingestion
token—not a settings-management API token or an Uptime token. Store it under an
operator-chosen key in an ordinary Secret in the effective workload namespace,
then set `observability.otel.collector.betterStack.existingSecret` and
`tokenKey`. Set `endpoint` to the bare HTTPS OTLP origin shown by that source,
and set explicit `observability.otel.resource.deploymentEnvironment`
(`dev|staging|prod`) and non-unknown `cloudRegion`. The chart README contains a
stdin-based Secret creation example that avoids placing the token in Helm
values or application Pods.

## Contract checks

CI should reject a change unless it proves all of the following:

1. every facade's instrument inventory exactly matches `contract.yaml`;
2. Python and Rust agree on shared names, attributes, units, and buckets;
3. a golden collector `:9464/metrics` scrape contains every KEDA family with
   the exact aliases and labels;
4. rendered KEDA PromQL references only contract-declared families and labels;
5. an OTLP test receiver gets one observation from one facade event and the
   collector exposes that same observation once through Prometheus;
6. collector filters and dashboards reference contract-declared signals;
7. managed logs match the allowlisted schema and contain no forbidden fields;
8. an OTLP protobuf capture proves the remote trace branch removes undeclared
   fields, events, linked spans, trace-state and status text while the local
   Tempo branch remains unchanged;
9. median-of-three warmed telemetry-off/on benchmarks cover the gateway
   facade/Tower path, the managed cloud-gateway final-dispatch wrapper, Python
   and Rust workers, config, sidecar, and dispatcher hot paths; a paired
   durability-disabled/enabled benchmark separately gates the gateway
   dispatch-durability lifecycle before the change is declared ready;
10. an end-to-end KEDA signal respects the declared five-second OTLP export and
   Prometheus scrape budgets, and collector, producer-export, scrape, or query
   failure activates the worker lane's declared safe fallback.
