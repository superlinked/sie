# SIE Cluster Helm Chart

Deploy SIE (Search Inference Engine) to Kubernetes with autoscaling and observability.

## Quick Start

```bash
helm install sie-cluster oci://ghcr.io/superlinked/charts/sie-cluster \
  --namespace sie \
  --create-namespace
```

## Local validation

Prepare dependencies from the checked-in `Chart.yaml` and `Chart.lock`, then
render with an explicit non-secret payload-store choice:

```bash
mise run helm -- dependencies
mise run helm -- lint --set payloadStore.enabled=false
mise run helm -- template --set payloadStore.enabled=false
```

The task temporarily stages the public model and bundle YAML files into the
chart and removes them after each render. Helm's generated `charts/` directory
is ignored and must not be committed.

## Architecture

```
┌─────────────┐     ┌─────────────────────────────────────┐     ┌──────────────┐
│   Client    │────▶│      Gateway (1 replica; 2+ for HA)  │◀───▶│  sie-config  │
└─────────────┘     └───────────────┬─────────────────────┘     │ (singleton)  │
                                    │                           └──────────────┘
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌─────────┐     ┌─────────┐     ┌─────────┐
              │ L4 Pool │     │A100 Pool│     │ CPU Pool│
              │ 0-N     │     │ 0-N     │     │ 0-N     │
              └─────────┘     └─────────┘     └─────────┘
```

- **Gateway**: Stateless request proxy that routes to workers based on GPU type and model affinity. Consumes config via GET/NATS from `sie-config`.
- **sie-config**: Authoritative control plane for model/bundle configuration. Serves `/v1/configs/*` writes and publishes NATS deltas to the gateway and workers. Deployed as a singleton (`replicas: 1`, `strategy: Recreate`).
- **Worker Pools**: StatefulSets per enabled worker group, each with KEDA autoscaling. Routing, metrics, and KEDA scale on the full `(queuePool, machineProfile, bundle)` lane.

Helm resolves that physical lane once and reuses it for worker/sidecar env,
heartbeats, physical queue-pool pod metadata, gateway configured profiles, and
every KEDA query. Immutable workload selectors retain the logical pool/bundle
identity published by v0.6.20. Tokens are trimmed, limited to 63 characters, validated against
`^[A-Za-z0-9_-]+$`, and lowercased. Defaults apply only to omitted
`queuePool`/`machineProfile` fields; explicit blank values fail rendering, as do
two enabled entries that normalize to the same physical tuple.
The same render publishes at most 1024 exact tuples to the gateway in
`SIE_GATEWAY_CONFIGURED_PHYSICAL_LANES`; pending demand and scale-related
rejections are recorded only for a catalog-resolved tuple.
Worker-pool and bundle map keys remain stable Kubernetes/KEDA object identities
and must be lowercase DNS-1123 labels; KEDA's internal `metricName` and object
names use only those validated identities, while PromQL always uses the
canonical physical tuple.

## Cold Start Expectations

The current GKE/EKS scale-from-zero guidance is:

| Step | Duration | Notes |
|-------|----------|-------|
| **Node provisioning** | 2-5 min | GKE/EKS spins up GPU node (spot may be slower) |
| **Container startup** | 20-40s | Pull image, start process, health checks |
| **Model loading** | 10-120s | Download weights (if not cached), load to GPU |
| **Total cold start** | 3-7 min | First request to a scaled-to-zero pool |

These figures are not an ACK performance claim; this repository has no ACK
cold-start baseline. The Alibaba Terraform module defaults its shared NAT EIP
to 100 Mbit/s, and observed ACK cold start depends on current GPU inventory and
quota, effective bandwidth, image and model size, and cache state.

### Reducing Cold Start Time

1. **Use cluster cache**: Pre-populate object storage with model weights (`--cluster-cache`)
2. **Set minReplicas=1**: Keep one warm replica per critical GPU type
3. **Use reserved capacity**: Avoid spot for latency-sensitive workloads
4. **Pre-warm models**: Call `/v1/encode/{model}` on startup to load weights

### Client Handling

When a pool is scaling from zero, the gateway returns:
- **503 Service Unavailable** with `X-SIE-Error-Code: PROVISIONING`
- **Retry-After: 60** header
- Client should retry after the indicated delay

The SDK handles this automatically with configurable retries.

## Cluster model cache

Pre-populate shared object storage with model weights so worker pods don't re-download from HuggingFace on every cold start. The Python SDK pulls from the cache first and falls back to HF on miss.

**AWS example:**

```bash
# 1. Provision object storage and populate its /models prefix with model weights.
MODEL_CACHE_URL="s3://<bucket>/models"

# 2. Wire the readable object-store URL into Helm.
helm upgrade --install sie-cluster . \
  --set workers.common.clusterCache.enabled=true \
  --set workers.common.clusterCache.url="$MODEL_CACHE_URL"
```

The configured URL must include the `/models` prefix and be readable by the worker workload identity. Because `payloadStore.enabled` defaults to `true`, the chart also derives a `/payloads` URL from this cache URL; grant the gateway workload write access and worker workloads read access to that prefix, configure a separate writable `payloadStore.url`, or set `payloadStore.enabled=false`.

**Other clouds / BYO bucket:** point `workers.common.clusterCache.url` at any `s3://...`, `gs://...`, `abfs://...`, `abfss://...`, or `oss://...` URL the worker workload identity can read, and populate the `/models` prefix with your object-storage tooling or existing cache pipeline. With the default payload store enabled, also grant the gateway workload write access and worker workloads read access to the derived `/payloads` prefix, configure a separate writable `payloadStore.url`, or set `payloadStore.enabled=false`.

## SGLang kernel cache

SGLang, FlashInfer, DeepGEMM, Triton, and the CUDA driver can compile kernels
after model weights are available. SIE remains fully cacheless-capable, but a
persistent compiler cache can avoid repeating that work after a container or
worker-pod replacement. It does not remove the first compilation on a genuinely
empty cache, and CUDA graphs are still captured by each live SGLang process.

For a local CUDA container, mount one named volume at the image's default cache
path:

```bash
docker run --gpus all -p 8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  -v sie-kernel-cache:/app/.cache/sie-kernels \
  ghcr.io/superlinked/sie-server:latest-cuda13-sglang-cu130
```

CUDA images use `/app/.cache/sie-kernels` automatically; the named volume only
upgrades that container-local default to survive container replacement. The
adapter namespaces artifacts by the installed JIT dependency ABI and exact GPU
product name, then sets upstream cache locations only when operators have not
set those variables explicitly. An unwritable or unidentifiable cache falls
back to ordinary compilation.

For Kubernetes, enable retained per-replica claims globally or only for a
specific generation lane:

```yaml
workers:
  common:
    kernelCache:
      enabled: false
      storageSize: 20Gi
      storageClassName: ""  # cluster default
  pools:
    h100:
      bundles:
        sglang-cu130:
          kernelCache:
            enabled: true
          preloadModels:
            - Qwen/Qwen3.8-27B-FP8:h100-256k
```

Each StatefulSet ordinal receives its own `ReadWriteOnce` PVC, retained across
scale-down and pod replacement by the StatefulSet's default PVC policy. Do not
replace this with one concurrently writable RWX DeepGEMM cache. Ensure the
chosen StorageClass can provision in every zone that may host the GPU pool;
retained claims otherwise constrain later scheduling to their volume topology.
`preloadModels` fills the cache while the worker is still NotReady.

Adding or removing `volumeClaimTemplates` is an immutable StatefulSet change.
For an existing release, enable or disable this setting during a controlled
recreation of the affected worker StatefulSet; retained cache PVCs are
disposable acceleration state and may be deleted separately when no longer
needed.

## Payload store

Work items larger than 1MB (for example images or long documents) are too big to put on the NATS queue inline, so the gateway offloads the payload to object storage and enqueues only a reference; workers fetch it back. **This is required for >1MB requests**: without a payload store the gateway cannot enqueue them and the request fails.

It is therefore **enabled by default** (`payloadStore.enabled=true`) and is **decoupled from the optional cluster cache** above. When the payload store is enabled, the chart resolves a store URL and **fails the install if none is found**, so a missing payload store surfaces at deploy time instead of silently failing >1MB requests at runtime.

URL resolution, in order:

1. `payloadStore.url`, if set: the terraform `payload_store_url` output (the `/payloads` prefix of the shared bucket), or any `s3://` / `gs://` / `abfs(s)://` / `oss://` URL the workload identity can read and write.
2. otherwise derived from `workers.common.clusterCache.url` by swapping the trailing `/models` prefix for `/payloads` (the same bucket).

```bash
# The terraform modules provision the bucket by default (create_model_cache=true)
# and expose its /payloads URL:
helm upgrade --install sie-cluster . \
  --set payloadStore.url=$(terraform output -raw payload_store_url)
```

To run without large-payload support (for example a local/dev cluster), opt out:

```bash
helm upgrade --install sie-cluster . --set payloadStore.enabled=false
```

> **Upgrade note:** the payload store is on by default. An existing queue-mode install with *no* payload store and *no* cluster cache will fail on upgrade until it either sets a URL (above) or `payloadStore.enabled=false`. Installs that already set `workers.common.clusterCache.url` keep working; the payload store derives its URL from it.

### Alibaba Cloud ACK OSS and RRSA

`values-ack.yaml` enables the payload store and configures every OSS-consuming
container for `eu-central-1`, the private OSS endpoint, and disabled ECS
metadata credentials. Install with Terraform's native OSS outputs and exact
RRSA role-name annotation:

```bash
helm upgrade --install sie-cluster . -f values-ack.yaml \
  --set workers.common.clusterCache.enabled=true \
  --set workers.common.clusterCache.url="$(terraform output -raw model_cache_bucket_url)" \
  --set payloadStore.url="$(terraform output -raw payload_store_url)" \
  --set-string 'serviceAccount.annotations.pod-identity\.alibabacloud\.com/role-name'="$(terraform output -raw rrsa_workload_role_name)"
```

The shared `sie-server` ServiceAccount and
`pod-identity.alibabacloud.com/injection: "on"` pod label engage ACK RRSA. Do
not render long-lived AccessKeys or `ALIBABA_CLOUD_STS_ENDPOINT`; the Terraform
webhook config leaves STS convenience-env injection off so the runtime keeps
its fixed official endpoint policy. Mutable `sie-config` state remains on its
local/PVC `SIE_CONFIG_STORE_DIR`; `oss://` is intentionally rejected for that
epoch/CAS authority.

ACK may install disk CSI classes without marking one as default. The ACK
overlay therefore binds the config, Prometheus, Grafana, and Loki claims to
`alicloud-disk-topology-alltype`, the topology-aware WFFC class, and raises the
config/Grafana claims to ACK's 20 GiB dynamic-disk minimum. The retained
capacities are 20 GiB config, 20 GiB Grafana, 100 GiB Prometheus, and 50 GiB
Loki. The built-in class does not itself prove encryption. Deployments
requiring encrypted PVCs must provide a governed encrypted WFFC class and
override all four paths together:

```yaml
config:
  configStore:
    storageClassName: your-encrypted-class
kube-prometheus-stack:
  prometheus:
    prometheusSpec:
      storageSpec:
        volumeClaimTemplate:
          spec:
            storageClassName: your-encrypted-class
  grafana:
    persistence:
      storageClassName: your-encrypted-class
loki:
  singleBinary:
    persistence:
      storageClass: your-encrypted-class
```

The overlay also runs `sie-config` as UID/GID 1000 with `fsGroup: 1000` and
`fsGroupChangePolicy: OnRootMismatch`, keeping the CSI volume writable without
granting root execution.

Storage class is not an ordinary values-only in-place migration for a bound
PVC, and a StatefulSet volume-claim template is immutable. For an empty failed
install, uninstall the exact release, independently prove that every retained
old claim is `Pending`, unbound, and has no volume, delete only those exact
empty claims, verify their absence, and then reinstall. StatefulSet and
operator-created claims can survive Helm uninstall. A data-bearing installation
requires a separately planned backup and volume migration; do not apply these
overrides as an in-place Helm-only change.

### Alibaba Cloud ACK ingress

`values-ack.yaml` sets `ingress.enabled=false` because the Alibaba Terraform
module does not install an ingress controller. Install and secure an
ACK-compatible controller such as ingress-nginx first, wait for its controller
and load balancer to be ready, and only then enable this chart's Ingress with a
matching `ingress.className`. Until then, use cluster-private service access or
an explicitly controlled port-forward.

## Autoscaling

KEDA-based autoscaling with scale-to-zero support:

```yaml
autoscaling:
  enabled: true
  # Scale-to-zero after 10 min idle
  cooldownPeriod: 600
  # Check metrics every 15s
  pollingInterval: 15
```

`autoscaling.enabled=true` is a complete telemetry dependency: it turns on
canonical OTLP metric emission, the bundled collector's Prometheus exporter,
and the collector ServiceMonitor. `keda.install` controls only whether this
chart installs the bundled KEDA controller; it does not control the chart's
ScaledObject manifests.
The ServiceMonitor uses a five-second interval and a timeout no greater than
that interval so KEDA never evaluates a stale or invalid scrape path.
Whenever `autoscaling.enabled=true`, the chart also renders mandatory
post-install/post-upgrade gates for every canonical Prometheus query and exact
KEDA ScaledObject/HPA health. `healthGates.enabled` independently enables the
optional gateway/config HTTP smoke Jobs; Helm `--wait` still requires those
workloads' Kubernetes readiness.

The chart stores hook-applied ScaledObjects in deterministic Helm-owned
ConfigMap shards of at most 32 worker lanes. Chart-managed autoscaling admits
192 physical worker lanes when KEDA and Prometheus are external, or 96 when
either dependency is installed by this release; the gateway's telemetry
runtime can still represent 1,024. These chart limits keep the complete Helm
release record conservative with maximum-length, high-entropy identities. With
Helm 3.16.4, the external profile encodes 192 lanes to about 840 KiB and the
worst bundled profile encodes 96 lanes to about 853 KiB, each leaving at least
64 KiB below the 917,504-byte release budget at this revision.
The ordinary post-install/post-upgrade hook applies the target shards and
prunes removed release-managed ScaledObjects. It refuses to adopt a same-name
object unless that object already carries this Helm release's exact identity.
Because those manifests are intentionally non-secret ConfigMaps, restrict
write access in the workload namespace to trusted control-plane principals.

When autoscaling is enabled, the gateway Deployment and worker StatefulSets
use Helm `lookup` during an upgrade to render their current live replica counts.
Fresh installs and offline/client-only renders use the configured initial
floors. This prevents Helm's resource patch from resetting the observed live
KEDA/HPA-controlled count without
storing a permanent replica-pin map in Helm. Use an actual Helm upgrade, or
`helm upgrade --dry-run=server` for a faithful preview; `helm template` and
client-only dry runs cannot exercise the live lookup.

The first migration from canonical `sie-cluster-0.6.20` KEDA metrics to the
collector-backed OTLP topology is one ordinary, supervised forward Helm
upgrade. Schedule a maintenance window, stop new caller demand, suspend other
reconcilers, and keep the release name, namespace, name overrides, worker/lane
keys, enabled lanes, scale targets, Prometheus backend, and KEDA ownership
(`keda.install`) unchanged. The only
exception is the documented over-limit catalog reduction below. Then run the
normal target upgrade with hooks and waiting enabled:

```bash
helm upgrade <RELEASE> oci://ghcr.io/superlinked/charts/sie-cluster \
  --namespace <NAMESPACE> \
  --version <TARGET_CHART_VERSION> \
  -f <REVIEWED_TARGET_VALUES_FILE> \
  --wait --timeout 20m
```

The mandatory Prometheus gate first proves the collector-backed metrics. The
target apply hook then updates the existing ScaledObjects in place with the same
names and scale targets, and the KEDA/HPA gate proves their queries before Helm
succeeds.

Do not use `helm rollback`, `helm upgrade --atomic`, `--cleanup-on-fail`, or
`--no-hooks` across this boundary. If a hook fails, keep demand stopped, repair
the collector, Prometheus, KEDA, image, quota, or permission issue, and rerun
the same forward upgrade. There is no extra migration command, pause
annotation, values overlay, second release, or target-owned migration state.
Canonical `0.6.20` can leave one inert hook ConfigMap containing obsolete
manifest text. After the target health hooks succeed, inspect that exact source
object and remove it once; deletion is post-success hygiene, not part of the
cutover:

```bash
LEGACY_KEDA_CONFIGMAP="<SOURCE_FULLNAME>-keda-scaledobjects"
kubectl get configmap "$LEGACY_KEDA_CONFIGMAP" -n <NAMESPACE> -o yaml
# Delete only after confirming component=keda-apply and the old Helm hook annotations.
kubectl delete configmap "$LEGACY_KEDA_CONFIGMAP" -n <NAMESPACE> --ignore-not-found
```

Fresh installs and later compatible releases use the normal Helm procedure.

Disabling autoscaling or uninstalling runs a small hook that deletes only
ScaledObjects carrying this release's managed identity or the exact canonical
`0.6.20` identity. Do not pass
`--no-hooks`. If the same operation also moves `global.namespace`, disable
autoscaling in the old namespace first, then move the already-static release.
Foreign, repackaged, or manually relabelled historical ScaledObjects remain an
explicit cluster-administrator cleanup responsibility.

Gateway/config Deployment, worker StatefulSet, and image-prepull DaemonSet
selectors retain the published `0.6.20` label identity because Kubernetes
selectors are immutable. Physical queue identity lives in the separate
`sie.superlinked.com/queue-pool` label. Target KEDA queries are
collector-backed and canonical-only; the temporary compatibility is the
unchanged ScaledObject name and scale target, not a second metric path.

The KEDA readiness hook selects the exact-revision ScaledObjects and all
release-owned HPAs, then requires their names and ownership labels to match
one-for-one. It waits a complete trigger-failure window and
requires every trigger to be `Happy` with zero failures. Reads are paged in
batches of 32 through the 192-lane chart domain. Near that maximum, use a Helm
client timeout of at least 20 minutes. Helm applies that timeout to each
Kubernetes operation/hook; it exceeds the longest default 15-minute Job. With
a custom `pollingInterval` above 30 seconds, also make the timeout exceed the
KEDA health deadline of `3 * pollingInterval + 240` seconds.

### Scale-from-Zero Trigger

The gateway emits `sie.gateway.pending_demand` over OTLP when requests arrive
for queue lanes with no available workers. The collector exposes that one
observation as
`sie_gateway_pending_demand{pool="...",machine_profile="...",bundle="..."}`;
KEDA uses it to trigger scale-up even when there are 0 workers.
For gpu-agnostic cold requests, including `X-SIE-Pool` requests that omit
`X-SIE-MACHINE-PROFILE`, the gateway emits concrete lane signals for every
machine profile the backing pool can provision. Multi-profile pools therefore
wake candidate lanes from zero without relying on an empty `machine_profile`
label that KEDA cannot match.

### Scaling Metrics

| Metric | Source | Purpose |
|--------|--------|---------|
| `sie_gateway_pending_demand` | Gateway | Trigger scale from 0 |
| `sie_gateway_lane_queue_depth` | Gateway JetStream backlog reconciler | Scale up on exact durable-consumer `num_pending + num_ack_pending`, including work held by a dead/loading worker |
| `sie_gateway_lane_queue_snapshot_timestamp_seconds` | Gateway JetStream backlog reconciler | Prove that the exact lane queue value, including zero, came from a fresh successful broker read |
| `sie_gateway_active_lease_gpus` | Gateway pool manager | Hold distinct assigned-worker capacity for active pool leases in the exact pool/profile/bundle lane |
| `sie_gateway_pool_warm_floor` | Gateway pool manager | Enforce a configured per-lane minimum without bypassing KEDA |
| `sie_gateway_rejected_requests_total` | Gateway | Scale up on sustained capacity/no-worker rejections after retryable cold-load reasons are excluded |
| `sie_gateway_requests_total` | Gateway request facade | Scale the gateway Deployment from its release-scoped request rate |
| `sie_gateway_capacity_snapshot_timestamp_seconds` | Gateway capacity reconciler | Reject stale gateway-to-collector state before it can drive scaling |

The bundled collector separates OTLP ingress by trust class. Gateway pods in
the exact Helm release use `4317`/`4318`; config and worker pods use the
application receiver on `4327`. A generated ingress `NetworkPolicy` enforces
the release/component selectors, while Prometheus exposition stays reachable
only to same-namespace pods and explicitly configured scrape namespace names
(`observability.otel.collector.prometheus.networkPolicy.scrapeNamespaceNames`).
A bundled Prometheus in another namespace is admitted by its exact
operator-owned identity. An external Prometheus is not: autoscaling renders
only when the list explicitly names every namespace containing scraper pods
(including the workload namespace when that is where external Prometheus
runs). Kubelet health probes need no `13133` ingress rule.
Treat the effective workload namespace (`global.namespace` when set,
otherwise the Helm release namespace) as a workload trust boundary and do not grant
untrusted principals pod-create or label-spoofing rights there.

Every application selector is bound to this release's exact collector target
with `namespace`, `service`, `endpoint="prometheus"`, and
`producer_service="sie-gateway"`. The ServiceMonitor makes its target labels
authoritative (`honorLabels: false`) and copies collector-exported producer
identity into `producer_service`/`producer_instance`. Before HA aggregation,
each non-queue business series is joined on `producer_instance` to the global
capacity timestamp from that same gateway process. The lane queue series uses
its stricter same-label broker-snapshot timestamp and requires a matching sample
count greater than zero, so a successful explicit zero differs from a missing
read. Every value is returned only while the collector target is up and at
least one release-scoped global snapshot is less than 20 seconds old. This
prevents retained points from a terminated gateway replica from being
legitimized by a fresh replica. Empty results are errors
(`ignoreNullValues: "false"`), so worker ScaledObjects enter their bounded
fallback instead of treating a broken or stale telemetry path as zero.

For a lane configured with `minReplicas: 0`, the chart renders
`idleReplicaCount: 0` plus a non-zero `minReplicaCount` equal to that lane's
bounded fallback floor. KEDA therefore still scales an inactive, healthy lane
to zero, but its HPA has a safe activation/fallback floor when Prometheus or the
gateway-to-collector path fails. Lanes with a positive configured minimum keep
that static minimum and omit `idleReplicaCount`. `autoscaling.fallbackReplicas`
must be at least one and is clamped to each lane's declared maximum.

KEDA scales worker StatefulSet replicas. It does not add or remove Python
processes or containers inside a running worker pod, so a multi-GPU worker is
one larger scaling unit with richer per-replica pressure metrics.

`alertRules.enabled=true` and the bundled `kube-prometheus-stack` are also
complete application-metrics consumers: either setting enables canonical OTLP
metrics, the bundled collector Prometheus exporter, and its ServiceMonitor.
The chart therefore cannot render collector-scoped application alerts without
a scrape path.

### Worker-sidecar Telemetry

When `workers.common.workerSidecar.enabled=true`, each worker Pod includes a
`worker-sidecar` container. The sidecar image is
`ghcr.io/superlinked/sie-server-sidecar`, the Rust binary is
`sie-server-sidecar`. The application emits canonical `sie.worker.*` metrics
once over OTLP. The collector translates those observations to the
`sie_worker_*` Prometheus families; the worker Service has no metrics port and
Prometheus never scrapes application containers.

For multi-child workers, the same worker Service also exposes adapter child
metrics ports `http-1` through `http-(N-1)` in addition to the baseline `http`
port, but those ports are for request traffic, not Prometheus. Availability
alerts use kube-state-metrics, while queue pressure comes from the
gateway-owned KEDA contract. Runtime pressure still travels in the worker
heartbeat that feeds the gateway; it is not duplicated as application
Prometheus instrumentation.

## Configuration

See `values.yaml` for all options. Key settings:

**Important**: All worker pools are disabled by default. You must explicitly enable
the pools you need in your values override.

> **Bundle rename:** the CUDA 13 SGLang bundle formerly named `gemma` is now
> `sglang-cu130`, and its image tag is `cuda13-sglang-cu130`. Existing values
> overrides must rename the `workers.common.bundlePlatforms` key, worker bundle
> key, and any explicit image tag together. There is no runtime alias because
> the bundle name is part of worker routing and release-image identity.

The values below are an illustrative shape; concrete per-cluster sizes belong
in each deployment's own values file.

```yaml
# Worker pool configuration (must explicitly enable pools)
# The map key is the Kubernetes capacity family/resource name. When
# machineProfile is omitted, it defaults to that map key. machineProfile is the
# runtime lane label used by routing and metrics. queuePool is the physical
# NATS queue namespace consumed by the workers; by default all worker groups
# use the shared `default` queue pool so SDK calls can pass just
# gpu="<profile>". Set queuePool on a worker group only for dedicated physical
# capacity, then create logical pools backed by it through `/v1/pools`
# (`queue_pool: "<queuePool>"`) and target them as
# gpu="<logicalPool>/<machineProfile>".
# With poolAdmission enabled, named non-default queue pools pull only when
# their physical queue pool is admitted directly (staticQueuePools) or when an
# assigned logical pool is backed by that queue.
# Each worker group renders its own StatefulSet + ScaledObject named
# worker-<pool>-<bundle>.
# Physical lane tokens are canonicalized once (trim, validate, lowercase), and
# duplicate canonical queuePool/machineProfile/bundle tuples fail rendering.
workers:
  pools:
    l4:
      enabled: true       # Enable this pool (disabled by default)
      bundles:
        default:
          minReplicas: 0  # Scale to zero
          maxReplicas: 10
    rtx6000:
      enabled: true
      bundles:
        default:          # embedding/rerank baseline
          minReplicas: 1
          maxReplicas: 5
        sglang:           # generation, warm baseline on same GPUs
          minReplicas: 1
          maxReplicas: 5
          preloadModels:
            - Qwen/Qwen3-4B-Instruct-2507

# Gateway configuration
gateway:
  replicas: 2

# Autoscaling
autoscaling:
  enabled: true
  cooldownPeriod: 600  # 10 min before scale-down
```

### Multi-GPU Worker Shapes

Req6 separates the Kubernetes allocation shape from the worker execution shape.
Set `workers.pools.<name>.gpu.count` above `1` when one worker pod should
consume multiple GPUs from a single VM. The chart requests that many
`nvidia.com/gpu` devices and renders one adapter worker child per GPU.

The queue-mode multi-GPU shape is one pod with one `worker-sidecar` plus
`worker-0` through `worker-(N-1)` adapter containers. Python/PyTorch and
Rust/Candle children use the same sidecar IPC fanout. Each child requests one
GPU, sees `SIE_DEVICES=cuda:0` when the runtime consumes that env var, binds a
unique HTTP/probe port, and serves a unique IPC socket. The sidecar receives
`SIE_IPC_SOCKET_PATHS`, owns the pod-level queue consumer, and places models
onto child sockets by child readiness, placed-model count, pending scheduler
cost, pending item count, and in-flight batch count. That placement scoring is a
fixed sidecar policy.

For local CPU-only dev, `workers.pools.<name>.sidecar.emulatedChildCount` can
render multiple CPU worker children with distinct IPC sockets and no
`nvidia.com/gpu` request. This is intended for local integration coverage of child
routing and metrics only; GPU pools must use `gpu.count` for real capacity.

This is the Req6 one-child-at-a-time model placement topology. It distributes
different models from one runtime bundle across the pod's GPU slots. A child can
own multiple models, but one model is not spread across children, replicated for
throughput, or tensor/model-parallelized for an oversized model.

Each worker pod serves exactly one runtime bundle. Different bundles render as
different worker StatefulSets and therefore different worker identities.

Example AWS shape for a 4-GPU L4 node:

```yaml
workers:
  pools:
    l4-4x:
      enabled: true
      machineProfile: l4-4x
      gpuType: nvidia-l4
      gpu:
        count: 4
        product: NVIDIA-L4
      bundles:
        default:
          minReplicas: 0
          maxReplicas: 3
```

### Queue Pool Patterns

Use one of these patterns deliberately:

- **Shared baseline pool**: leave `workers.common.queuePool: default`.
  Workers render with `SIE_POOL=default` and their own
  `SIE_MACHINE_PROFILE`; SDK calls use `gpu="<machineProfile>"`.
  API-created logical pools can use arbitrary valid names over this lane by
  omitting `queue_pool`.
- **Static custom queue namespace**: set a worker group's `queuePool` to a
  named value and declare the same name under
  `queueRouting.staticQueuePools`. These pool objects are synthesized by the
  gateway at startup and do not expire. Queue pool names are rendered and
  routed in lowercase. Example:

  ```yaml
  queueRouting:
    staticQueuePools:
      company-a:
        gpus:
          l4: 0
        gpuCaps: {}
  workers:
    pools:
      l4:
        queuePool: company-a
  ```

  `gpuCaps: {}` means uncapped admission for matching workers. Use
  `gpuCaps: {l4: 10}` to cap admission for that machine profile.
- **Dynamic isolated pool**: set a worker group's `queuePool` to the dedicated
  physical queue name, declare the same name under
  `queueRouting.staticQueuePools`, keep
  `queueRouting.poolAdmission.enabled=true`, and create/renew logical pools
  through `/v1/pools` with `queue_pool` set to that physical queue. Use a
  logical pool name that does not collide with the protected static queue-pool
  name, or target the static pool directly. SDK calls use
  `gpu="<logicalPool>/<machineProfile>"`.

Missing named pools intentionally do not fail open. Falling back from
`pool=default,machineProfile=l4` to `pool=l4,machineProfile=l4` would cross the
logical capacity boundary without an explicit caller request.

For emergency or legacy static namespaces that are not backed by either
`queueRouting.staticQueuePools` or a logical `/v1/pools` object with matching
`queue_pool`, disabling
`queueRouting.poolAdmission.enabled` lets workers pull without the admission
gate. Prefer declaring static pools instead, so capped/dynamic pools keep their
fail-closed isolation behavior.

### Work-queue durability (memory vs file storage)

The work queues ship **memory-backed and single-replica**, which is the only
behavior this chart has ever had. `WORK_POOL_{pool}`, the per-worker
direct-dispatch streams, and `DEAD_LETTERS` live in the NATS broker's RAM. That
is durable against *worker* failure — a pod that dies mid-item leaves the item
unacked and another worker picks it up — but **not** against *broker* failure. A
restart of the NATS pod erases every queued and delivered-but-unacked item along
with the dead-letter records that would have named them, and because the gateway
is queue-only with no direct-HTTP fallback, the restart is a full inference
outage for its duration.

The trigger is specifically a restart of the **NATS** pod: an OOM kill, a NATS
version or config rollout, or a node drain that evicts that pod. Draining a node
that hosts only workers or gateways is harmless here, as is a worker or gateway
rollout. At the default `streamReplicas: 1` any eviction of that single pod
loses the state; with a replicated stream it survives while one peer holding it
stays up, which is what makes rolling NATS maintenance safe.

To opt into file-backed, replicated storage:

```yaml
queueRouting:
  streamStorage: file   # default: memory
  streamReplicas: 3     # default: 1
nats:
  config:
    cluster:
      enabled: true     # required for streamReplicas > 1
      replicas: 3
    jetstream:
      fileStore:
        enabled: true   # required for streamStorage=file
        pvc:
          size: 10Gi
```

`streamStorage` and `streamReplicas` are rendered into `SIE_STREAM_STORAGE` /
`SIE_STREAM_REPLICAS` on **both** the gateway and the worker sidecars. Both
create streams and whoever gets there first wins, so the chart is the single
source of truth for the pair; the template rejects a render where the requested
durability outruns what the broker is configured to provide, and both env names
are reserved, so a `gateway.extraEnv` or
`workers.common.workerSidecar.extraEnv` entry that tried to override one on a
single side fails the render rather than silently splitting the two creators.

That render guard is load-bearing for `streamReplicas`, because **NATS will not
catch the mistake for you**: a single-node server accepts `num_replicas: 3` and
then reports R3, with no peers to replicate to (verified against nats-server
2.12.6). Without the guard, a replicated-looking stream would run
single-copy.

**The trade-off.** File storage puts a disk write in front of every publish
ACK, so it costs publish latency on the inference hot path, and it needs a PVC
per NATS pod sized for peak queue depth (`max_msgs` is 100,000 per pool stream)
plus 24 h of `DEAD_LETTERS` retention. Replication multiplies that cost by the
replica count and adds a quorum round trip. This is why the default is left
memory-backed: whether the latency and disk are worth the durability depends on
the workload, and that is an operator decision, not a chart default.

**Changing it on a live cluster.** JetStream refuses an in-place storage-type
change (err_code 10052), so flipping `streamStorage` does **not** convert
existing streams. The gateway and sidecars log a warning naming the divergence
and leave the streams alone, which is deliberate: converting would mean deleting
and recreating the stream, destroying exactly the queued work that durability is
meant to protect. Drain the pool and delete the streams during a maintenance
window to pick up the new storage. `streamReplicas`, by contrast, **is**
reconciled onto existing streams.

### Upgrading from the legacy single-bundle pool schema

Releases up to and including 0.4.x used a flat schema where each pool
declared a single `bundle:` plus `minReplicas:`/`maxReplicas:` at the
pool level. That shape is no longer accepted — `bundles:` is required
(see schema docs in `values.yaml`).

The rename also changes resource names from `worker-<pool>` to
`worker-<pool>-<bundle>` for StatefulSets, KEDA ScaledObjects, PDBs, and
the image-prepull DaemonSet. `helm upgrade` creates the new resources
but does not delete the old ones. The legacy resources are
distinguishable from the new ones by the absence of the
`sie.superlinked.com/bundle` label:

```bash
NS=sie  # effective workload namespace: global.namespace or release namespace

# Pre-refactor worker family (no bundle label) — delete before/after upgrade
kubectl -n "$NS" delete statefulset,pdb,daemonset \
  -l 'app.kubernetes.io/component=worker,!sie.superlinked.com/bundle'

# Pre-refactor image-prepull DaemonSets
kubectl -n "$NS" delete daemonset \
  -l 'app.kubernetes.io/component=image-prepull,!sie.superlinked.com/bundle'

# Pre-refactor KEDA ScaledObjects. Current managed revisions are pruned by the
# chart; these older unlabeled objects remain an explicit one-time cleanup.
kubectl -n "$NS" delete scaledobject \
  -l 'app.kubernetes.io/component=worker,!sie.superlinked.com/bundle'
```

Run these once per cluster after the upgrade settles. Leftover
ScaledObjects will keep trying to scale deleted StatefulSets and spam
KEDA logs; leftover PDBs will block node drains.

## Ingress

Enable the Ingress with `ingress.enabled=true` and route traffic to the gateway by
hostname. Use the list-valued `ingress.hosts` to front the gateway with one or more
hostnames — each entry becomes an Ingress rule (and, when TLS is enabled, a SAN on
the cert):

```yaml
ingress:
  enabled: true
  className: nginx
  hosts:
    - sie.example.com
    - api.example.com
```

The singular `ingress.host` is the backward-compatible single-host shorthand; it is
ignored whenever `ingress.hosts` is non-empty. With neither set the chart renders a
host-less catch-all Ingress. All hosts share the single `ingress.tlsConfig.secretName`
(one multi-SAN certificate).

## TLS / HTTPS

The chart supports four TLS modes for the Ingress (set via `ingress.tlsConfig.mode`):

- `byo` — bring your own `kubernetes.io/tls` Secret (default, backward compatible).
- `cert-manager` — chart annotates the Ingress; [cert-manager](https://cert-manager.io/) provisions and renews the certificate. Default flavour is ACME (HTTP-01 challenge to Let's Encrypt); you can also point at an existing internal Issuer/ClusterIssuer.
- `self-signed` — chart bootstraps a self-signed root CA, a CA ClusterIssuer, and a leaf cert for the Ingress. Intended for air-gapped / on-prem / VPC-isolated clusters where Let's Encrypt is unreachable.
- `disabled` — no TLS resources rendered. Use when TLS is terminated upstream (cloud load balancer, sidecar, service mesh).

> **Exactly one cert-manager per cluster.** cert-manager's CRDs, webhooks, and `cert-manager` ClusterRoleBinding are cluster-scoped singletons. Two controllers racing on the same CRDs corrupt issuance state. The chart enforces this with a pre-install Job that aborts when bundled cert-manager would collide with an existing install — see "Bundling cert-manager" below.

Only HTTP-01 ACME challenges are supported by the chart. DNS-01 / wildcard certs (which require cloud-provider IRSA / Workload Identity for Route53 / Cloud DNS) are out of scope — set them up manually outside the chart and reference the resulting Secret via `mode: byo`.

### `mode: byo` — bring-your-own certificate

Create the TLS Secret yourself (e.g. from a corporate CA, ACM cert exported to a Secret, or an existing wildcard cert), then point the chart at it:

```bash
kubectl -n sie create secret tls sie-tls --cert=path/to/tls.crt --key=path/to/tls.key
```

```yaml
ingress:
  enabled: true
  className: nginx
  host: sie.example.com
  tlsConfig:
    enabled: true
    mode: byo            # default
    secretName: sie-tls  # default
```

**When to use this**: you already manage TLS centrally, or you have a wildcard cert from a corporate CA, or you need DNS-01 / non-ACME issuance.

### `mode: cert-manager` — automated issuance via cert-manager

Prerequisite: either install cert-manager once in the cluster (its CRDs are cluster-scoped and must exist exactly once), OR opt in to the bundled subchart (see "Bundling cert-manager" below — single-tenant clusters only).

External install (recommended for shared clusters):

```bash
helm repo add jetstack https://charts.jetstack.io && helm repo update
helm install cert-manager jetstack/cert-manager \
  --set crds.enabled=true -n cert-manager --create-namespace
```

For single-tenant clusters where SIE is the only workload, the chart can also
install cert-manager as an opt-in subchart:

```yaml
certManagerBundle:
  certManager:
    install: true
```

Then enable cert-manager mode in your SIE values:

```yaml
ingress:
  enabled: true
  className: nginx
  host: sie.example.com
  tlsConfig:
    enabled: true
    mode: cert-manager
    certManager:
      email: ops@example.com
      # Use Let's Encrypt staging while iterating to avoid the 50 new-cert/registered-domain/week prod limit (duplicate-cert limit is 5/week):
      # server: https://acme-staging-v02.api.letsencrypt.org/directory
      kind: ClusterIssuer  # cluster-scoped; share across namespaces. Use "Issuer" for namespace-scoped.
      create: true         # chart renders the Issuer/ClusterIssuer
```

The chart renders a `{kind}` named `{release-fullname}-letsencrypt-prod` (release-scoped to avoid collisions when multiple SIE releases share a cluster) and adds the appropriate `cert-manager.io/cluster-issuer` (or `/issuer`) annotation to the main Ingress. cert-manager populates `ingress.tlsConfig.secretName` (default `sie-tls`); the same Secret is referenced by the oauth2-proxy Ingress when auth is enabled.

Note: Helm's standard `fullname` collapses when the release name already contains the chart name, so `helm install sie-cluster …` produces `sie-cluster-letsencrypt-prod` (not `sie-cluster-sie-cluster-letsencrypt-prod`). If you override `certManager.name`, set the full intended name explicitly rather than expecting a particular default.

Issuer kind tradeoff:

- `ClusterIssuer` — single ACME account / private key shared across all namespaces. Best for shared clusters.
- `Issuer` — namespace-scoped. Use for hard tenant isolation, or when you don't have permission to create cluster-scoped resources.

**Reusing an existing ClusterIssuer/Issuer.** In multi-tenant clusters where a platform team already manages a shared `ClusterIssuer`, set `create: false` and reference it by name:

```yaml
ingress:
  tlsConfig:
    enabled: true
    mode: cert-manager
    certManager:
      kind: ClusterIssuer
      create: false
      name: platform-letsencrypt-prod
```

The chart only adds the annotation — it does not render any Issuer resource.

**When to use this**: ACME / Let's Encrypt is reachable from your cluster (or you already have an internal Issuer/ClusterIssuer) and your platform team is OK with cert-manager being installed.

### `mode: self-signed` — self-signed CA (air-gapped / on-prem)

For clusters that cannot reach Let's Encrypt — typical for on-prem, regulated, or VPC-isolated environments — the chart can bootstrap a self-signed root CA and use it to issue the Ingress leaf cert. cert-manager is still required.

```yaml
certManagerBundle:
  certManager:
    install: true      # bundle cert-manager (SINGLE-TENANT clusters only — see warning above)

ingress:
  enabled: true
  className: nginx
  host: sie.example.com
  tls:
    enabled: true
    mode: self-signed
    secretName: sie-tls
    selfSigned:
      rootCA:
        commonName: "Acme Corp SIE Root CA"
        # 43800h = 5y, 720h = 30d renewBefore
      leaf:
        # 2160h = 90d, 360h = 15d renewBefore — match Let's Encrypt lifetimes
        dnsNames: []   # extra SANs in addition to ingress.host
        ipAddresses: []
```

Chain:

1. `SelfSigned` ClusterIssuer (bootstrap, name `{fullname}-selfsigned-bootstrap`).
2. Root CA `Certificate` (`{fullname}-root-ca`, isCA, 5y, RSA-4096) -> Secret `sie-root-ca-key-pair`.
3. CA `ClusterIssuer` (`sie-self-signed-ca`) backed by the root CA secret.
4. Ingress leaf `Certificate` (`{fullname}-ingress-leaf`, ECDSA-P256, 90d) -> Secret `sie-tls`, consumed by the Ingress.

Clients (browsers, curl) must trust the root CA. Export it with:

```bash
kubectl -n sie get secret sie-root-ca-key-pair \
  -o jsonpath='{.data.ca\.crt}' | base64 -d > sie-root-ca.crt
```

> **Root CA namespace constraint.** Two independent namespace-scoped lookups apply:
>
> 1. **cert-manager** only resolves Secrets referenced by a `ClusterIssuer` inside its `--cluster-resource-namespace` (defaults to its own Deployment's namespace). The chart writes the root CA to `ingress.tlsConfig.selfSigned.rootCA.namespace` (defaults to the release namespace, which is correct for the **bundled** subchart since cert-manager also runs in the release namespace).
> 2. **trust-manager** only resolves source Secrets for `Bundle` resources inside its `--trust-namespace` (defaults to `cert-manager`, regardless of where trust-manager itself runs).
>
> If you also enable `certManagerBundle.trustBundle.enabled: true` with the bundled trust-manager, override the trust namespace at install time so it matches where the root CA lives:
>
> ```bash
> helm install ... --set "trust-manager.app.trust.namespace=<release-namespace>"
> ```
>
> Otherwise the Bundle stays `Synced=False` with `SourceNotFound`. For external cert-manager (typically in `cert-manager` namespace), set `ingress.tlsConfig.selfSigned.rootCA.namespace: cert-manager` so the CA `ClusterIssuer` can find its Secret; the default trust-namespace then already matches.

**When to use this**: air-gapped / on-prem clusters where you can distribute the root CA to client machines (e.g. via MDM, internal trust store, or workload mount), and you want a single `helm install` to land a working HTTPS path.

### `mode: disabled` — no TLS resources

Use when TLS is terminated upstream of the Ingress (cloud LB, sidecar, service mesh):

```yaml
ingress:
  enabled: true
  className: nginx
  host: sie.example.com
  tls:
    enabled: false
    mode: disabled
```

### Trust distribution with trust-manager

When `mode: self-signed`, you can replicate the root CA into other namespaces as a ConfigMap so non-SIE workloads can trust SIE without out-of-band copying.

```yaml
certManagerBundle:
  trustManager:
    install: true      # required when not already installed externally
  trustBundle:
    enabled: true
    name: sie-root-ca-bundle
    target:
      configMapKey: ca.crt
    namespaceSelector:
      matchLabels:
        sie.io/trust: "true"  # label target namespaces to opt them in
```

Workloads mount the resulting ConfigMap and point their HTTP client at it as the CA bundle.

### Bundling cert-manager

The chart can install cert-manager and trust-manager as opt-in subchart dependencies (default off). **This is reserved for single-tenant clusters where SIE is the sole workload.** In any multi-tenant or shared cluster, install cert-manager once out-of-band and leave `certManagerBundle.certManager.install: false`.

Guards:

1. Both subcharts are gated by `*.install` flags that default to `false`; default chart behaviour is unchanged.
2. A template-time `lookup` aborts the install when `certManagerBundle.certManager.install: true` is combined with an existing `certificates.cert-manager.io` CRD.
3. A `pre-install` Job hook re-checks at apply time and aborts before any subchart resources are created (`lookup` returns empty during `helm template` / `--dry-run`, so the Job is the real safety net).
4. Jetstack's `crds.keep: true` default is left in place, so `helm uninstall` does not silently delete `Certificate` / `Issuer` resources belonging to other operators.

Override the conflict guard (DANGER):

```yaml
certManagerBundle:
  allowExistingCRDs: true   # bypass both guards; only if you accept the consequences
```

#### Vendored CRDs

cert-manager and trust-manager CRDs are vendored at `deploy/helm/sie-cluster/crds/`. Helm applies files in a chart's `crds/` directory **before** rendering templates, which is the only mechanism that lets bundled mode complete in a single `helm install` (the subcharts' own templated CRDs would land too late — Helm's RESTMapper discovery runs at install start and fails to resolve `Certificate` / `Bundle` references). Both subcharts have `crds.enabled: false` set in `values.yaml` so they don't try to re-install the same CRDs.

The vendored bundles are pinned to the same version as the subchart pins in `Chart.yaml`:

- `crds/cert-manager.crds.yaml` — fetched from the cert-manager release: `curl -fsSL -o crds/cert-manager.crds.yaml https://github.com/cert-manager/cert-manager/releases/download/v<X.Y.Z>/cert-manager.crds.yaml`
- `crds/trust-manager.crds.yaml` — extracted from the subchart tarball: `helm template trust-manager charts/trust-manager-v<X.Y.Z>.tgz --show-only templates/crd-trust.cert-manager.io_bundles.yaml > crds/trust-manager.crds.yaml`

When bumping the subchart version pin in `Chart.yaml`, re-vendor both files and re-run the golden-diff tests to surface CRD schema changes.

### Uninstall caveat

`crds.keep: true` is the Jetstack default. `helm uninstall sie-cluster` will **leave cert-manager CRDs behind on purpose**, so that `Certificate` / `Issuer` resources owned by other operators are not silently deleted. To remove them, run `kubectl delete crd <name>.cert-manager.io <name>.acme.cert-manager.io <name>.trust.cert-manager.io ...` explicitly. See the [cert-manager uninstall docs](https://cert-manager.io/docs/installation/helm/#uninstalling) for the full CRD list.

## Gated Models

Some HuggingFace models require authentication to download (gated models). Examples:

- `google/embeddinggemma-300m` - Manual gating (requires approval)
- `naver/splade-v3` - Auto gating (requires license acceptance)

### Prerequisites

1. Create a HuggingFace account and generate an access token at <https://huggingface.co/settings/tokens>
2. For manually gated models, request access on the model page (e.g., <https://huggingface.co/google/embeddinggemma-300m>)
3. For auto-gated models, accept the license agreement on the model page

### Kubernetes Setup

Create a secret with your HuggingFace token:

```bash
kubectl create secret generic hf-token \
  --namespace sie \
  --from-literal=token=hf_your_token_here
```

Configure the Helm chart to use the secret:

```yaml
workers:
  common:
    hfCache:
      tokenSecret: hf-token      # Secret name
      tokenSecretKey: token      # Key within the secret
```

The token is mounted as the `HF_TOKEN` environment variable, which HuggingFace libraries automatically detect.

### Local Development

For local development, set the `HF_TOKEN` environment variable:

```bash
# Option 1: Direct export
export HF_TOKEN=hf_your_token_here
mise run serve

# Option 2: From file
export HF_TOKEN=$(cat ~/.secrets/hf_token)
mise run serve
```

### Docker

Pass the token as an environment variable:

```bash
docker run -e HF_TOKEN=hf_your_token_here \
  -p 8080:8080 \
  sie-server:cuda12-default
```

## Telemetry

SIE collects anonymous usage telemetry (version, OS, architecture, GPU type) to help maintainers understand adoption and hardware distribution. Telemetry is on by default and sends a lightweight heartbeat once per hour.

**No IP addresses, hostnames, cluster names, API keys, or request data are collected.**

Disable telemetry:

```yaml
telemetry:
  enabled: false
```

Enterprise customers can route heartbeats through their own collector:

```yaml
telemetry:
  url: "https://telemetry.internal.example.com/api/telemetry"
```

Tag non-production deployments to filter them out of dashboards:

```yaml
telemetry:
  deploymentEnv: staging  # production (default) | staging | development | ci
```

> **Non-production deployments:** set `telemetry.deploymentEnv` to one of
> `staging | development | ci`. The chart default is `production`, so every
> non-production values overlay must opt out explicitly to keep its signals out
> of production dashboards.

## Observability

Observability components (Prometheus, Grafana, Loki, Tempo, DCGM Exporter, Alloy, Event Exporter) are included as optional sub-chart dependencies. Enable them in your values overlay (e.g. `kube-prometheus-stack.install: true`, `observability.logs.install: true`, `observability.tracing.tempo.install: true`, or `kubernetes-event-exporter.install: true`).

Every enabled OpenTelemetry producer and every bundled-collector branch uses
the same canonical resource identity. Local installs may fall back to
`unknown`; a Better Stack forwarding collector fails Helm rendering unless the
dedicated environment is exactly `dev`, `staging`, or `prod` and the region is
explicit and non-unknown:

```yaml
observability:
  otel:
    resource:
      deploymentEnvironment: dev
      cloudRegion: us-east-1
```

### Better Stack OTLP destination

Use the Telemetry source's ingestion token. A Better Stack settings-management
API token and an Uptime API token are not runtime OTLP credentials. Copy the
bare HTTPS OTLP origin shown by that source; do not guess or hardcode a global
Better Stack hostname. Create one source and one Secret per environment and
region in the effective workload namespace:

```bash
WORKLOAD_NAMESPACE=sie
BETTER_STACK_SECRET=sie-betterstack-otlp-dev-us-east-1
read -rsp 'Better Stack Telemetry source token: ' BETTER_STACK_SOURCE_TOKEN
printf '\n'
printf '%s' "$BETTER_STACK_SOURCE_TOKEN" |
  kubectl create secret generic "$BETTER_STACK_SECRET" \
    -n "$WORKLOAD_NAMESPACE" --from-file=token=/dev/stdin \
    --dry-run=client -o yaml |
  kubectl apply -f -
unset BETTER_STACK_SOURCE_TOKEN
```

Reference it without placing the token in Helm values:

```yaml
observability:
  otel:
    resource:
      deploymentEnvironment: dev
      cloudRegion: us-east-1
    collector:
      betterStack:
        enabled: true
        endpoint: "<BARE_HTTPS_OTLP_ORIGIN_FROM_SOURCE_UI>"
        existingSecret: sie-betterstack-otlp-dev-us-east-1
        tokenKey: token
```

Only the collector Pod receives this Secret. Application producers receive the
collector endpoint, never the Better Stack token. Keep dev, staging, and prod
sources, dashboards, and Secrets isolated per region.

For remote logs and traces, those operated-environment values are
collector-authoritative rather than trusted from an application resource. The
gateway-only receiver also authors `service.name=sie-gateway`; the application
trace receiver accepts only the declared config, dispatcher, worker, and
worker-sidecar service names. Only the gateway receiver is connected to the
allowlisted request-completion log pipeline. A simultaneous local Tempo trace
pipeline intentionally bypasses these remote-only identity and privacy
processors and receives the producer trace unchanged.

The collector Deployment annotation `checksum/otel-config` is the SHA-256 of
the exact `collector.yaml` ConfigMap value mounted into the pod. Any collector
configuration change therefore creates a new pod template and rolls the
singleton collector instead of leaving it on stale mounted configuration.

Pre-configured dashboards:

- Cluster overview (QPS, latency, GPU utilization)
- Per-model performance
- Worker health
- Queue routing
- Generation
- Performance tuning
- SIE Tracing

### Distributed Tracing (OTLP)

Distributed tracing is **off by default** — the rendered chart is unchanged unless you opt in. Enabling injects the OpenTelemetry exporter env onto the gateway, worker sidecar, adapter worker, and Rust worker so a request is traced end to end (OTLP gRPC, `:4317`). Queue-mode endpoints (encode/score/extract/embeddings) publish through `gateway.publish` before `sidecar.dispatch` and `worker.run_batch`; generation (`/v1/generate`, `/v1/chat/completions`) also opens gateway-originated spans for the streaming path. Sampling defaults to a head-based parent sampler (`parentbased_traceidratio` at `0.05`), which honors an inbound `traceparent` decision and otherwise samples 5% of new traces.

**Bring your own collector** (Tempo, Jaeger, or an existing OTel Collector) is
available only when the bundled collector is not required by metrics, logs, or
KEDA:

```yaml
observability:
  tracing:
    enabled: true
  otel:
    endpoint: "http://tempo:4317"   # OTLP gRPC producer destination
```

Or with `--set`:

```bash
helm upgrade ... \
  --set observability.tracing.enabled=true \
  --set observability.otel.endpoint=http://tempo:4317
```

**Bundled collector** is installed explicitly or implied by application
Prometheus consumers such as KEDA. Leave
`observability.otel.collector.traces.endpoint` empty to debug-log spans in the
collector pod (handy for a first run), or set it to forward to Tempo/Jaeger:

```yaml
observability:
  tracing:
    enabled: true
  otel:
    collector:
      install: true
      traces:
        endpoint: "http://tempo:4317"   # optional downstream; omit to debug-log
        insecure: true                  # false for a TLS-enabled downstream
```

The bundled collector forwards over plaintext by default (`insecure: true`);
set it to `false` when the trace endpoint uses TLS.
`observability.otel.endpoint` is mutually exclusive with the bundled collector;
it does not suppress a collector forced by autoscaling, ServiceMonitor, alert
rules, kube-prometheus-stack, or `observability.otel.collector.install`.

**Bundled Tempo backend** (opt-in; never installed by default) — renders Grafana Tempo as an in-cluster trace backend via the `grafana-community/tempo` 2.2.3 chart (Tempo app 2.10.7) from the grafana-community repo. With tracing enabled and no explicit endpoint or bundled collector, the gateway and workers automatically export OTLP gRPC spans to the namespace-qualified Tempo Service on port 4317. The Tempo query API is exposed on port 3200, and the chart renders a Grafana Tempo datasource ConfigMap for the already-enabled Grafana datasource sidecar. Namespace-qualified DNS keeps this working when `global.namespace` differs from the Helm namespace:

```yaml
observability:
  tracing:
    enabled: true       # required for gateway/worker span emission
    tempo:
      install: true     # installs Tempo and defaults pods to its OTLP Service
kube-prometheus-stack:
  install: true         # required for bundled Grafana to pick up the datasource
```

Installing Tempo without `observability.tracing.enabled=true` is allowed; it
creates an idle backend and, when bundled Grafana is installed, a datasource.
Installing both the bundled collector and bundled Tempo points pods at the
collector and makes the collector forward to Tempo's namespace-qualified OTLP
Service unless `observability.otel.collector.traces.endpoint` is set. A direct
`observability.otel.endpoint` is valid only without the bundled collector. The
downstream trace endpoint must not name this release's own OTel collector
Service; the chart rejects that export loop.

The SIE Tracing dashboard renders with the standard dashboards gate (`dashboards.enabled=true` or `kube-prometheus-stack.install=true`). Its Tempo panels require a Grafana datasource with `uid: tempo`; the chart auto-provisions that datasource only when both `observability.tracing.tempo.install=true` and `kube-prometheus-stack.install=true`. External Grafana installs, or bundled Grafana pointed at an external Tempo through an OTel endpoint, must provision the datasource themselves or the trace panels will report "datasource not found".

Bundled Tempo enables a persistent volume (`~10Gi`) and requires a default StorageClass in the target cluster; otherwise the Tempo pod will not schedule. The upstream chart also exposes unused legacy receiver Service ports (`9411`, `55680`, `55681`) even though SIE only configures OTLP gRPC ingest on `4317` and the query API on `3200`.

Tunables: `observability.tracing.sampler` / `samplerArg` (sampling), and
`observability.otel.serviceName.{gateway,config,worker,workerSidecar}` for the
canonical cross-signal `service.name`. Defaults are `sie-gateway`, `sie-config`,
`sie-worker`, and `sie-worker-sidecar`; the collector's allowlist is built
around those identities. When tracing is enabled, configure
`observability.otel.endpoint`, install/require the bundled collector, or install
Tempo, otherwise the chart fails fast.

Local / non-Helm note: the gateway, Python worker, and Rust worker-sidecar all require `SIE_TRACING_ENABLED=true` and an OTLP endpoint (`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, or `OTEL_EXPORTER_OTLP_ENDPOINT`) before exporting traces. Setting only one yields no traces rather than a partial trace. In-cluster, the Helm chart sets both for you.
