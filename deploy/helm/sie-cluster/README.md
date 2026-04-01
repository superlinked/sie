# SIE Cluster Helm Chart

Deploy SIE (Search Inference Engine) to Kubernetes with autoscaling and observability.

## Quick Start

```bash
helm install sie-cluster oci://ghcr.io/superlinked/charts/sie-cluster \
   --namespace sie \
   --create-namespace
  --namespace sie \
  --create-namespace
```

## Architecture

```
┌─────────────┐     ┌─────────────────────────────────────┐
│   Client    │────▶│           Router (2+ replicas)      │
└─────────────┘     └───────────────┬─────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌─────────┐     ┌─────────┐     ┌─────────┐
              │ L4 Pool │     │A100 Pool│     │ CPU Pool│
              │ 0-N     │     │ 0-N     │     │ 0-N     │
              └─────────┘     └─────────┘     └─────────┘
```

- **Router**: Stateless proxy that routes requests to workers based on GPU type and model affinity
- **Worker Pools**: StatefulSets per GPU type, each with KEDA autoscaling

## Cold Start Expectations

When scaling from zero, expect the following latencies:

| Phase | Duration | Notes |
|-------|----------|-------|
| **Node provisioning** | 2-5 min | GKE/EKS spins up GPU node (spot may be slower) |
| **Container startup** | 20-40s | Pull image, start process, health checks |
| **Model loading** | 10-120s | Download weights (if not cached), load to GPU |
| **Total cold start** | 3-7 min | First request to a scaled-to-zero pool |

### Reducing Cold Start Time

1. **Use cluster cache**: Pre-populate S3/GCS with model weights (`--cluster-cache`)
2. **Set minReplicas=1**: Keep one warm replica per critical GPU type
3. **Use reserved capacity**: Avoid spot for latency-sensitive workloads
4. **Pre-warm models**: Call `/v1/encode/{model}` on startup to load weights

### Client Handling

When a pool is scaling from zero, the router returns:
- **202 Accepted** with `Retry-After: 120` header
- Client should retry after the indicated delay

The SDK handles this automatically with configurable retries.

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

### Scale-from-Zero Trigger

The router exposes `sie_router_pending_demand{gpu="..."}` metric when requests
arrive for GPU types with no available workers. KEDA uses this to trigger scale-up
even when there are 0 workers (and thus no worker metrics).

### Scaling Metrics

| Metric | Source | Purpose |
|--------|--------|---------|
| `sie_router_pending_demand` | Router | Trigger scale from 0 |
| `sie_request_queue_depth` | Workers | Scale up on load |
| `sie_active_requests` | Workers | Scale up on concurrent requests |

## Configuration

See `values.yaml` for all options. Key settings:

**Important**: All worker pools are disabled by default. You must explicitly enable
the pools you need in your values override.

```yaml
# Worker pool configuration (must explicitly enable pools)
workers:
  pools:
    l4:
      enabled: true     # Enable this pool (disabled by default)
      minReplicas: 0    # Scale to zero
      maxReplicas: 10
      gpuType: nvidia-l4
      gpu:
        count: 1
        product: nvidia-l4

# Router configuration
router:
  replicaCount: 2  # HA by default

# Autoscaling
autoscaling:
  enabled: true
  cooldownPeriod: 600  # 10 min before scale-down
```

## Gated Models

Some HuggingFace models require authentication to download (gated models). Examples:
- `google/embeddinggemma-300m` - Manual gating (requires approval)
- `naver/splade-v3` - Auto gating (requires license acceptance)

### Prerequisites

1. Create a HuggingFace account and generate an access token at https://huggingface.co/settings/tokens
2. For manually gated models, request access on the model page (e.g., https://huggingface.co/google/embeddinggemma-300m)
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

## Observability

Observability components (Prometheus, Grafana, Loki, DCGM Exporter, Alloy, Event Exporter) are included as optional sub-chart dependencies. Enable them in your values overlay (e.g. `kube-prometheus-stack.install: true`).

Pre-configured dashboards:
- Cluster overview (QPS, latency, GPU utilization)
- Per-model performance
- Worker health
