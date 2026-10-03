# Serving a model through a remote backend

A remote profile serves a catalog model through an operator-defined upstream.
The public model name stays the same whether the model runs here or elsewhere.
The worker makes the outbound call; a cluster gateway only routes and queues
work. Applications continue to use SIE's API and SDK.

This guide describes the implementation on `main`. Use a build containing the
linked changes; availability in a published package or image depends on its
release. Follow [issue #415](https://github.com/superlinked/sie/issues/415) for
remaining work.

## Supported operations and policies

| Upstream | Operations on `main` |
| --- | --- |
| `sie` | Native `encode` (dense, sparse and multivector), `score`, and `extract`, including supported image inputs |
| `openai` | Dense text embeddings at `/embeddings` and Cohere-shape reranking at `/rerank` |

Native SIE generation is being added in
[PR #514](https://github.com/superlinked/sie/pull/514). OpenAI-compatible remote
chat and completions are separate work; declaring an endpoint alone does not
implement generation.

A remote-backed model has only remote profiles. Its bare name serves remotely;
`routing: {policy: remote_only}` makes that policy explicit. It needs no local
weights or accelerator.

Single-node `fallback` is available for extraction-only models. A request that
names a profile bypasses the bare-model policy. Hybrid `encode` and `score`
remain refused at configuration load until identity or equivalence proof can
be checked. Generation fallback and `threshold` are also refused. Cluster
remote profiles use the queue, but cluster fallback is still being built.

## Single-node embedding example

Start with another SIE deployment that already serves
`sentence-transformers/all-MiniLM-L6-v2`. Create `upstreams.yaml`:

```yaml
upstreams:
  team-sie:
    kind: sie
    base_url: https://sie.example.internal
    api_key_secret: TEAM_SIE_KEY
    rate_cap:
      requests_per_minute: 600
      max_concurrency: 8
```

`TEAM_SIE_KEY` is the name of an environment variable, not its value. Supply
that variable through your deployment's secret mechanism. The adapter reads
it when sending a request. Do not put a key in the YAML or a URL.

Create `remote-models/sentence-transformers__all-MiniLM-L6-v2.yaml`:

```yaml
sie_id: sentence-transformers/all-MiniLM-L6-v2
remote_backed: true
routing:
  policy: remote_only
max_sequence_length: 256
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.remote.sie:SieUpstreamAdapter
    max_batch_tokens: 8192
    adapter_options:
      loadtime:
        upstream: team-sie
        upstream_model: sentence-transformers/all-MiniLM-L6-v2
```

The declared dimensions and input limit must describe the upstream model.
`remote_backed: true` excludes local weight-source fields such as `hf_id`.
The model configuration can name an upstream; it cannot create one.

From the repository root, start the server:

```bash
mise exec -- uv run --frozen --project . --package sie-server sie-server serve \
  --models-dir ./remote-models --upstreams-file ./upstreams.yaml --device cpu
```

Call it through the SDK:

```python
from sie_sdk import SIEClient

with SIEClient(base_url="http://localhost:8080") as client:
    result = client.encode(
        "sentence-transformers/all-MiniLM-L6-v2", [{"text": "Find related documents"}]
    )[0]
    print(result["request"].get("served_by"))  # remote
    print(result["request"].get("upstream"))   # team-sie
```

The upstream must return usage for successful work. Missing or malformed usage
fails the request instead of substituting an estimated count.

## OpenAI-compatible embeddings and rerank

Define a `kind: openai` upstream with a base URL that includes the provider's
API prefix, for example `https://host.example.com/v1`. Declare only endpoints
it offers: `endpoints: [embeddings, rerank]`. Use
`sie_server.adapters.remote.openai:OpenAIUpstreamAdapter` in the profile and
set `upstream_model` to the provider's model id.

OpenAI-compatible embedding profiles support dense text only. Sparse,
multivector, image input and extraction are rejected before dispatch. A rerank
upstream must accept the Cohere-shaped request and report usage. SIE restores
scores to document order rather than exposing the provider's ranked order.

The operator may add `set_params` and `strip_params` on the upstream. Reserved
input and response-shaping fields, including `model`, cannot be overridden or
stripped. Callers cannot pass additional upstream fields or credentials.

## Controls and failure behavior

No upstream is configured by default. Without an upstream and a remote
profile, nothing is sent remotely. Once configured, the global serving switch
is on unless disabled. Start with `--no-remote-serving` or set
`SIE_REMOTE_SERVING=false` to refuse all remote serving. Unknown environment
values also disable it.

A caller can narrow that permission with `X-SIE-Remote: forbid`, or
`SIEClient(..., remote="forbid")`. Other header values are rejected. A
remote-backed model has no local alternative, so forbidding remote serving
returns a client error.

Responses disclose `X-SIE-Served-By` and, when remote, `X-SIE-Upstream`.
Single-node fallback adds `X-SIE-Fallback-Reason`; a failed remote attempt
returns the original local refusal and `X-SIE-Fallback-Error`. The SDK exposes
these fields in response request metadata. `/v1/models` reports routing and
upstream kind.

Each upstream has a required rate cap and a circuit breaker. The limits are
shared by that worker process's adapters, not across replicas: adding remote
worker replicas increases the aggregate permitted traffic. Under
`remote_only`, unavailable upstreams, open breakers and reached caps return
retryable `503` responses. Under single-node `fallback`, an upstream failure
returns the original local refusal with its retry delay. A client error is
never retried remotely, and fallback cannot replay work after local acceptance
or after output reaches the caller.

TLS is required outside loopback. URLs containing credentials, a query or a
fragment are rejected. Redirects are refused, ambient proxy variables are
ignored, and inbound authorization is not forwarded. An explicit `proxy_url`
is the supported egress proxy setting.

## Cluster deployment

Use the Helm chart's [remote worker pool instructions](../../deploy/helm/sie-cluster/README.md#remote-backends).
Only remote worker containers receive upstream configuration and credentials;
the gateway, config service and local workers do not. The chart requires
network-policy coverage for a remote lane. Its standard policy filters CIDRs
and ports, not DNS hostnames. Use the chart's documented egress proxy or CNI
host allowlist when you need to restrict traffic to declared hosts.

See [Moving to self-hosted serving](MIGRATING_TO_SELF_HOSTED.md) for a migration
that keeps the application model name stable.
