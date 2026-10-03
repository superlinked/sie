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
| `sie` | Native `encode` (dense, sparse and multivector), `score`, `extract`, and buffered or streaming `generate`; supported image inputs for non-generation primitives |
| `openai` | Dense text embeddings at `/embeddings`, Cohere-shape reranking at `/rerank`, chat at `/chat/completions`, and native prompt generation through `/completions` |

Native SIE generation uses the upstream's `/v1/generate` endpoint, with exact
terminal usage. Single-node and queued `/v1/chat/completions` serving also forward to the
SIE upstream's chat endpoint. OpenAI profiles require the corresponding
`chat` or `completions` endpoint declaration. Both generation modes require
exact final usage and bounded responses; streaming cancellation closes the
upstream call. Queued remote-backed chat uses the upstream chat endpoint without
loading a local tokenizer. An onboarded queued OpenAI model prefers its local
chat template and raw completions when that endpoint is declared; strict JSON
Schema requests, enforced tool choices, multiple choices, and chat-only
deployments use upstream chat. Native raw completions refuse
media, grammar and nonstandard sampling fields they cannot enforce; strict
JSON Schema output is supported through chat and verified locally. Queued chat
preserves tools, tool choice, choice indices and usage; unsupported media, local
chat-template kwargs and nonstandard sampling fields fail before dispatch.

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

SIE encode can succeed when upstream usage is absent. In that case the adapter
omits `input_token_counts` instead of estimating them. Malformed usage fails
the request. Native generation requires terminal usage for successful output.

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

## Local profile identity

The single-node model detail and catalog responses include
`profiles.<name>.identity`, a versioned digest or `null`. Version 1 conservatively
identifies revision-pinned native BGE-M3 profiles. It includes model/tokenizer pins,
resolved profile settings, engine configuration, device/platform, serving
Python sources, installed inference-library versions and current Torch
precision/determinism settings. Alias names and
inheritance do not change a profile with identical resolved settings.

Local weight paths, mutable revisions, custom/checkpoint code, child engines,
unidentified precision and LoRA-bearing models report `null`. SentenceTransformers
and CrossEncoder require verified checkpoint module metadata: disabling
`trust_remote_code` alone does not identify installed checkpoint-selected code. FlagEmbedding
BGE-M3 cannot identify its effective revision; flash/LoRA and other engines need
additional runtime evidence. An identity is a descriptor, not a numerical
measurement. Hybrid routing remains refused until upstream comparison and
measured equivalence gates are delivered; this field alone does not activate it.
