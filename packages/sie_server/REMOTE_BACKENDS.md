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
loading a local tokenizer. Single-node and queued onboarded models prefer their
local chat template and tool parser through native SIE generation or declared
OpenAI raw completions. Strict JSON Schema requests, strict tool schemas,
enforced tool choices, multiple choices, and chat-only
deployments use upstream chat. Native raw completions refuse
media, grammar and nonstandard sampling fields they cannot enforce; strict
JSON Schema output is supported through chat and verified locally. Queued chat
preserves tools, tool choice, choice indices and usage; unsupported media, local
chat-template kwargs and nonstandard sampling fields fail before dispatch.
Single-node chat retains upstream chat ownership for request fields the raw
generation contract cannot represent, including MLX sampling and role mapping.
Selection completes before dispatch; a started raw request is never retried as
chat. Native SIE already suppresses private reasoning from its rendered prompt;
the receiving worker preserves its answer and strips any explicit private blocks.

A remote-backed model has only remote profiles. Its bare name serves remotely;
`routing: {policy: remote_only}` makes that policy explicit. It needs no local
weights or accelerator.

Single-node `fallback` is available for extraction-only models and for models
with local and remote generation profiles whose remote profile produces every
declared output (see [Single-node generation fallback](#single-node-generation-fallback)).
A request that names a profile bypasses the bare-model policy, including an
explicit `default`. Single-node OpenAI hybrid `encode` and `score` require the
operator-owned equivalence admission described below. SIE hybrid encode and
score require the fresh identity admission described below. Cluster remote
profiles use the queue. Cluster generation and extraction fallback are described
below. Cluster saturation and unhealthy-worker spill require explicit triggers.
Cluster `encode` and `score` bridges need a numerical admission (see
[Cluster numerical bridges](#cluster-numerical-bridges)). Experimental `threshold` routing
is cluster-only and opt-in, as described below. A single node refuses a
`threshold` routing block at configuration load; use `fallback` there.

## Single-node embedding example

Start with another SIE deployment that already serves
`sentence-transformers/all-MiniLM-L6-v2`. Create `upstreams.yaml`:

```yaml
upstreams:
  team-sie:
    kind: sie
    base_url: https://team-sie.example.com
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

Usage counts come from the upstream. When an upstream reports no usage, a
native `encode`, `score` or `extract` request still succeeds, and the response
omits `usage` instead of estimating it. This holds for the `encode`, `score`
and `extract` operations each upstream kind supports, on a single node and in a
cluster. The OpenAI-compatible `/v1/rerank` route also omits `usage`. The
OpenAI-compatible `/v1/embeddings` route always returns a `usage` object, so it
reports a character-based estimate instead, which the gateway marks with
`sie_token_source: character_estimate`. For an OpenAI-compatible
upstream, a `usage` object with no count in it counts as no usage. Malformed
usage fails the request: a value of the wrong type, an invalid count, or an SIE
upstream's `usage` without `input_tokens`. Generation fails closed: a
generation, chat or completion response without exact final usage is an
error, never a success.

## OpenAI-compatible embeddings and rerank

Define a `kind: openai` upstream with a base URL that includes the provider's
API prefix, for example `https://host.example.com/v1`. Declare only endpoints
it offers: `endpoints: [embeddings, rerank]`. Use
`sie_server.adapters.remote.openai:OpenAIUpstreamAdapter` in the profile and
set `upstream_model` to the provider's model id.

OpenAI-compatible embedding profiles support dense text only. Sparse,
multivector, image input and extraction are rejected before dispatch. A rerank
upstream must accept the Cohere-shaped request. Without upstream usage,
`/v1/score` and `/v1/rerank` both omit `usage`, as described above.
SIE restores scores to document order rather than exposing the provider's
ranked order.

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
`SIEClient(..., remote="forbid")`. Other header values, and more than one
`X-SIE-Remote` field, are rejected. A remote-backed model has no local
alternative, so forbidding remote serving returns a client error.

On the gateway, `forbid` for a model with a `fallback` or `threshold` policy
selects only a fresh worker that advertises the versioned execution fence for
the current configuration. The request stays on that worker's updated-only
queue and backend IPC method; it never retries on an ordinary pool subject or
older backend method. Missing worker support returns `503` with
`Retry-After: 5`. Custom gateway dispatch transports refuse such a request
unless they explicitly implement this contract. A local model with no routing
policy has no remote route, so `forbid` leaves its ordinary dispatch unchanged
on every transport. Rolling back a worker or backend
therefore refuses verified work before inference. Explicit remote profiles are
refused with `400`. Model ids that cannot round-trip through the current queue
subject encoding (including literal `__` and `_dot_` collisions) are refused for
verified dispatch. Workers reject any verified subject/payload model mismatch
before readiness or payload retrieval.

Responses disclose `X-SIE-Served-By` and, when remote, `X-SIE-Upstream`.
Single-node fallback adds `X-SIE-Fallback-Reason`; a failed remote attempt
returns the original local refusal and `X-SIE-Fallback-Error`. The SDK exposes
these fields in response request metadata. `/v1/models` reports routing and
upstream kind.

`X-SIE-Fallback-Error` is the SIE error code the remote attempt answered with,
for example `RESOURCE_EXHAUSTED` from a capped or breaker-open generation
upstream, or `QUEUE_FULL` and `MODEL_LOADING` from an encode or extraction
upstream that cannot serve now. The OpenAI codes `server_overloaded` and
`invalid_request` become `QUEUE_FULL` and `INVALID_INPUT`. A remote attempt
refused with `PROVISIONING`, because the remote capacity is still starting,
received no answer, so it is `QUEUE_FULL`. Any other failure is
`INFERENCE_ERROR` for a server error and `INVALID_INPUT` for a client error.

Each upstream has a required rate cap and a circuit breaker. The limits are
shared by that worker process's adapters, not across replicas: adding remote
worker replicas increases the aggregate permitted traffic. Under
`remote_only`, unavailable upstreams, open breakers and reached caps return
retryable `503` responses with `Retry-After`. In a cluster the remote worker
answers such a request at once instead of redelivering it, and the gateway
returns the worker's code and wait. Under single-node `fallback`, an upstream failure
returns the original local refusal with its retry delay. A client error is
never retried remotely, and fallback cannot replay work after local acceptance
or after output reaches the caller.

TLS is required outside loopback. URLs containing credentials, a query or a
fragment are rejected. Redirects are refused, ambient proxy variables are
ignored, and inbound authorization is not forwarded. An explicit `proxy_url`
is the supported egress proxy setting.

An upstream's certificate is verified against the public CA bundle that the
server's HTTP client ships with (certifi). `SSL_CERT_FILE`, `SSL_CERT_DIR` and
other environment settings are ignored, and no setting adds a private CA
bundle. An upstream whose certificate is issued by a private CA is therefore
refused; give it a certificate from a public CA.

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
`profiles.<name>.identity`, a versioned digest or `null`. Version 2 conservatively
identifies revision-pinned BGE-M3 profiles on the native and the flash adapter,
and profiles on the BERT, BERT cross-encoder, Qwen2 cross-encoder and Nomic
flash adapters. Those four fall back to adapters that have no identity when
flash attention is unavailable, so they are identified only on a CUDA device
with compute capability 8.0 or newer and an installed flash-attn. It
includes model/tokenizer pins, resolved profile settings, engine configuration,
device/platform, serving Python sources, the exact builds of installed inference
libraries (each library's version and installed-file record, including
flash-attn, Triton and PEFT), current Torch precision/determinism settings and
the selected attention and BLAS backends. Hardware observations include the kernel,
CPU model/features and selected instruction capability, plus the observed CUDA
device properties and installed NVIDIA driver revision for CUDA execution.
Numerical library builds, the kernel and thread selection observed for
NumPy's own BLAS and environment settings also participate. Libraries that
other packages load later are not observed, so a server reports the same
identity before and after it loads a model. Apple Accelerate is bound to the
installed OS version/build. Otherwise NumPy's BLAS must be OpenBLAS or BLIS,
loaded from NumPy's own installation with observable kernel facts; wrapper
backends, system libraries and libraries without kernel facts report `null`,
including MKL.
Unknown hardware or numerical libraries report `null`;
configured device labels are insufficient. Alias names and
inheritance do not change a profile with identical resolved settings.

Local weight paths, mutable revisions, custom/checkpoint code, child engines,
unidentified precision and profiles that use a LoRA report `null`. A profile that
uses no LoRA keeps its identity when a sibling profile declares one only on the
flash BGE-M3 adapter, which applies LoRA per request and disables the adapter
layers for base requests; on other adapters any declared LoRA reports `null`.
SentenceTransformers and CrossEncoder require verified checkpoint module metadata:
disabling `trust_remote_code` alone does not identify installed checkpoint-selected
code. FlagEmbedding BGE-M3 cannot identify its effective revision. The other
flash adapters have inputs the identity does not bind: CUDA graphs that replay
or not depending on free memory and timing, side files fetched on a best-effort
basis, model code from the Hub that cannot be pinned, or LoRA support. Other
engines need additional runtime evidence. An identity is a descriptor, not a numerical
measurement. This field alone does not activate hybrid routing: OpenAI profiles
need passing numerical evidence; SIE profiles require the fresh comparison below.


## Measuring remote equivalence

An encode/score comparison runs through the Python SDK against one direct SIE worker
that has both the local default profile and an explicit remote profile. Keep
hybrid routing disabled while measuring. The server must expose a non-null
local identity, `profiles.default.runtime_instance_id`,
`profiles.<remote>.remote_contract_sha256` and
`profiles.<remote>.remote_execution_sha256`. The remote contract binds
its installed endpoint, model serving configuration, credential reference and request
transforms to the operator files supplied to the probe. The remote execution
digest identifies the serving code and inference libraries that run the remote
profile. Credential values are never included. Version 2 local identities
cover the adapters listed under [Local profile identity](#local-profile-identity).

From the locked public workspace, run:

```bash
mise exec -- uv run --frozen --project . python tools/remote_equivalence.py \
  --model-file /path/to/model.yaml \
  --upstreams-file /path/to/upstreams.yaml \
  --local-url https://sie.example.com \
  --api-key-env SIE_PROBE_API_KEY \
  --remote-profile remote \
  --output /path/to/new-evidence.json
```

The API key argument names an environment variable; omit it for a server that
does not require authentication. The output path must be new. Each case makes
four local calls with remote serving forbidden, followed by one explicit remote
call. Two local calls send the case alone, one sends it beside a longer item and
one sends it inside a batch of sixteen items, because the batch a worker forms
changes its numerical output. Cases cover short and long inputs, both sides of the pinned tokenizer's
truncation boundary including the local instruction prefix, query and document
instruction prefixes, default query instructions, empty prefixes, and score scale
when scoring is declared. All declared encode/score outputs must be measured.
Generation is outside this numerical probe. The routing policy is excluded from
the model digest, so evidence can be measured before enabling hybrid routing;
all local and remote profile settings remain bound. Version 3 records bind the
local profile's runtime options, float32 output and remote execution digest
explicitly. A bridged request runs with the remote profile's own runtime
options, so those options must equal the local profile's. A remote profile that
adds, omits or changes an option is refused by the probe and by admission, and
each measured profile receives its own options. Older records must be remeasured.

A pass requires matching layouts and finite values whose maximum absolute
error against every local observation does not exceed the local serving
envelope: the largest difference measured between any two local observations.
Identical local observations require identical remote values. There is no
tolerance setting. Each case records the batch composition of every local
observation; records that list none measured two single-request runs. Matching
boundary input refusals are recorded; a suite of refusals cannot establish
numerical equivalence. Endpoint/model contracts and local identity must remain
unchanged throughout the probe. Every local and remote observation, including
input refusals, must come from the same worker process identified by the initial
metadata. A load balancer mixing workers cannot produce admissible evidence.
The record names the local execution identity it measured, not the process, so
it also covers other processes that report the same identity.

In a cluster, add `--cluster` and pass the gateway URL as `--local-url`. The
probe then takes process provenance from cluster status, so it needs no
upstream credential and runs no model itself. The local processes are those of
non-remote workers that list the model in their numerical diagnostics, and they
must all report one identity and the supplied model contract. A worker that has
the model loaded but reports no diagnostics refuses the run. The remote
processes are those of remote workers that report the model's remote contract,
and they must agree with the supplied files. When the model runs on several
machine profiles with different identities, measure each one with `--gpu
<machine profile>`, which pins the local calls to that profile. The probe first
sends one local and one remote call that wait for capacity, so a cold lane
starts through ordinary demand before cluster status is read. Cluster status is
read again after the run, and any change in the processes, identities or
contracts refuses the result, including a lane that scales up or down during
the run.

Exit status is `0` for passing evidence, `1` for a measured failure, or `2` when
valid evidence could not be produced. Records contain input hashes, token
counts, serving identities, contract hashes and measured errors; they contain
no inputs, vectors or credential values. They carry a measurement timestamp so
an admission gate can reject stale or future evidence. Remeasure after changes
to weights, serving settings, software, endpoint or request transforms.
Writing a record does not activate hybrid routing; activation also requires
deployment-owned admission policy and a model routing update.

## Admitting SIE identity fallback

Single-node `fallback` for encode/score can use an SIE upstream when both sides
report the same immutable weights revision and non-null local execution identity.
The remote profile must name an explicit upstream profile, for example
`upstream_model: BAAI/bge-m3:default`, so the upstream's bare-model routing policy
cannot change where the request runs. Both deployments must use the same pinned
execution contract on the same identified adapter (see
[Local profile identity](#local-profile-identity)), including hardware, libraries
and resolved profile settings. Unknown identities remain refused.

Configuration load and each bridge compare bounded metadata obtained through
`SIEClient` with the deployment's configured credential, TLS and proxy policy.
A successful observation lasts at most 30 seconds. A read that completes
replaces it, including with a refusal; a read that fails leaves it to expire on
its own and holds off the next read for 2 seconds. The next check after expiry
refreshes metadata. A concurrent refresh
refuses another bridge instead of waiting or starting a second metadata request.
Changes to the installed upstream discard the previous observation.

Because configuration load runs this comparison, a single-node server whose
models directory holds a hybrid SIE-identity `encode` or `score` model depends
on the upstream at startup. When the upstream cannot be reached, or reports a
different weights revision or identity, the server refuses the model and does
not start. A hot reload of that model runs the same comparison, reusing a
matching observation up to 30 seconds old, and a refused reload is logged. A
refused reload of a loaded model leaves it unloaded. To start while the upstream is down, remove the model's `routing` block, then
add it back by hot reload once the upstream answers.

Metadata is uncompressed and limited to 64 KiB. Its pool/socket operations share
a 5-second deadline, including partial headers and chunk framing; OS hostname
resolution follows the platform resolver's timeout. Synchronous primitive
transports use the same socket deadline mechanism with their request budget.
Redirects, unavailable credentials and invalid responses cannot admit a bridge.
The remote-serving switch also blocks metadata dispatch.

Request transforms, non-float32 wire output and Muvera defaults are refused.
Runtime overrides outside the matched local defaults stay local. Expired or
changed metadata preserves the original local refusal and retry hint while
starting local warm-up. This admission path requires one concrete execution
device and does not enable gateway fallback or fleet-wide identity rollout.

## Admitting measured OpenAI fallback

For an OpenAI upstream, declare the absolute evidence path in its startup YAML:

```yaml
upstreams:
  vendor:
    kind: openai
    base_url: https://vendor.example/v1
    endpoints: [embeddings]
    rate_cap: {requests_per_minute: 600, max_concurrency: 8}
    equivalence:
      max_age_s: 3600
      record_files:
        BAAI/bge-m3: /absolute/path/bge-m3-equivalence.json
```

The map keys are local catalog model IDs. The referenced file holds one
version 3 record, or a bundle of records for several local identities (see
[Collecting numerical fleet evidence](#collecting-numerical-fleet-evidence)).
A passing record admits only the exact model, remote profile, endpoint/model
contract, remote execution digest and declared numerical outputs it measured,
and only for the local identity it names.
The age is a strict integer from 1 to 604800 seconds. Paths and proof authority
come only from startup upstream configuration; model API requests cannot supply
or install records. SIE upstreams refuse this policy and require their own
immutable identity comparison instead.

Boot with both profiles configured and no `routing` block, so the bare model
name is served locally. The valid policies are `remote_only`, `fallback` and
`threshold`; an absent block means local only. Run the probe against that
direct worker, writing to the declared evidence path, then add
`routing: {policy: fallback, fallback_profile: remote}` to the model YAML.
Model-config hot reload admits the change in the same process. All profile
settings must stay unchanged between measurement and activation. The local
profile must have an identity (see [Local profile identity](#local-profile-identity));
unidentified engines remain closed.

Every bridge rechecks the record and its age before loading or calling the remote
profile. Expired, missing, failed or mismatched evidence preserves the original
local refusal and retry hint while starting local warm-up. Valid local requests
with unmeasured runtime overrides or non-float32 output also remain local;
explicit profiles still serve as requested. Rejection of one exported model
retains its current local configuration without blocking unrelated updates.

A record is bound to the local execution identity it measured, not to the
measured process. A restarted server, or another server with the same software,
inference libraries, hardware and settings, reports the same identity and is
admitted by the same record, also when it starts from a hybrid YAML. A change
to any of them changes the identity and needs a new measurement. Admission
requires one configured concrete device (`cuda:0`, rather than `cuda`, for a
CUDA worker). Multiple devices or a model loaded outside that placement cannot
report an admission identity or use the proof. This does not enable gateway
fallback.

## Collecting numerical fleet evidence

Servers with different hardware, drivers, inference libraries or settings
report different local identities, and each identity needs its own
measurement: the local observations and one explicit remote run on a server
with that identity. The bundle tool collects one version 3 record per identity into one
evidence file, which an upstream's `record_files` entry can name in place of a
single record. It accepts repeated `--record` paths, writes a new `--output`
file, and checks a `--max-age-s` window from 1 to 604800 seconds (default 3600):

```bash
mise exec -- uv run --frozen --project . python tools/remote_fleet_equivalence.py \
  --record /absolute/path/identity-a.json \
  --record /absolute/path/identity-b.json \
  --output /absolute/path/bundle.json --max-age-s 3600
```

The version 2 bundle retains the original measurements, including misses.
Exit status 0 means all records passed and are fresh, 1 means valid evidence
contains a failure or expired/future measurement, and 2 means collection failed.
Existing output files are never overwritten. Inputs are bounded regular files;
the bundle contains at most 256 records and occupies at most 8 MiB.

Every record must measure the same model, profiles, endpoint/model contract,
remote execution digest, runtime defaults, outputs and probe inputs, and no
identity may appear twice. Canonical digests bind the entire record, including
its identity, timestamp and measured errors. A server is admitted only by a
passing, fresh record for its own identity, so a failed or expired record for
one identity leaves the other identities admitted.

Sidecars independently poll every adapter child and attach optional
`numerical_process_inventory` diagnostics to NATS health messages. Cluster
status exposes each worker's observation timestamp and all child statuses
(`observed`, `incomplete`, `unavailable`, or `invalid`), including saturated
workers. The metadata is limited to 256 children and 64 KiB; an oversized
inventory is omitted whole. Failed probes, legacy heartbeats and observations
older than ten seconds do not retain a previous process's inventory. Normal
health publication does not wait for these probes, and diagnostics use dedicated
IPC connections so they do not occupy serving or readiness connection slots.
A remote-lane process also reports, for each model with a hybrid `encode` or
`score` policy, the local identities its current evidence covers, as an
`admission` with an expiry and a digest, together with the remote profile's
contract and serving-code digests. Admitted remote attempts travel on a worker
subject of their own that only a sidecar with the admission check consumes.
Before such a process sends an admitted remote attempt upstream, it derives its
admission again and refuses the item unless the item names the current digest,
requests only admitted outputs and keeps the measured runtime options. A bridged
caller then receives its local refusal with
`X-SIE-Fallback-Error: INFERENCE_ERROR`. The check covers batches that start
after a change: a batch that has already passed it completes its upstream calls,
and an SIE upstream's identity comes from a cache trusted for up to 30 seconds.

An observation alone grants no routing authority, and an `observed` child can
still lack a local identity. The gateway combines these observations into the
admission decision described next.

## Cluster numerical bridges

In a cluster, a bare `encode` or `score` request for a model with a `fallback`
or `threshold` policy runs on the remote profile only under a current numerical
admission. `/v1/embeddings` follows the same rule, because it wraps `encode`.
The gateway validates such a request before it counts threshold demand,
publishes load-only work or makes a decision, so an invalid request gets its
`400` and nothing else. It decides when it is about to commit to the remote
attempt:

- The request must set no runtime option other than `is_query`, and every
  output it asks for must be listed in the admission. Any other option, such as
  `output_dtype`, keeps the request local, because the remote process would
  refuse it.
- A remote-lane worker qualifies when it is fresh, eligible and positively
  supports both execution authority and the numerical admission method,
  consumes the admission subject, carries the remote profile's exact
  configuration hash, and every adapter child reports
  the same admission for the model with the model contract it reports itself.
  That admission must expire at least five seconds later.
- Its admission must cover every local process that could serve the model.
  Each child of every worker on the model's local bundles and in the model's
  pool (`default` when it names none), starting and degraded workers included,
  must report an admitted identity and the admission's model contract.
  A worker without a complete inventory, a worker past the heartbeat timeout
  that has not been evicted, or a child without an identity or without the
  model keeps the request local. So does every request until the gateway has
  heard worker health for one heartbeat timeout after its health subscription
  starts or resumes, or after a longer silence. With no local worker the remote
  admission decides alone.
- The request is pinned to one admitted remote worker that still reports the
  capability and the same admission digest. Its items carry that digest on a
  subject that only a worker with the numerical admission check consumes, and
  the worker checks the admission again before calling the upstream.

A request that names a profile, including one in its body options, stays on its
selected route. Generation and extraction requests of a model with numerical
outputs never bridge in a cluster. If the remote worker refuses an admitted
attempt, for example because the admission changed after the gateway checked
it, a fallback route answers with its local refusal, and a threshold route
answers `503 INFERENCE_ERROR` with the worker's `Retry-After`. Each request's
decision is counted once on `sie.gateway.remote.numerical_admissions`.

Upgrade workers and gateways before applying a hybrid `encode` or `score`
configuration. A remote lane rolled back below the numerical admission subject
stops numerical bridging: its older sidecar never consumes admitted work, so
queued attempts time out instead of running unchecked, and a fenced sidecar
that later consumes them drops those past their deadline. Configuring a numerical
bridge also changes local behavior while no admission holds: a trigger that
would bridge a request commits to its local refusal, so a cold model answers
`MODEL_LOADING` instead of waiting for its load, and an opted-in `saturated` or
`unhealthy` trigger refuses instead of queueing. A request that could never
bridge, because it sets a runtime option other than `is_query`, keeps its
ordinary local path.


## Single-node generation fallback

A model with a local generation profile and a remote generation profile may
use `routing: {policy: fallback, fallback_profile: remote}`. Native generation,
chat, completions and supported buffered Responses share the bridge decision.
The request is validated before loading or remote dispatch. A local
`MODEL_LOADING` refusal starts local warm-up and makes one remote attempt;
`unhealthy` remains an explicit trigger. Remote serving forbidden by the
caller keeps the request local.

Streaming native generation and completions, like chat, read their first event
before committing HTTP success. If the bridge fails before output, the response
retains the original local refusal and retry delay and discloses the fallback
error. After the first event the stream reports failure without replaying work.
An explicitly named remote profile is served directly.


## Cluster fallback

A bare generation model with `routing: {policy: fallback, fallback_profile: remote}`
can bridge `provisioning` and `model_loading` refusals on native generation,
chat, completions and supported buffered Responses. Native generation, chat
and completions also support streaming bridges. Explicit profiles, bundle
pins, machine or pool overrides, and `X-SIE-Remote: forbid` retain their selected
route. Under a model access policy, a bare model goes to its remote profile
only when the policy admits that route and serves the remote profile, and
generation stays local while the policy governs generation routes. A request
that names the remote profile follows the same visibility and serving rules as
any other model. A caller who may not see the remote profile gets the bare model
from `/v1/models` without that profile and with no remote routing.

Cold capacity retains local pending demand. An available local worker whose
model is unloaded receives load-only work, and the gateway waits for broker
acceptance before attempting remote generation. If load acceptance fails, the
caller receives the local loading refusal. A loaded local model serves locally.

A dispatch transport that manages its own capacity can report a lane with no
ready capacity while the registry still lists workers for it. When a
`provisioning` bridge is admitted for such a lane, the gateway asks the
transport to wake the lane with load-only work on the lane's pool, and makes
the single remote attempt only after the transport accepted the wake. A wake
that is not accepted leaves the caller with the local `503 PROVISIONING`. When
the bridge is not admitted, the request takes the ordinary local path and no
wake is sent.

The remote attempt pins a fresh worker with the exact current configuration
hash and positive versioned execution capability. It cannot retry on the ordinary
pool subject. Its work item names the trigger in `fallback_reason`, so the remote
worker answers it at once whenever it cannot serve it now, rather than
redelivering it until the gateway's request timeout. Remote failure restores the
original local refusal body and `Retry-After`, with `X-SIE-Fallback-Reason` and
`X-SIE-Fallback-Error`; success discloses the remote profile's upstream. The
customer model name remains the requested model.

The gateway derives `X-SIE-Fallback-Error` from the remote worker's error code
as a single node derives it from its remote attempt, with one difference. When
a gateway deadline passes before the remote attempt answers anything (the
queued-result deadline, or a generation's first-chunk or overall deadline before
its first output), the gateway reports `QUEUE_FULL`: the remote lane never
answered. On a single node the same generation deadlines end the remote attempt
itself, with `first_chunk_timeout` or `overall_timeout`, and are reported as
`INFERENCE_ERROR`.

A streaming bridge waits for its first valid event before returning HTTP success.
A worker error, cancelled/failed terminal, transport failure or durability failure
before output restores the local refusal, and `X-SIE-Fallback-Error` carries the
worker's error code. Once an event is ready, subsequent errors remain in the
stream, with cleanup and no replay of inference. Explicit profiles keep ordinary
streaming behavior.

Extraction-only models can use the same cold/loading bridge on native extraction
and OpenAI audio transcription. JSON and MessagePack extraction are validated
before local demand or remote dispatch. Text plus metadata is bounded by
`SIE_MAX_ITEM_TEXT_BYTES` (2 MiB by default); set this existing worker setting
consistently on gateways and workers. Extraction labels, schema size/depth and
media field types are checked before admission. Metadata size reflects decoded
worker values, including MessagePack float widening; metadata extensions with
unknown decoded size keep local-only behavior. The selected model must accept
the input kind and support extraction.

Audio bridges retain the local refusal until the requested transcription format
is ready. A malformed upstream extraction result, or missing subtitle/timestamp
fields, restores that refusal before any response is returned. The compatibility
response preserves fallback reason/error and retry headers. Native extraction
preserves its existing successful partial-result contract.

### Opt-in spill before local acceptance

Set `routing.triggers` to include `saturated` or `unhealthy` to enable those
classes. Omitting `triggers` enables only `provisioning` and `model_loading`.
A usable worker takes precedence over degraded peers in the exact local bundle,
pool, machine profile and configuration hash. A worker still starting remains
a provisioning case. Confirmed unhealthy workers do not become a default-enabled
provisioning bridge.

Saturation checks include the worker saturation signal, cached pool queue
pressure, and an enforced lane in-flight ceiling. Shadow lane admission does
not cause remote spill. The pressure check accepts no local work and reserves
no capacity; the publisher still checks actual admission. A later publish
failure stays local and cannot trigger a remote replay. Each admitted spill
records local pending demand. A failed remote attempt restores the original
local refusal and retry interval; streaming failures after the first output
remain in the stream. Explicit selectors and `X-SIE-Remote: forbid` retain
their existing authority.

Numerical models bridge only as described in
[Cluster numerical bridges](#cluster-numerical-bridges). Threshold routing requires its separate deployment opt-in.

### Observing cluster fallback

The queue-routing dashboard shows gateway fallback response rates by model,
operation, reason and commitment outcome, plus observed remote-serving duration.
Enable the chart's existing alert rules to receive `SIERemoteFallbackPersistent`;
set `alertRules.remoteFallbackPersistenceSeconds` to the desired threshold
(default 600 seconds). The rule requires recent committed remote activity from
the same gateway replica. Successful local serving clears that replica's
observed duration. See [the telemetry contract](../../telemetry/README.md#remote-fallback-observations)
for bounded-label and replica semantics.


## Experimental cluster threshold routing

Threshold routing is available only in a cluster: the gateways count the
demand and make the decision. A server admits a `threshold` block only as a
queue worker behind a threshold-enabled gateway, which requires
`SIE_THRESHOLD_ROUTING_ENABLED=true` and the `SIE_IPC_SOCKET_PATH` that the
chart sets for queue workers. A single node runs without that socket path, so
it refuses the block at configuration load, and its own request path never
applies `threshold`. Use `fallback` on a single node.

Set `gateway.thresholdRouting.enabled: true` in the `sie-cluster` chart only
when opting into shared demand routing for generation or extraction. It is off
by default. The chart requires authenticated inference NATS, sie-config and
queue worker sidecars, and starts a separate ephemeral control broker that
accepts only gateway credentials. Workers retain their existing queue
connection and upstream secrets. The control broker has a gateway-only ingress
NetworkPolicy; the inference broker and its persisted work are unchanged.

A local model with an existing remote profile can use:

```yaml
routing:
  policy: threshold
  fallback_profile: remote
  wake_above: 2
  sleep_below: 0.5
  window_s: 10
  cooldown_s: 60
```

Rates count validated bare-model requests per second across all gateway
replicas. Once low demand is established, requests use the remote profile and
create no local demand or load-only work. Sustained high demand resumes local
routing: a cold local lane receives demand and load-only work while the remote
profile bridges provisioning and model loading. Once local capacity is ready,
requests stay local. Sustained low demand returns requests to remote; existing
worker idle eviction and autoscaling govern when the local lane sleeps.

Explicit profiles, bundle/pool/machine/engine selectors and `X-SIE-Remote:
forbid` retain caller authority. Managed deployment routes remain outside this
flag. A numerical `encode` or `score` model takes the remote route only under a
current numerical admission (see
[Cluster numerical bridges](#cluster-numerical-bridges)); otherwise its request
stays local. Invalid requests do not add
threshold demand. A remote-selected request still requires the current exact
worker execution contract; it cannot select a legacy worker or retry into
another backend after output has started. An unavailable remote lane returns
the ordinary remote provisioning/refusal response, without waking the local
lane or recursively bridging to itself.

A newly started, disconnected or configuration-skewed coordinator retains the
ordinary local warm-up/fallback behavior until it has fresh sustained evidence.
Windows have at least one-second resolution and a maximum of 86400 seconds.
The isolated broker is memory-backed and single-replica in the chart: a restart
rebuilds evidence conservatively. This is an experimental routing control,
without a measured throughput, cold-start or cost acceptance claim.
See the [coordinator contract](../sie_gateway/docs/threshold-routing.md).
