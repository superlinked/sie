# Moving from a hosted model to self-hosted serving

Put the hosted model behind SIE first, then change where SIE serves it. The
application keeps its SIE base URL, SDK and model name. Repeat for one model
at a time.

Use the [remote backend guide](REMOTE_BACKENDS.md) to configure an upstream
and a remote-backed profile. A provider's model id belongs in `upstream_model`;
`sie_id` is the stable name your application uses. This guide describes the
current implementation, not a released-version guarantee.

## 1. Establish a remote-only baseline

Declare the actual model dimensions, supported operations and input limit.
For an SIE upstream, keep the upstream's model config as evidence. For an
OpenAI-compatible upstream, use its documented model facts and verify the
returned vector dimensions and usage.

Run representative application requests through SIE, including long inputs,
instructions, refusals and your normal concurrency. Confirm remote serving
in SDK request metadata and record the outputs and usage. When the upstream
reports no usage, SIE's native `encode`, `score` and `extract` responses omit
it rather than estimate it (an OpenAI-compatible upstream serves no `extract`),
so take those figures from the upstream's own records. This is your own
migration baseline; it does not establish equivalence between backends.

## 2. Prepare the local model separately

For the embedding example in the remote guide, the local configuration is
already in the catalog:
[`sentence-transformers__all-MiniLM-L6-v2.yaml`](models/sentence-transformers__all-MiniLM-L6-v2.yaml).
Use the same weights revision and the intended pooling, normalization,
truncation and instruction behavior. Load it on a separate test deployment
before changing the serving deployment's default profile.

Compare two local runs to measure their noise, then compare the remote
baseline against them. Check short and long inputs, the truncation boundary
and instruction prefixes. For reranking, compare score scale as well as
ordering. For extraction, compare schemas and extracted results. A shared
model name or matching vector dimension alone does not prove equivalence.

Single-node hybrid embedding and score routing has narrowly scoped admission:
revision-pinned native BGE-M3 profiles can use a fresh SIE identity comparison
or a measured OpenAI equivalence record bound to the same worker process.
Follow the [admission instructions](REMOTE_BACKENDS.md#admitting-sie-identity-fallback)
before enabling fallback. With an SIE upstream the comparison also runs at
startup, so that single node does not start while the upstream is unreachable
or reports a different identity. Other local engines, including the MiniLM example
above, remain outside that admission; compare their outputs for a deliberate
remote-only to local-only cutover. Numerical cluster bridges remain gated under
[issue #415](https://github.com/superlinked/sie/issues/415).

## 3. Switch the model's default profile

After validating the local deployment, replace the remote-backed model config
with the local catalog config while keeping the same `sie_id`. Remove
`remote_backed: true` and `routing: {policy: remote_only}`; use the catalog's
weight source and local default adapter. Do not leave duplicate configs for
one model id in the models directory.

On a single node, restart with the chosen local bundle and device. In a
cluster, use the normal config-service and worker-pool rollout. Verify model
readiness and make a request with `remote="forbid"` to confirm local serving:

```python
from sie_sdk import SIEClient

with SIEClient(base_url="http://localhost:8080", remote="forbid") as client:
    result = client.encode(
        "sentence-transformers/all-MiniLM-L6-v2", [{"text": "Find related documents"}]
    )[0]
    assert result["request"].get("served_by") == "local"
```

If embedding outputs are not equivalent, rebuild stored document vectors
with the local backend before querying that index with local vectors. Keep
separate indexes during the cutover so one search never mixes incompatible
vectors. For reranking, recalibrate score thresholds if the scale changes.

## 4. Verify and retain rollback configuration

Check the served-by metadata, application quality, usage and resource demand
after the cutover. Keep the prior remote config as an operator-controlled
rollback artifact outside the active models directory. Restoring it returns
the model to remote-only serving; it does not repair an index built with
incompatible vectors.

Single-node extraction and generation models can use fallback when their local
and remote profiles satisfy the [operation-specific contracts](REMOTE_BACKENDS.md#supported-operations-and-policies).
A bridged request starts local warm-up. Cluster generation and extraction
fallback use the normal queue path and require workers that support the
versioned execution fence. Saturation and unhealthy-worker spill require
explicit triggers. [Experimental cluster threshold routing](REMOTE_BACKENDS.md#experimental-cluster-threshold-routing)
requires a separate deployment opt-in and is available only in a cluster; a
single node uses `fallback`. None of these modes makes unmeasured embedding
outputs interchangeable.

Record performance only from actual runs, including refusals and failures.
No latency or cost improvement is assumed by this guide.
