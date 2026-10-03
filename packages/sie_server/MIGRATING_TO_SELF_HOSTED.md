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
in SDK request metadata and record the outputs and usage. This is your own
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

Automated identity/equivalence admission is still being built under
[issue #415](https://github.com/superlinked/sie/issues/415). Current SIE refuses
hybrid embedding and score routing. Do not enable a fallback policy to bypass
that refusal.

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

An extraction-only model can use the current single-node fallback after its
local and remote profiles support the same declared outputs. A bridged request
starts local warm-up. Cluster fallback, generation fallback and threshold
routing remain unfinished, so they are not steps in this migration.

Record performance only from actual runs, including refusals and failures.
No latency or cost improvement is assumed by this guide.
