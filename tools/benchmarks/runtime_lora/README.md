# Runtime LoRA acceptance probe

This standalone probe tests SGLang runtime LoRA through `sie_sdk.SIEClient` on
one CUDA device. It runs pinned Qwen3-0.6B generation and Qwen3-Embedding-0.6B
embedding engines sequentially, each with `mem_fraction_static: 0.5`.
It requires the runtime LoRA implementation and SGLang `0.5.10.post1`.

Both engines receive an explicit `--max-total-tokens 4096` KV pool upper bound. The
fraction remains the production admission/static-memory setting; it does not
imply a 4096-token cap when explicitly overridden. Generation uses its supported
`extra_launch_args` option. Embedding has no such option, so a benchmark-only
subclass appends the same fixed flag to its memory arguments while preserving
the production launch and inference paths. CPU tests capture both real launch
commands and check that only the KV cap differs from the uncapped profile.

The local fixtures are **untrained correctness fixtures**, not useful task
adapters. Fixed-seed, nonzero rank-4 `q_proj`/`v_proj` weights make incorrect
selection observable without depending on a mutable Hub adapter. No quality
or performance conclusions follow from their outputs.

From a public checkout with its dependencies installed:

```sh
mise run test -- tools/benchmarks/runtime_lora
mise exec -- uv run --no-sync python -m tools.benchmarks.runtime_lora.fixtures \
  --output /tmp/runtime-lora-fixtures --download-weights
mise exec -- uv run --no-sync python -m tools.benchmarks.runtime_lora.probe \
  --fixtures /tmp/runtime-lora-fixtures --output /tmp/runtime-lora-evidence
```

For an installed CUDA bundle, run the same Python modules in that bundle's
environment; do not overlay an incompatible dependency environment. The CPU
tests verify fixture loading with PEFT, base restoration, SDK retry observation,
configuration, and observer behavior. **CPU tests are not GPU acceptance.**

The acceptance run checks:

- Actual `503 LORA_LOADING` followed by another successful SDK request, with no
  direct engine calls substituted for native generation, chat, or embeddings.
  Encode retries automatically. The current generation SDK surfaces this
  refusal, so the probe records and makes bounded fresh SDK calls only for
  pre-execution `503 LORA_LOADING`; it never retries a failed open stream.
- Adapter effects in native/chat token log probabilities and embedding vectors;
  base restoration after switching and after eviction. Both engines record a
  first response and two warm baseline repeats before any runtime selection.
  For generation, restored warm text must match exactly. At every matching token
  context, chosen and shared top-token probability drift must stay below the fixed `1e-3` bound.
  Top-k support changes are recorded: absent tokens have probability bounded by
  the other top-k cutoff, and conservative difference lower bounds also must stay
  below `1e-3`. Unreported tail probabilities remain unmeasured; small lower bounds
  do not establish upper bounds on their drift. First-token
  pairwise selection effects must exceed `1e-4` and five times baseline/reset
  first-token drift. Embeddings compare raw SDK vectors without extra
  normalization: warm-repeat and reset maximum absolute error must stay below
  `5e-5`, while all pairwise selection effects must exceed `1e-5` and five times
  the larger warm-repeat/reset error. Non-finite vectors and shape changes fail.
  All full text, vectors, and token probabilities are retained, including cold/warm
  differences. Native streaming also probes one token after a fixed common
  prefix where an earlier run diverged; this holds context constant when
  comparing base/A/B/reset. These fixtures establish selection, not quality.
- Stable engine PID, rejection of undeclared adapters, ready-name listing,
  preservation of one pinned adapter, and churn across four dynamic adapters.
- Overlapping A/B streams **and a successful engine forward containing both**.
  A benchmark-only subclass prepends an observer to the generation child's
  startup hooks, preserving SIE's normal compatibility hooks. The observer
  records request IDs, adapter UUIDs and aliases, and actual GPU pool slot
  assignments after `Scheduler.run_batch`. The A/B request UUIDs must occupy
  distinct valid slots, with consistent inverse mappings. Completed SDK streams
  establish completion; the hook timestamp is not a GPU timing measurement.
  It also requires SIE's slot-release compatibility hook to be present.
  Cache-prefix/input/output lengths are captured before the forward, using
  scalar fields and tensor shape metadata only. Observation does not alter
  scheduling, cache settings, or model tensors.

`evidence.json` contains the final verdict. `events.jsonl`, `batches.jsonl`,
fixture hashes, generated model configurations, and `source-manifest.json`
provide the supporting record. Use a fresh output directory. The run has a
780-second default deadline and unloads each owned engine before starting the
next; any external compute allocation must have its own finite lifetime and
be stopped and verified separately by its owner.
Numeric/restoration assertion failures are recorded while independent phases
continue on a healthy engine; the final verdict remains failed and the process
exits unsuccessfully. Transport, observer, or lifecycle failures stop the run.

Pinned primary sources: [generation config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json),
[embedding config](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/config.json),
[SGLang scheduler](https://github.com/sgl-project/sglang/blob/v0.5.10.post1/python/sglang/srt/managers/scheduler.py),
and [LoRA GPU pool](https://github.com/sgl-project/sglang/blob/v0.5.10.post1/python/sglang/srt/lora/mem_pool.py).
