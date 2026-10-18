# Model crash recovery probe

This bounded experiment starts its own loopback SIE server, pins
`Qwen/Qwen3Guard-Gen-4B` with `AppStateConfig.pinned_models`, and loads
`ibm-granite/granite-guardian-3.0-2b` beside it. It sends a single SIGKILL to
Granite's exact owned engine child, lets the ordinary registry monitor detect
the exit, and requests one reload after the normal cooldown. All inference
calls use `sie_sdk.SIEClient`.

The default protocol records 25 healthy-model requests before the crash, up to
50 during detection/cooldown/reload, and 25 after it, with a two-second pause
between requests. This is a low-rate recovery experiment, not a throughput
benchmark. The same healthy process must remain loaded and answer every
request. During the crash cooldown the victim returns `502 MODEL_LOAD_FAILED`
without `Retry-After`, which the SDK does not retry. The probe records that
response and waits for the observed registry cooldown before making a fresh
request. Reload then returns `503 MODEL_LOADING` with `Retry-After` before a
successful answer from a new process. A second failure aborts the experiment.

Run only on an isolated H100 with at least 75 GiB GPU memory, 8 CPU cores and
64 GiB host RAM. The public checkpoints in the checked-in model configurations
must be available. The environment must support the SGLang bundle; the probe
records exact installed package versions and source hashes. The experiment's
1300-second deadline leaves time for server shutdown inside a 1500-second
external machine/container lifetime. Stop the machine/container after collecting
the evidence, including after a failure.

From a clean public checkout with its locked workspace installed, first run
the CPU smoke (no model download or GPU is used):

```sh
mise exec -- uv run --frozen --no-sync python tools/benchmarks/model_recovery/probe.py \
  --smoke --phase-requests 3 --during-requests 20 --interval-s 0.1 \
  --deadline-s 60 --recovery-timeout-s 20 --output tmp/recovery-smoke
```

The smoke uses real AppFactory startup/shutdown, registry state transitions,
HTTP/SDK calls, child processes, and the same measurement/persistence code.
Only the adapters, weight cache, and cooldown duration are synthetic. It uses
the native generate route rather than real model chat templates. Its timings
are never model-performance or GPU-recovery evidence.

Then run in the isolated SGLang environment:

```sh
python tools/benchmarks/model_recovery/probe.py --output /tmp/recovery-evidence
```

The output directory must not exist; attempts never overwrite prior evidence.
`events.jsonl` is flushed after each event, including HTTP status/Retry-After,
PID and residency snapshots, cooldown state, one crash, recovery, and cleanup.
`summary.json` records nearest-rank p50/p95 latency, errors and request counts
per phase, with the actual sampling window. Recovery longer than the bounded
during-phase sampling window is not measured continuously. Warmups and victim
recovery probes are excluded from healthy-model
latency summaries. Failure and cleanup events remain available when a protocol
assertion fails.

This does not test CUDA OOM recovery, multiple simultaneous crashes, repeated
failure thresholds, high-load fairness, hardware isolation, another serving
engine, or throughput equivalence. Pinning is configured, but its causal effect
against eviction requires a separate controlled memory-pressure experiment.
The script intentionally reads private registry/process fields to verify
identities; it is a revision-specific benchmark, not a stable SDK example.
