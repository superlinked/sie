# Suggested SIE examples gallery entry

This is a prepared entry for `examples/README.md`, not an applied repository
change. Copy this self-contained directory to `examples/watermark-robustness/`
when preparing a contribution.

| Example | Best for | SIE primitives | Setup | Status |
|---|---|---|---|---|
| [Measure text-watermark robustness](./watermark-robustness) | Understanding how translation and paraphrase change a known watermark signal | `chat_completions`, optional `extract` | SIE generation endpoint; standalone Python 3.12/uv project; saved data and offline tests | Educational evaluation example; new SDK path tested offline |

Keep `EXPERIMENT.md` as the long-form explanation and `README.md` as the
runnable entry point. Retain both datasets and their separate provenance.
Do not label the updated SDK runner as live-verified until a current run is
saved and inspected. No key, model weights, virtual environment or
publication archive should be committed.
