Run `uv run fetch.py` from the example folder to download the six inputs here.
The files and verified manifest are ignored locally; this README is the only
input-directory file kept in Git.

The [public workshop dataset](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/487cdc3bd0480659afd7ee88329449e6a6c8655d/cognee/workshop-starter/2026-11-06/v1)
is pinned to revision `487cdc3bd0480659afd7ee88329449e6a6c8655d`, prefix
`cognee/workshop-starter/2026-11-06/v1`. The fetch checks the pinned manifest's
SHA-256 and every input's length and SHA-256. It works anonymously and reuses
verified local copies.

These are hand-authored synthetic workshop records, revision 1 (2026-11-06),
released under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
They contain no real people, venue records or customer data.

`readiness-policy.txt`, `projector-check.txt` and `microphone-check.txt` seed
the initial memory. `access-note.png` is a rendering of `access-note.txt`;
the text is authored input, not an OCR response. `correction.txt` teaches
experiment 04 and is deliberately absent from the initial memory.

`make_document.py` renders an optional local variant under `runs/` without
replacing the downloaded PNG. Its bytes can vary with Pillow versions.
