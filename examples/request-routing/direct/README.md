# Select a handler with one native call

Supply a request and the handlers your application supports. SIE returns a
score for each handler. This example selects the unique highest score in caller
code and prints it beside the complete native response; it never executes the
selected action.

The sample asks to remove a calendar event. Its eight supplied handlers cover
calendar, meeting-room, reminder and to-do workflows. The original text is
CLINC150 test row 3982, `calendar_update`, from revision
`828f8093932c8fe6ca7936c3d2e52903b1c523de`, published by Larson et al. under
[CC BY 3.0](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE).

## Run one request

From this directory:

```sh
uv sync --frozen
SIE_BASE_URL=http://localhost:8000 uv run --frozen python run.py
```

The endpoint must expose `fastino/GLiNER2.5-Decide`. To use a hosted endpoint,
set `SIE_BASE_URL` and `SIE_API_KEY` for that deployment. The call can spend its
API credits. It makes at most one inference send and blocks automatic SDK
retries.

You can run the same native contract with the instruction-tuned GLiClass model:

```sh
uv run --frozen python run.py --model knowledgator/gliclass-instruct-large-v1.0
```

Both public default profiles use single-label classification. The request
preserves the full source text and every allowed handler; `overflow_policy=error`
rejects a request that exceeds the model context. Native scores are not a
calibrated probability of choosing the right business action. Your application
owns authorization, ambiguity handling and dispatch.

## Test without inference

```sh
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen ruff check run.py tests
uv run --frozen ruff format --check run.py tests
```

The tests exercise native request serialization and reject incomplete, non-finite
or tied scores. Their responses are explicit offline fixtures, not recorded
model results.
