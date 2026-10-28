# Translate one product string with Hy-MT2 1.8B

One English string from Android, translated into Japanese with one call to
`tencent/Hy-MT2-1.8B` on SIE. It is the call behind
[superlinked.com/translation](https://superlinked.com/translation), whose
sources are in its
[SOURCES.md](https://superlinked.com/reference/translation/SOURCES.md).

The string is Android's notification-permission message:

> Starting in Android 13, apps that you install need your permission to send
> notifications. Tap to change this permission for existing apps.

Recorded through SIE Serverless (`https://api.superlinked.com`) on 2026-10-25,
Hy-MT2 returned:

> Android 13以降では、インストールしたアプリが通知を送信するにはあなたの許可が必要です。既存のアプリのこの許可を変更するにはタップしてください。

It finished with `stop` after 50 input and 51 output tokens. The version
number comes through as "Android 13以降"; Android's own Japanese writes
"Android 13 以降". Where Android's Japanese says 権限, its word for an app
permission, Hy-MT2 writes 許可, the everyday word for permission.

## The call

```text
POST https://api.superlinked.com/v1/generate/tencent__Hy-MT2-1.8B
{"prompt": "<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following text into Japanese. Note that you should only output the translated result without any additional explanation: Starting in Android 13, apps that you install need your permission to send notifications. Tap to change this permission for existing apps.<｜hy_Assistant｜>",
 "max_new_tokens": 256,
 "temperature": 0}
```

With the Python SDK, as `run.py` sends it:

```python
from sie_sdk import SIEClient

text = (
    "Starting in Android 13, apps that you install need your permission to send notifications. "
    "Tap to change this permission for existing apps."
)
prompt = (
    "<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following text into Japanese. "
    "Note that you should only output the translated result without any additional explanation: "
    f"{text}<｜hy_Assistant｜>"
)

# SIE_API_KEY for Serverless, or SIE_BASE_URL for your own server
with SIEClient() as client:
    result = client.generate(
        "tencent/Hy-MT2-1.8B",
        prompt,
        max_new_tokens=256,
        temperature=0,
        provision_timeout_s=300,  # wait for capacity
        read_timeout_s=600,  # Hy-MT2's per-request limit
    )
print(result["text"])
```

`/v1/generate` takes a raw prompt and applies no chat template, so the prompt
carries Hy-MT2's own turn markers around the English instruction from its
[model card](https://huggingface.co/tencent/Hy-MT2-1.8B/blob/9a341cd1b679d3efd23b46e847b01745a71ed792/README.md).
Name the target language in English. The card puts a blank line between the
instruction and the text; the recorded call used one space, and `run.py`
builds the prompt the same way. The call sets temperature 0 because the model
profile defaults to 0.7.

## Run it

From this directory:

```sh
uv sync --frozen
uv run --frozen python run.py --show
SIE_API_KEY=sk-sie-... uv run --frozen python run.py
```

`--show` prints the request and sends nothing, so it needs no key. The live run
sends one generate call and prints the translation, its finish reason, its
token counts and whether the text equals the recorded one:

```json
{
  "model": "tencent/Hy-MT2-1.8B",
  "text": "Android 13以降では、インストールしたアプリが通知を送信するにはあなたの許可が必要です。既存のアプリのこの許可を変更するにはタップしてください。",
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 51
  },
  "matches_recorded": true,
  "recorded_on": "2026-10-25"
}
```

That is the output of a run on 2026-10-28. Your text can differ: SIE's output
is not deterministic even at temperature 0. In the study below, pairs of
identical requests got different outputs 21% to 25% of the time. A `false` in
`matches_recorded` is not an error.

To translate something else, pass `--text` and `--into`, the target language
named in English:

```sh
uv run --frozen python run.py --text "Your order has shipped." --into German
```

The comparison with the recording is then left out. `run.py` sizes
`max_new_tokens` from the input the way the page does: 5 tokens per 3
characters plus 16, rounded up to 64, 128, 256, 512 or 1024. The recorded
string gets 256, and text over about 300 characters gets the model's maximum
of 1024. If a translation still reaches its cap, the finish reason is `length`
and the text is cut off: `run.py` prints the output, warns on stderr and exits
with status 1. Split longer text into parts.

The first call after the model has been idle can wait a few minutes while
capacity starts. The SDK waits through the responses that come before
generation starts (`429`, and `503` while capacity starts or the model loads),
for up to five minutes here. It does not send the call again after a gateway
timeout or a dropped connection, when the generation may already have run.

The full response also carries request ids, timings and billing fields; the
example prints none of them. On a failure it prints only the error type and the
HTTP status, because SDK error text can carry request context. The key is read
from `SIE_API_KEY` and never printed.

### On your own server

Hy-MT2 runs on the GPU generation image:

```sh
docker run --gpus all -p 8080:8080 \
  -v sie-hf-cache:/app/.cache/huggingface \
  ghcr.io/superlinked/sie-server:latest-cuda12-sglang
SIE_BASE_URL=http://localhost:8080 uv run --frozen python run.py
```

SIE's [model config](../../packages/sie_server/models/tencent__Hy-MT2-1.8B.yaml)
pins Hugging Face revision `9a341cd1b679d3efd23b46e847b01745a71ed792` and
serves it with SGLang in bfloat16. SIE Serverless also serves
`google/madlad400-3b-mt`, which takes the source after a target tag such as
`<2ja> ` instead of an instruction; this example does not call it.

## How well it translates

The page's figures come from a WMT24++ study. Its sample,
every published output, both score files, the scorer and a script that
recomputes the scores are public:
[superlinked/sie-task-evidence @c4add1e0, translation/wmt24pp-study-2026-10](https://huggingface.co/datasets/superlinked/sie-task-evidence/tree/c4add1e09ebffc5f0d0909b88b9671597de0bebc/translation/wmt24pp-study-2026-10).

## Test without a key

```sh
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

The tests run the real SDK against an HTTPX `MockTransport`. They check that it
sends one POST to `/v1/generate/tencent__Hy-MT2-1.8B` whose JSON equals the
recorded body, sizes the output cap from the input as the page does, sends the
key as a bearer token, prints no error detail, flags a cut-off translation and
does not resend after a gateway timeout. They compare the parsed JSON, not the
bytes: the SDK escapes non-ASCII characters, and the recorded call was sent as
UTF-8. No model runs and nothing leaves the machine.

## Source

The English string is `review_notification_settings_text` from
[`core/res/res/values/strings.xml`](https://android.googlesource.com/platform/frameworks/base/+/refs/tags/android-15.0.0_r1/core/res/res/values/strings.xml)
in the Android Open Source Project's `frameworks/base`, tag
`android-15.0.0_r1`, under the Apache License 2.0. [recorded.json](recorded.json)
holds the string, the recorded request and the response fields this example
prints.
