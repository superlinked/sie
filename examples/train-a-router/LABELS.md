# Where the router’s labels come from

This example labels **model outcomes**, not whether a question looks “easy” or “hard.” For every training question, Luna and Sol each produce an answer independently. A deterministic grader checks each answer against the BBEH reference. The resulting labels are:

| Luna correct? | Sol correct? | What the router learns |
|---|---|---|
| Yes | Yes | The cheaper answer can be enough |
| No | Yes | Paying for Sol can recover accuracy |
| Yes | No | The stronger model is not always the better choice |
| No | No | Paying more does not necessarily solve this question |

Both correctness labels are retained. The model also learns expected costs from actual token usage. Neither router sees a reference, an answer, a correctness label, or the test cost when processing a new question.

## If you have reference answers

Use them to check real outputs from the two models you intend to serve. This is the validated route in this example. It avoids assuming that a teacher can predict which model will solve a question without seeing their answers.

BBEH provides reference answers. The supplied recordings were collected with `gpt-6-luna` and `gpt-6.1-sol`, medium reasoning, a 16,384-output-token limit and the same question-only JSON answer contract. A format or quality failure is retained as a failure; it is not regenerated until it looks correct. Transport attempts are tracked separately, including unresolved cost reservations.

The grader normalizes source-validated equivalent answer formats and preserves the raw upstream score too. For example, a unique option letter plus its matching text can be equivalent to the letter alone. It does not use the expected answer to repair a model’s response. The primary and source-format sensitivity results are both supplied.

## If you do not have reference answers

A teacher model can **judge the answers produced by both candidate models**. A useful protocol is:

1. Collect representative questions and both candidate answers. Keep related questions together when splitting the data.
2. Give a teacher the question, any authoritative context and each anonymized answer. Use a concrete correctness rubric and allow “uncertain.” Randomize answer order if asking for a comparison.
3. Retain its two judgments, confidence/uncertainty and explanation. Avoid reducing every pair to a forced winner: both answers can be right or wrong.
4. Check a separate sample with people, executable tests or authoritative references. Measure teacher errors and disagreements before treating its labels as truth.
5. Train using the accepted labels; choose settings on development data and policies on calibration data. Keep an independently checked test set for the money–quality comparison.

A frontier API model or a model served through SIE can be the teacher. Its usefulness depends on the task and the quality of its judgments, not only parameter count. If Sol itself judges Sol versus Luna, correlated errors and self-preference can distort the result; an independent judge and an audited sample are preferable.

**Teacher-only labeling is not a measured result of this BBEH example.** We have not established that Qwen 27B, GLM Flash or another frontier model produces equivalent labels here. The current results should not be presented as evidence of that. Teacher labeling is a way to extend the workflow, with its own validation cost.

## Collect fresh Sol/Luna answers

The published recordings are enough for the complete training exercise. The following optional commands make a new paid collection with the same questions and settings. They keep the old recordings intact. Run them from this example folder.

```bash
# Enter your OpenAI key and inspect the training/development collection plan without making calls.
export OPENAI_API_KEY='your-openai-api-key'
uv run --frozen --group collection python router.py collect \
  --data data --phase learn --output fresh-answers/learn --budget-usd 50
```

```bash
# Make the planned training/development calls, retaining physical attempts and their costs.
uv run --frozen --group collection python router.py collect \
  --data data --phase learn --output fresh-answers/learn --budget-usd 50 --execute
```

```bash
# Fetch the exact public benchmark source and grade the new answers against its references.
uv run --frozen python -X utf8 source/fetch.py --output bbeh-source
uv run --frozen --group collection python -X utf8 grade_v2.py \
  --prepared data/source --collection fresh-answers/learn \
  --official bbeh-source/bbeh/evaluate.py --pair pair.json --output fresh-grades/learn
```

The budget is a ceiling, not the expected bill. The collector reserves room for in-flight requests and unresolved executions, and stops rather than silently exceeding it. An identical plan can resume in the same output directory: completed responses are reused and unresolved attempts retain their reservations. Inspect the receipts first. A different plan cannot overwrite that run.

Use `--phase calibration` for a separate calibration collection. Final collection additionally requires `--frozen PATH`, pointing to a policy bundle made before those final answers. When repeating the published experiment, `data/reference/frozen-frontiers.json` is the supplied unchanged bundle:

```bash
# Plan a fresh final-answer repeat for the already frozen reference policies; add --execute to dispatch it.
uv run --frozen --group collection python router.py collect \
  --data data --phase final --output fresh-answers/final --budget-usd 25 \
  --frozen data/reference/frozen-frontiers.json
```

These commands create new answer and grade directories; they do not replace the downloaded evidence or silently change a prepared training run. For example, train a separate CPU embedding router directly from the newly graded answers:

```bash
# Prepare a new CPU study and fit neighbors/classifiers using the fresh training and development labels.
uv run --frozen python -X utf8 embedding/recipes.py prepare --source data/source --pair pair.json --run fresh-cpu-router
uv run --frozen --group outcomes python -X utf8 embedding/recipes.py fit \
  --run fresh-cpu-router --features data/features \
  --train-grades fresh-grades/learn/grades-train.json \
  --development-grades fresh-grades/learn/grades-development.json
```

This lower-level study is separate from the README’s published-data reproduction. Direct scripts use `python -X utf8` so multilingual inputs and JSON files use UTF-8 regardless of the host locale; `router.py` sets this mode automatically, including for its child processes. `native/bbeh_native.py` also accepts explicit grade paths; its prepared plan must bind the new acquisition protocol. Preserve source, answer-pair, grading and acquisition bindings together; changing models or prompts is a new experiment.

For production feedback, collect outcomes on a representative sample of **both** routes. Labeling only the arm chosen by the current router hides the alternative’s performance and can reinforce its mistakes. New labels can update the neighbor collection, refit the small embedding heads or support another LoRA adaptation. Re-evaluate before replacing a serving policy.
