# Answer several triage questions about one record, each with a probability

One record goes in with a few typed questions: pick one of N options, or a
yes-or-no check. Every answer comes back with a probability a program can
branch on. This example measures how well models served by SIE answer those
questions about public CVE records, against the NVD analysts' own assessments.
It covers zero-shot label classifiers, typed decision models and a small LLM.

Everything that decides what is reported was fixed on 224 earlier records and
committed before the 160 test records were sent, in
[PREREGISTRATION.md](PREREGISTRATION.md). The test slice was then recorded
once, against SIE server commit `28a07f5`.

## What this shows

Three questions are asked of every CVE description:

| Question | Type | Gold, from the NVD record |
|---|---|---|
| `weakness` | pick one of 8 classes | the primary CWE, mapped to eight classes |
| `attack_vector` | pick one of 4 | CVSS v3.1 attack vector |
| `remote_unauthenticated` | yes or no: "anyone who can reach the system over the network can exploit this without logging in" | attack vector network and privileges none |

### The results on the 160 December 2023 test records

Each lane is one model. A cell is filled only where the model passed the
pre-registered page rule on the dev records, with a bootstrap margin (see
PREREGISTRATION.md). Right answers are out of 160; balanced accuracy is in
brackets.

| Lane | Model | Vendor | `weakness` | `attack_vector` | `remote_unauthenticated` | Median per record |
|---|---|---|---|---|---|---|
| Fast | `knowledgator/gliformer-large-v1` | Knowledgator | not shown (failed on dev) | 144 (0.873) | not asked | 69 ms, 1 call |
| Smart | `knowledgator/gliclass-large-v1.0` | Knowledgator | 142 (0.887) | 147 (0.796) | 115 (0.725) | 205 ms, 3 calls |
| LLM | `Qwen/Qwen3-4B-Instruct-2507` | Qwen | not shown (failed on dev) | 150 (0.867) | 129 (0.815) | 2,671 ms, 1 call |

- **By question type:** Smart answered 289 of 320 pick-one questions and 115
  of 160 yes-or-no questions.
- **Medians are round trips, not a service level.** Each server ran on one
  NVIDIA L4 GPU, with the client on the same machine, one record at a time.
- **One speed claim clears its pre-registered 1.5x margin.** Smart answers the
  same three questions 13.0 times faster than the LLM. Fast's one call and
  Smart's attack-vector request take about the same time (1.02x), so no speed
  claim is made between them.
- **The LLM gives no probabilities.** It returns values, so its answers carry
  none.
- **No credit figures.** `page.py --prices` prices only models the published
  SIE Cloud price list names. On 2026-09-24 it named none of these three.

GLiFormer is not asked the yes-or-no question. Asked all three in one call, it
confused "exploitable remotely without a login" with "remotely over the
network".

**More models, each on the questions it passed on dev:**

| Backend | Model | Vendor | Right of 160 |
|---|---|---|---|
| `gliclass-instruct-large` | `knowledgator/gliclass-instruct-large-v1.0` | Knowledgator | weakness 142; attack vector 149 |
| `laya` | `convaiinnovations/laya` | Convai Innovations | weakness 134 |
| `laya-typed-decisions` | `convaiinnovations/laya-typed-decisions` | Convai Innovations | weakness 145 |

`laya-typed-decisions` also passed the yes-or-no question on dev. On test its
balanced accuracy there fell to 0.673, below 0.70, so the pre-registered
withdrawal rule took it off every surface.

**Three records illustrate the answers.** They were chosen by the stated rule
in `page.py`, which picks one record for each argument:

| Record | Argument |
|---|---|
| CVE-2023-49378 | anyone on the network can exploit it without an account |
| CVE-2023-6826 | reachable over the network, but only with an account |
| CVE-2023-44278 | local access only |

Every answer shown on them is right, and each is the model's own top option.

### Also recorded, and not on the page

**A cascade.** Fast answers first. Smart answers what Fast is not asked, and
any answer whose top probability is below a cut-off fixed on dev. The cascade
passes its pre-registered control only if it meets all three of these against
Smart alone:
- it matches or beats Smart;
- it answers wrong no record that Smart answers right;
- it costs less.

On test:

| Question | Escalated to Smart | Cascade | Smart alone |
|---|---|---|---|
| weakness | 67.5% | 0.919 accuracy | 0.887 |
| attack vector | 100% | 0.919, 0.796 balanced | 0.919, 0.796 |
| remote without a login | 100% (Fast is not asked) | 0.719, 0.725 balanced | 0.719, 0.725 |

It matches Smart with zero regressions and beats it on weakness. But it makes
3.68 calls per record against Smart's 3.00, so it fails its control, as it did
on dev.

**Severity, computed and never asked.** `cvss.compose_severity` scores every
combination of a backend's three answers with the CVSS v3.1 base-score formula,
weighted by their probabilities. The metrics a description rarely states come
from tables fitted on the dev records' NVD vectors, in `cvss.py`. The formula
reproduces NVD's base score for all 384 records. Scored on the exact level:

| Severity from | Accuracy | Balanced accuracy |
|---|---|---|
| the NVD analyst's own three answers (the ceiling) | 0.719 | 0.743 |
| Smart | 0.600 | 0.601 |
| the LLM, from its whole CVSS vector | 0.675 | 0.613 |
| always the most common level | 0.463 | |

The three questions a description grounds do not carry enough of the vector,
so no backend passes. GLiNER2 returns only its top option on pick-one
questions, so no severity can be composed for it.

**A GLiFormer multi-task run.** On 12 pre-registered test records,
`multitask.py` asks one GLiFormer call for entities, relations and a typed
record together, and scores them against NVD's CPE vendor and product. This
recorded run did not meet its pre-registered bar. The product was found in 9
of 12 records, against a bar of 10. The combined and separate calls agreed on
the weakness class in 11 of 12, against a bar of 11. Nothing from it is
claimed.

**The workflow set.** This is the 400-case test split of the public
[typed-decisions benchmark](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
with five questions per case. Its gold is a teacher model's label, and the
benchmark's own card puts that teacher's self-agreement at 0.735.
`convaiinnovations/laya-typed-decisions` was fine-tuned on that benchmark's
training split, so it is a specialist scored against the teacher labels of its
own distribution. Every other backend is zero-shot.

| Backend | Accuracy over 2,000 decisions | Macro-F1 |
|---|---|---|
| `laya-typed-decisions` (fine-tuned on the benchmark's training split) | 0.769 | 0.663 |
| always the most common answer | 0.522 | 0.230 |
| `gliner2-large` (Fastino), the best zero-shot backend | 0.516 | 0.345 |
| `laya` (Convai Innovations), the base checkpoint | 0.360 | 0.239 |

### Every backend recorded

| Backend | Model | Vendor | How the questions are sent | Calls per record |
|---|---|---|---|---|
| `laya` | `convaiinnovations/laya` | Convai Innovations | the question dict as `output_schema` | 1 |
| `laya-typed-decisions` | `convaiinnovations/laya-typed-decisions`, fine-tuned on the workflow benchmark's training split | Convai Innovations | the question dict as `output_schema` | 1 |
| `gliformer-large` | `knowledgator/gliformer-large-v1` | Knowledgator | every question a label group, named by the question; on vulnerability triage it is asked `weakness` and `attack_vector` only | 1 |
| `gliclass-instruct-large-grouped` | `knowledgator/gliclass-instruct-large-v1.0` | Knowledgator | every question a label group, the questions in the instruction | 1 |
| `gliclass-instruct-large` | `knowledgator/gliclass-instruct-large-v1.0` | Knowledgator | one question per call, the question as the instruction | one per question |
| `gliclass` | `knowledgator/gliclass-large-v3.0` | Knowledgator | one question per call, labels only | one per question |
| `gliclass-large-v1`, `gliclass-base-v1`, `gliclass-small-v1` | `knowledgator/gliclass-{large,base,small}-v1.0` | Knowledgator | one question per call, labels only | one per question |
| `gliner2-base`, `gliner2-large` | `fastino/gliner2-{base,large}-v1` | Fastino | one question per call, the question id as the task name | one per question |
| `nli-modernbert-base` | `MoritzLaurer/ModernBERT-base-zeroshot-v2.0` | MoritzLaurer | one question per call, each label as an NLI hypothesis | one per question |
| `qwen3-4b-instruct` | `Qwen/Qwen3-4B-Instruct-2507` | Qwen | one chat completion with a JSON schema, returning the CVSS v3.1 base metrics and the weakness class (vulnerability triage only) | 1 |
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B`, through its non-speculative profile, which keeps the JSON schema enforced | Qwen | the same chat completion; recorded on dev only | 1 |

Every model is Apache-2.0. "One call" means one request per record. It says
nothing about how many encoder passes the server runs inside it.

A label classifier takes a text and a list of labels, so each label has to
carry its meaning (`questions.py` holds every rendering). A yes-or-no question
becomes two labels, and the workflow set's rubric questions become one label
per level. GLiNER2 in single-label mode returns only its top label, so it has
no Brier score for pick-one questions. GLiFormer scores each label
independently, so a missing label counts as zero, and each question's scores
are divided by their sum. That makes Brier and ECE computable, but not
calibrated.

`laya-package` and `laya-typed-decisions-package` run the same checkpoints
through the reference `laya==0.3.11` package at pinned revisions (`aa8c91ca`
and `bc76315b`). They are not in the plan; they check that SIE answers the way
the reference implementation does.

## Run it

The code is in this repository. The inputs and the recorded responses are in
the public Hugging Face dataset
[`superlinked/sie-task-evidence`](https://huggingface.co/datasets/superlinked/sie-task-evidence),
under `typed-decisions/`, so a clone alone is not enough. Fetch, then check and
score:

```sh
python3 fetch.py                   # downloads the pinned revision into data/
python3 run.py --check             # rebuilds all 17,856 recorded requests from the inputs
python3 score.py                   # every backend and question on the test slice; asserts the published figures
python3 page.py                    # the page's lanes, cards, catalog and cascade; asserts what the page shows
python3 multitask.py --score --calls data/multitask/test.json
python3 -m unittest discover -s tests -v
```

These need nothing installed: they are standard library only, and none of them
needs an API key, a Hugging Face token or any inference spend. `fetch.py`
pins a commit SHA, not `main`, and checks every downloaded file against a
digest. It replaces `--dest` wholesale, so it refuses to touch anything
without the `.sie-evidence` marker it writes.

`python3 score.py --split dev` reports the dev records the settings were
chosen on. `python3 tune.py phrasing|rules|lanes --calls data/calls.json`
re-derives `tuning.json` from them. `run.py --show vulnerability-triage
CVE-2023-49378 gliclass-large-v1 tuned` prints the requests one record sends,
without sending them.

### Record it yourself

This needs an SIE server that serves these models. The LLMs need a server
built with SGLang (the `sie-server:latest-cuda12-sglang` image). The steps are
the ones PREREGISTRATION.md fixes: every phrasing on dev, then the tuned
configuration on dev, then the test slice once.

```sh
uv sync --group build
uv run python build_inputs.py                  # rebuilds data/inputs/<set>/cases.json from the NVD and Hugging Face

BACKENDS="laya laya-typed-decisions gliformer-large gliclass-instruct-large-grouped gliclass-instruct-large
          gliclass gliclass-large-v1 gliclass-base-v1 gliclass-small-v1 gliner2-base gliner2-large nli-modernbert-base"
URL=http://localhost:8080

# 1. Dev, every phrasing, then choose the phrasing per question
for b in $BACKENDS; do
  uv run python run.py --record --set vulnerability-triage --split dev --variant short described concrete \
    --backend $b --url $URL --out run-output/dev--$b.json
done
for b in qwen3-4b-instruct qwen3.5-4b; do
  uv run python run.py --record --set vulnerability-triage --split dev --backend $b \
    --url $URL --out run-output/dev--$b.json
done
python3 tune.py phrasing --calls run-output/dev--*.json

# 2. Dev in the chosen phrasing, then fit the decision rules, the lanes and the cascade
for b in $BACKENDS; do
  uv run python run.py --record --set vulnerability-triage --split dev --variant tuned \
    --backend $b --url $URL --out run-output/dev-tuned--$b.json
done
python3 tune.py rules --calls run-output/dev-tuned--*.json
python3 tune.py lanes --calls run-output/dev-tuned--*.json run-output/dev--qwen3-4b-instruct.json

# 3. Test, once, in the tuned configuration; and the workflow set
for b in $BACKENDS qwen3-4b-instruct; do
  uv run python run.py --record --set vulnerability-triage --split test --backend $b --url $URL
done
for b in $BACKENDS; do
  uv run python run.py --record --set workflows --backend $b --url $URL
done
uv run python multitask.py --record --split test --url $URL --out run-output/multitask/test.json

python3 run.py --merge run-output/*--*.json --server-commit <sie commit>   # calls.json and manifest.json
```

Rebuilt inputs can differ from the published ones, because NVD records keep
changing after publication. Set `SIE_BASE_URL` (or pass `--url`) to point the
backends at another server, and `SIE_API_KEY` when that server needs one.
`--limit N` records the first N cases for a quick look and marks the recording
incomplete.

To compare SIE's Laya answers with the reference package, `uv sync --group laya`
and record `laya-package` and `laya-typed-decisions-package` the same way.
`score.py` reads both through the same normaliser. On a GPU the server runs
Laya in reduced precision, so the two agree on almost every top answer rather
than on every one, and their probabilities differ in the third decimal place.

## What to expect

`run.py --check` prints `17856 of 17856 recorded calls rebuilt from the inputs
and matched`. It fails if a call is missing, recorded twice, implied by no
case, or sent something other than what the inputs and `tuning.json` produce.

`score.py` prints, per set, the majority-option reference and then one block
per backend and question type:
- accuracy, macro-F1, Brier and ECE;
- for rubric questions, the mean absolute error of the expected level;
- latency per record;
- a per-question table.

It ends with `All 24 published figures reproduced.` and exits nonzero if any
figure in `PUBLISHED` stops reproducing.

`page.py` prints the lanes, the figures row, the cards, the catalog, the speed
claims and the cascade:

```
== page surfaces, test slice, 160 records
  fast  knowledgator/gliformer-large-v1           attack_vector 144/160 (bal 0.873)
  llm   Qwen/Qwen3-4B-Instruct-2507               attack_vector 150/160 (bal 0.867), remote_unauthenticated 129/160 (bal 0.815)
  smart knowledgator/gliclass-large-v1.0          attack_vector 147/160 (bal 0.796), remote_unauthenticated 115/160 (bal 0.725), weakness 142/160 (bal 0.887)
  figures  choice: 289 of 320 over 2 question(s)
  figures  noul: 115 of 160 over 1 question(s)
  card remote-no-login: CVE-2023-49378 (request_forgery)
  card network-needs-login: CVE-2023-6826 (file_upload)
  card local-access: CVE-2023-44278 (path_traversal)
  ...
  WITHDRAWN {'catalog': 'laya-typed-decisions', 'question': 'remote_unauthenticated', 'reasons': ['balanced accuracy 0.673']}

The page's cards, time medians, speed claims and withdrawals reproduced.
```

## What is in the dataset

```
typed-decisions/
  inputs/vulnerability-triage/cases.json  384 NVD records: description, gold, CVSS vector, CPE, source digest
  inputs/workflows/cases.json             400 typed-decisions benchmark cases with their questions and gold
  calls.json                              17,856 calls: request digest, response, status, timing
  summary.json                            score.py's figures for every backend, set and split
  page.json                               page.py's surfaces on the test slice
  multitask/test.json                     the GLiFormer multi-task run and its 160 embeddings
  manifest.json                           server commit, hardware, models, plan, sources and terms, digests
```

`calls.json` keeps every response as it was recorded. Each request body is
stored as its SHA-256 (`body_sha256`), because the body is fully determined by
the inputs and `tuning.json`. `run.py --check` rebuilds every body and compares
digests, and `run.py --show` prints any of them. Dev calls are included, in
every phrasing, so the tuning can be re-derived.

## The two sets

### `vulnerability-triage`

CVE records from the [NVD CVE API 2.0](https://nvd.nist.gov/developers/vulnerabilities).
The model reads the English description and nothing else. The gold answers are
the NVD analyst's own:
- the primary CWE, mapped to eight weakness classes in `build_inputs.py`;
- the CVSS v3.1 attack vector;
- whether the record is exploitable over the network without privileges;
- the base severity.

The selection rule is in `build_inputs.py` and was fixed before any model call.
Records are taken from three publication windows: October 2023 and 1 to 15
November 2023 (224 dev records), and December 2023 (160 test records), 48 per
weakness class in all. Each record must meet all of these:
- it is not rejected;
- it has exactly one NVD primary CVSS v3.1 metric;
- it has exactly one NVD CWE in the mapped set;
- it has an ASCII English description of 12 to 150 words.

Records are ordered by a salted SHA-256 of the CVE id, and the first N per
class in each window are taken. No record was chosen or dropped by hand.

This product uses data from the NVD API but is not endorsed or certified by the
NVD. CVE descriptions are used under the
[CVE Program Terms of Use](https://www.cve.org/Legal/TermsOfUse): Copyright (c)
1999-2026, The MITRE Corporation. CVE is a trademark and the CVE logo is a
registered trademark of The MITRE Corporation. `manifest.json` carries the
license text in full.

### `workflows`

The `test` split of the [typed-decisions benchmark](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
at commit `c76749ec58bd8c3d2ea706b31c333a9059c38f90`, used under Apache-2.0.
The states are synthetic: a model wrote the text around sampled scenario
factors. `cases.json` reformats 100 rows per workflow, in salted-hash order,
and changes no text. It has no dev split and is not tuned.

## What this does NOT establish

- **Nothing about served latency.** A median is a round trip on one machine,
  one record at a time, not a service level.
- **Nothing about correctness on the workflow set.** Its gold is a teacher
  model's answer to synthetic states, and the one strong score there comes
  from a checkpoint fine-tuned on that benchmark.
- **Nothing about the CVSS fields a description does not state.** NVD analysts
  also read vendor advisories and code. That is why severity is computed from
  three grounded answers and fixed tables, and why it still falls short.
- **Nothing about other phrasings or questions.** Each question got three
  phrasings, and each backend kept the one it did best with on dev. A
  different wording, or a question dropped on dev, could move any row.
- **No calibration claim.** A decision rule is chosen for accuracy on dev, not
  to make the probabilities calibrated.
- **Nothing about rare attack vectors.** The test slice has 130 network
  records, 27 local, 2 adjacent and 1 physical. Balanced accuracy counts only
  options with at least 10 records, so it covers network and local.
- **Nothing about the end of a long record, for GLiClass.** GLiClass requests
  ask SIE to cut the record text, never the labels, when both do not fit in 512
  tokens.
- **Nothing about calibration for `laya-multilingual`.** It is defined in
  `run.py` but not in the plan, and it ships no temperature table.
