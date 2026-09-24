# Pre-registration

Everything here was settled on the dev records and committed before any test
record was sent in the tuned configuration. That covers the questions, every
backend's phrasing and decision rules, the page rule, what the page shows, how
its cards are chosen and the cascade's control. The code that applies each rule
is named next to it. The test slice is recorded once, against SIE server commit
`28a07f5`, which the manifest records.

## Inputs, frozen

`build_inputs.py` builds both case files. They were built once, on 2026-09-24
(UTC), and are frozen at these SHA-256 digests. The vulnerability file's
`questions` field was later regenerated from `questions.py`, before any
recording used it, and the records were left untouched. NVD records keep
changing after publication, so a later rebuild may differ; the published files
are these.

| File | SHA-256 |
|---|---|
| `inputs/vulnerability-triage/cases.json` | `54217dc3ea90863d49d719fd936c8506d3ef93b1eb8946829b5e7b2535358650` |
| `inputs/workflows/cases.json` | `165a41b3acaa3ca8878a9ab738396aca4095907ec430e8ce48d22e73cda84b2e` |

The vulnerability-triage records split by NVD publication date:

| Window | Records | Split |
|---|---|---|
| 2023-10-01 to 2023-10-31 | 20 per weakness class, 160 | dev |
| 2023-11-01 to 2023-11-15 | 8 per weakness class, 64 | dev |
| 2023-12-01 to 2023-12-31 | 20 per weakness class, 160 | test |

The October records were scored while the example was planned. That is why
they are in the dev pool and December is the test slice. No setting is chosen
on December.

The workflow set is the full 400-case test split of the typed-decisions
benchmark. It has no dev split, is not tuned, and every figure on it uses the
benchmark's own questions. Nothing from it goes on the page.

## The questions

A question belongs to a genre, not to a document: a question is kept or dropped
for every record alike, never for one record. Every backend is asked these
three about every record (`questions.py`):

| Question | Type | Gold, from the NVD record |
|---|---|---|
| `weakness` | pick one of 8 | the primary CWE, mapped to eight classes |
| `attack_vector` | pick one of 4 | CVSS v3.1 AV |
| `remote_unauthenticated` | yes or no | AV is network and PR is none |

GLiFormer is the exception: it is asked `weakness` and `attack_vector` only
(`questions.ASKED`). On dev, asked all three in one call, it confused
"exploitable remotely without a login" with "remotely over the network", and
its attack-vector answer collapsed to network.

These were asked on dev and dropped, for every record and backend, before this
freeze. A description rarely states them:
- user interaction, whether an account is needed, and the impact pattern.
  Each failed the page rule on dev, even for the Smart model.
- privileges required, and severity asked as a rubric. No model is asked a
  rubric question on this set.

**Severity is never asked.** `cvss.compose_severity` computes it in caller code
from a backend's three answers. It uses the CVSS v3.1 formula and tables in
`cvss.py`, fitted once on the dev records' NVD vectors:
- user interaction by weakness class;
- impact pattern by weakness class;
- privileges none when the record is exploitable remotely without a login,
  and low otherwise;
- attack complexity low.

It is scored on the exact level, never within one level. It is labelled
computed wherever it appears. It does not go on the page (see below).

## What is tuned on dev, and how (`tune.py`, `tuning.py`)

Every backend and question gets the same procedure.

1. **Phrasing.** Each question is recorded on dev in three phrasings:
   - `short`: option names only.
   - `described`: name and description.
   - `concrete`: the short names, with a concrete name where `questions.py`
     gives one:
     - memory corruption becomes "buffer overflow or out-of-bounds memory access";
     - the yes-or-no options become "anyone on the network can do it without an
       account" and "only a local user or a logged-in user can do it".

   For each phrasing, a decision rule is fitted (step 2) and judged by the page
   rule (step 3). A phrasing that passes is kept over one that does not, and
   among those, the one with the highest 5th-percentile balanced accuracy. When
   none passes, the one with the highest balanced accuracy is kept. Ties go to
   `described`, then `short`, then `concrete`. The kept mix is the `tuned`
   variant.

   Grouped GLiClass carries every question in one prompt, so its `described`
   phrasing (573 tokens) does not fit the 512-token window. It is recorded in
   `short` and `concrete` only.
2. **Decision rule.** The rule is refitted on a dev recording made in the tuned
   mix.
   - **Yes or no:** a cut-off on P(true), from 0.05 to 0.95 in steps of 0.01.
     Among cut-offs whose dev accuracy is at least the majority answer's, the
     one with the highest balanced accuracy wins. If none reaches that accuracy,
     the highest balanced accuracy wins. Ties go to the cut-off nearest 0.5.
   - **Pick one:** the plain top option, or the top option after dividing each
     option's probability by its mean over the dev records. The division uses
     no labels. It is kept only if all three of these hold:
     - it raises dev balanced accuracy;
     - accuracy stays at or above the majority answer's;
     - the worst recall among options with at least 10 dev records does not
       fall.
   - A backend that returns only its top option (GLiNER2 on pick-one questions)
     keeps it. The LLM returns a value, not a distribution, and gets no rule.
3. **The page rule, with margin.** A question passes for a backend or a lane
   when all of these hold on its dev answers, with its settings fixed:
   - In 1,000 bootstrap resamples (seed 20260924), the 5th percentile of
     balanced accuracy is at least 0.70.
   - The 5th percentile of accuracy minus the resample's majority-answer
     accuracy is at least 0.
   - Every option with at least 10 dev records is answered right at least 40%
     of the time.

   Balanced accuracy is the mean recall over the options with at least 10
   records. An option with one or two records would move it by a third or more
   on a single answer.

## What the page shows, fixed on dev (`page.py`)

**The board.** Three lanes, each one model on its own (`tune.LANES`):

| Lane | Backend | Model |
|---|---|---|
| Fast | `gliformer-large` | `knowledgator/gliformer-large-v1` |
| Smart | `gliclass-large-v1` | `knowledgator/gliclass-large-v1.0` |
| LLM | `qwen3-4b-instruct` | `Qwen/Qwen3-4B-Instruct-2507` |

**Which LLM.** Two Qwen models of the same size were recorded on dev with the
same call (the CVSS v3.1 base metrics and the weakness class as JSON), and
judged by the same page rule. `Qwen/Qwen3.5-4B` is named in the published SIE
Cloud prices, so it would carry a credit figure. It was to replace
`Qwen/Qwen3-4B-Instruct-2507` only if it passed the same questions. It did not:

| Model | `weakness` | `attack_vector` | `remote_unauthenticated` |
|---|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | fails: 0.674 | passes: 0.924 accuracy, 0.813 balanced, 5th pct 0.749 | passes: 0.817, 0.804, 5th pct 0.765 |
| `Qwen/Qwen3.5-4B` (non-speculative profile) | fails: 0.737, memory corruption 11 of 28 | passes: 0.924, 0.822, 5th pct 0.756 | fails: 0.688, 0.664, 5th pct 0.627; "needs local access or a login" right 35 of 104 |

The LLM row is `Qwen/Qwen3-4B-Instruct-2507` (`run.PAGE_LLM`). The prices do
not name it, so it gets no credit figure. `Qwen/Qwen3.5-4B` stays a dev-only
backend and is not recorded on test.

A lane shows a question only if it passed the page rule on dev. Any other
question is left out, and the foot counts it. On dev:

| Lane | `weakness` | `attack_vector` | `remote_unauthenticated` | severity, computed |
|---|---|---|---|---|
| Fast | left out: 0.728 balanced, 5th pct 0.685, cross-site scripting 9 of 28 | **shown**: 0.915 accuracy, 0.844 balanced, 5th pct 0.779 | not asked | not computed |
| Smart | **shown**: 0.879, 0.879, 5th pct 0.845 | **shown**: 0.924, 0.774, 5th pct 0.706 | **shown**: 0.821, 0.817, 5th pct 0.774 (cut-off 0.79) | left out: 0.638, 0.642, 5th pct 0.591 |
| LLM | left out: 0.674, file upload 8 of 28, memory corruption 7 of 28 | **shown**: 0.924, 0.813, 5th pct 0.749 | **shown**: 0.817, 0.804, 5th pct 0.765 | left out: 0.545, 0.525 |

**Why severity is left out.** No backend can pass it. Composed from the NVD
analyst's own answers to the three questions, the computed severity reaches
0.692 accuracy and 0.706 balanced accuracy on dev, with a 5th percentile of
0.657. The best lookup from those three answers to a level, fitted and scored
on dev, reaches only a 5th percentile of 0.718. The three questions a
description grounds do not carry enough of the vector.

**The figures row.** One row per question type, for Smart: how many answers it
got right, of how many, summed over its shown questions of that type.

**The cards.** There are three, in the order of `page.CARDS`:
1. A record anyone on the network can exploit without an account.
2. A record reachable over the network but only with an account.
3. A record exploitable only with local or physical access.

Each card is the first test record, in the order of
sha256("sie-typed-decisions-cards-v1:" + CVE id), that meets all of these:
- it meets the card's condition on its gold answers;
- its weakness class is not on an earlier card;
- every answer the card shows is right, from Fast's and Smart's shown
  questions;
- every answer the card shows is the model's own top option, so no decision
  rule changed it.

A card with no such record is dropped, and the page says so. A card cell holds
the returned answer and the probability the model returned for it, nothing
else. `page.py` refuses to emit a card that would show a miss.

**The catalog.** These backends, each with the questions it passed on dev:
- `gliclass-instruct-large`: weakness, attack vector.
- `laya`: weakness.
- `laya-typed-decisions`: weakness, and remote without a login.

**The totals line.** It gives the records shown on cards of the records
recorded, and each shown question's right-of-answered over the whole test
slice.

**Withdrawal on test.** The test slice gets the page rule on its point figures,
without the dev margin:
- balanced accuracy of at least 0.70;
- accuracy at or above the majority answer's;
- every option with at least 10 records right at least 40% of the time.

A shown question that fails any of these is withdrawn from every surface and
listed as withdrawn. Nothing else is removed after the test run.

**Time and cost.**
- Each lane's median time per record is labelled "a round trip, not a service
  level".
- A speed claim is made only when the slower median is at least 1.5 times the
  faster on test, comparing the same answers (`page.SPEED_MARGIN`). There are
  two comparisons:
  - Fast's one call against Smart's attack-vector request. On dev this is
    1.02 times, so no claim is expected.
  - Smart's three requests against the LLM's one call. On dev this is about
    19 times.
- Calls per record and every request are recorded. Cost in credits comes from
  a copy of the published SIE Cloud prices (`page.py --prices`). It applies
  only to a model the prices list, and only where its recorded responses carry
  the units it is billed in. No credit figure is given otherwise. On
  2026-09-24 the prices (version `2026-09-05-production-bootstrap-v4`) list
  none of the three board models.

## The cascade, and its control (`tune.py lanes`)

The cascade runs Fast first and hands two kinds of answer to Smart:
- the questions Fast is not asked;
- any answer whose top probability is below a per-question cut-off.

It is gated on probability, never on whether an answer looks right. The cut-off
for each question is the lowest (least escalated) of 0.50 to 0.95 in steps of
0.05, or 1.01 (escalate all), that meets all of these on dev:
- accuracy at or above Smart's alone;
- balanced accuracy at or above Smart's alone;
- no record that Smart answers right is answered wrong.

The cascade then passes its control only if it also costs less than Smart alone,
in calls per record.

On dev:

| Question | Cut-off | Escalated | Cascade | Smart alone |
|---|---|---|---|---|
| weakness | 0.95 | 73.7% | 0.888 accuracy, 0.888 balanced | 0.879, 0.879 |
| attack vector | 1.01 | 100% | 0.924, 0.774 | 0.924, 0.774 |
| remote without a login | not asked of Fast | 100% | 0.821, 0.817 | 0.821, 0.817 |

It matches or beats Smart on every question, with zero regressions. But it
makes 3.74 calls per record against Smart's 3.00, so **it fails its control**.
It does not go on the page. The README reports it on test, under the same
control.

No LLM rung is added. The cascade fails before one would matter, and the LLM
does not pass weakness.

## Figures computed on the test slice only (`score.py`, `page.py`)

- Accuracy, balanced accuracy, macro-F1, Brier and ECE per backend and
  question, in the tuned configuration, with the majority answer beside them.
  This covers the computed severity, which is reported but not shown.
- Selective accuracy for pick-one questions. At each confidence cut-off in
  0.5, 0.7, 0.8, 0.9 and 0.95, this is the number of records whose top option's
  probability, after the rule, is at least the cut-off, and how many of those
  are right.
- The page surfaces above, and any withdrawal.
- The cascade against Smart alone.

## The GLiFormer multi-task run (`multitask.py`)

- **Records.** The first 12 test records, in the order of
  sha256("sie-typed-decisions-multitask-v1:" + CVE id), whose NVD entry lists
  exactly one vulnerable CPE vendor and product, both written in the
  description after lowercasing and keeping only letters and digits.
- **Combined call.** One call per record, at threshold 0.1, carries:
  - entity labels vendor, product and version;
  - relation labels "made by" and "affects version";
  - an output schema with `affected_component` and root enums for weakness and
    attack vector.
- **Separate calls.** The same tasks, repeated as three calls.
- **Found.** An extracted product span counts as found when, after lowercasing
  and keeping only letters and digits, it equals the NVD CPE product, or one
  contains the other and the shorter has at least 4 characters.
- **Pass rule.**
  - Product found for at least 10 of 12 records.
  - The combined and separate calls agree on the weakness enum for at least 11
    of 12.
- **Reported, not claimed.** Vendor (matched the same way) and the
  attack-vector enum. These were settled on 12 dev records before the test run:
  - Threshold 0.1 found the most products (12 of 12, against 10 at 0.3).
  - The combined call found the vendor for only 3 of 12 at any threshold or
    label wording, because it folds the vendor into the product span.
  - The attack-vector enum came back for 1 of 12.
- **Nearest neighbours.** Every test record is embedded with the same model.
  The report is how often a record's nearest other record by cosine similarity
  has the same weakness class, against chance.

## Revisions before any test record was sent

This file was revised on dev only. None of these changes looked at a test
record:
- **Questions.** Five questions (two of them rubrics) became three, and severity
  is now computed rather than asked. The case file's `questions` field was
  regenerated to match, which changed its digest.
- **Page rule.** It gained the bootstrap margin and a class floor of 40%, up
  from a quarter. Balanced accuracy now counts only options with at least 10
  records.
- **Phrasing choice.** It now prefers a phrasing that passes the page rule.
- **Prior correction.** It may no longer lower the worst recall.
- **Yes-or-no cut-offs.** They run from 0.05 to 0.95.
- **Page design.** GLiFormer is asked two questions. The Fast lane is
  GLiFormer alone, and the cascade is judged against Smart alone. The LLM row,
  the card rule, the withdrawal rule and the speed margin were added.
- **LLM row.** `Qwen/Qwen3.5-4B` was checked on dev and kept out, because it
  fails the remote-without-login question that `Qwen/Qwen3-4B-Instruct-2507`
  passes.
