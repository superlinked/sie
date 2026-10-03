# Second pass on doc field extraction: pre-registration

Written and committed **before any call in this experiment was made**.
Registered 2026-09-22.

## The question

The published run is 180 of 223 fields exact across 8 documents, one call each
(`../manifest.json`, `../evaluation.json`). 43 fields are wrong. Does a
second call fix a meaningful share of them without breaking the 180 that are
already right?

## What the 43 misses are, read off the published evaluation before any new call

| class | fields | where |
|---|---|---|
| Right text, extra printed token kept (`"Chromium (Cr)"` for `element`, `"SRM 1155a"` for `srm_number`, a name with its address attached, `"KÖLN 50823"` for `destination_city`) | 20 | NIST 15, Wolters Kluwer 2, USPS 2, FAA 1 |
| Values landed in the wrong field: meter current/previous swapped, four charge rows shifted by one | 10 | DEI electricity bill |
| Invented values on a page photographed sideways (another store's address and phone, 7.47 for a 191.13 total) | 7 | Walmart receipt |
| Character-level misreads and two wrong enum choices | 5 | OSHA 300 log |
| A tick read on the wrong box (`credit_card_type_checked` `"visa"` for `"none"`) | 1 | Wolters Kluwer |

The largest class, 20 of 43, is not a reading failure. The model read the
printed text correctly and kept a neighbouring token the schema wanted
elsewhere or did not want at all.

Three of those 20 fields already carry a schema `description` saying exactly
that (`organization_name`: "Block 4 organization name, without the address";
`sold_to`: "Organization name only"; `remit_to`: "Payee name for checks"). The
model attached the address anyway. So "add a field description" is a claim the
published run already tests on three fields and does not support, which is why
it is an arm below rather than an assumption.

## Arms

All three arms are scored against the **unchanged** pre-registered `expected`
values in `../inputs.json`, with the **unchanged** comparison rules from
`../evaluate.py`, over all 223 fields. Stage one is the published run; its
responses are not re-recorded.

- **A1, one call, described schema.** A single call, same image, same prompt,
  same model. The only change is a `description` added to every property that
  lacks one. This is the control the composition has to beat: if a better
  schema gets there, no second model call is justified. The descriptions in
  `experiment.json` were written **after** reading stage one's 43 misses, and
  they are deliberately generous: several of them name the exact distinction
  the model got wrong. That bias is in A1's favour, which is the right
  direction for a control a composition must beat, and it is why an A1 win
  would be reported as "the schema was underspecified" rather than as a
  property of the model.
- **A2, second call, no image.** Stage two receives the original schema and
  stage one's JSON as text, and returns a corrected object.
  `/v1/chat/completions` with `response_format: json_schema`, because
  `/v1/generate` skips the chat template for a plain text prompt on Qwen models,
  so a text-only second call there would be answered by a model that never saw
  its instructions as a message. An image input does go through the template,
  which is why A1 and A3 stay on `/v1/generate`.
- **A3, second call, with the image.** Stage two receives the page image, the
  original schema and stage one's JSON, and returns a corrected object.
  `/v1/generate`, the same path stage one uses.

`max_new_tokens` / `max_completion_tokens` is 2048 in every arm, the value the
published proof calls use. No sampling parameters are set in any arm, as in the
published run.

## Decision rule

Per arm, over all 223 fields:

- **total** = fields exact.
- **fixes** = wrong at stage one, right in this arm.
- **regressions** = right at stage one, wrong in this arm.

**Publish an arm only if `total >= 195` and `regressions <= 5`.**

195 is stage one's 180 plus 15: a third of the misses closed. Five is the most
regressions worth trading, because a repair pass that breaks correct fields is
the failure mode `composition-brief.md` records from `/guardrails`.

If more than one arm clears, publish the one with the higher total; on a tie,
the one with fewer regressions; on a further tie, the simpler arm, A1 before A2
before A3.

**If no arm clears, nothing about a second pass is published.** The page keeps
its single recorded call and the result is reported as a negative one.

A1 clearing is not a composition. It would be a stage-one improvement, and the
page would say that in those words.

Whatever arm wins, the seam is published: which of the 43 it does not fix, by
class.

---

*One wording change after the run, recorded rather than made silently: the A2
bullet originally cited a private issue tracker by name for the `/v1/generate`
chat-template behaviour. The public repository's tree policy forbids that
reference, so the bullet now states the behaviour itself. No arm, prompt, schema,
threshold or tie-break was touched, and `score_second_pass.py` rebuilds every
request from `experiment.json` rather than from this file, so the arms are
checkable independently of any edit here.*
