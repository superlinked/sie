# End-to-end ATT&CK mapping experiment

## Development result

The agent starts with complete report text. It extracted a behavior overlapping
92 of 128 known report-technique pairs across 16 AnnoCTR development reports.
The candidate ensemble kept the correct technique family for 84 of those 92
pairs.

The 27B verifier and nearest labeled training example agreed on 50 suggestions
that overlap an AnnoCTR annotation. Forty-four used the exact reference ID, for
88.0% selective precision. Those agreements become direct suggestions. Other
27B-supported mappings stay in the analyst review queue with the same source
quote and candidate ledger.

AnnoCTR has no annotation beside 48 other direct suggestions. The evaluator
does not assume those suggestions are correct. At the report-pair level, 29 of
54 direct suggestions match the published reference. That 53.7% reference
match rate remains in the run artifact.

Every one of the 351 extracted behaviors retains exact global character
offsets. The worked Proofpoint report is absent from the aggregate.

## Why the output changed

The first development hypothesis treated every annotated noun occurrence as a
separate target. It reached 4.7% implicit-mention recall and 12.5% precision.
That unit rewarded reproducing repeated words such as `ransomware` instead of
preparing a report-level review queue.

The next version emitted every mapping accepted by the 27B verifier. It found
more of the reference, then sent 201 unique report-technique pairs to review.
Only 41 matched an annotated span and ID. The verifier was acting as a mapper,
with almost no abstention.

The final development revision adds 1,915 labeled ATT&CK spans from the
AnnoCTR training split as an independent retrieval path. Dense definitions,
token-level MaxSim, and labeled examples form one candidate pool. A direct
suggestion requires agreement between the nearest labeled example and the 27B
verifier. Disagreements remain visible for analyst review.

## Frozen held-out result

The fixed pipeline ran once on the 33-report AnnoCTR test split on August 21,
2026. It saw 317 exact report-technique pairs after the frozen exclusions.

| Release check | Frozen result | Gate | Outcome |
|---|---:|---:|---|
| Direct-suggestion precision on annotated spans | 91/109 (83.5%) | 85% | Missed |
| Behavior extraction recall over report-technique pairs | 199/317 (62.8%) | 70% | Missed |
| Family finalist recall after behavior extraction | 180/199 (90.5%) | 90% | Passed |
| Exact source offsets | Passed | Required | Passed |
| Worked report absent from the aggregate | Passed | Required | Passed |

The aggregate did not clear its release gate. The held-out precision result was
4.5 points below development, while behavior recall fell 9.1 points. Candidate
retrieval held: once extraction found the behavior, the correct ATT&CK family
reached the finalist set in 90.5% of report-technique pairs.

No prompt, threshold, rank weight, candidate count, or routing rule changed
after this result. The worked Proofpoint report remains a separate case study;
it is not evidence that the aggregate passed.

## Dataset boundary

- Source: AnnoCTR commit `d510b6949e1938d47c93a43eedd562dc538439dc`.
- Labeled retrieval examples come from the published train split.
- Development uses the published dev split.
- Final measurement used the published test split once on August 21, 2026.
- `proofpoint_2022-02-03_mfa-psa-oh-my` is excluded from the aggregate because
  it informed the example design. It remains the worked report.

The aggregate uses the ATT&CK catalog bundled with AnnoCTR. The worked report
uses ATT&CK Enterprise 19.2. AnnoCTR's labels are a reproducible reference;
they are not an expert-consensus claim.

## Prediction and matching contract

A behavior contains an exact quote, global offsets, typed event fields, and a
candidate ledger. A direct suggestion adds one ATT&CK ID whose 27B selection
matches the top labeled-example technique. Every output still requires analyst
acceptance.

Gold mentions collapse to unique `(report, technique)` pairs. Direct
suggestions collapse the same way. A pair matches when the report and exact
technique ID match and at least one cited span overlaps a reference mention.
Parent and sub-technique IDs must match for the exact score. Family scores stay
separate.
