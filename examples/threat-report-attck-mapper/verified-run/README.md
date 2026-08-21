# Verified Proofpoint run

This directory records the August 21, 2026 run of `attck-map demo` against the
pinned Proofpoint report.

Proofpoint published “MFA PSA, Oh My!” and contributed the report to
[AnnoCTR](https://github.com/boschresearch/anno-ctr-lrec-coling-2024). Bosch
Research republishes the corpus under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The dataset is
documented in the [AnnoCTR paper](https://aclanthology.org/2024.lrec-main.103/).

- `parsed-report.md` is the complete AnnoCTR report text consumed by the agent.
- `api-calls.json` keeps each SIE request and raw response.
- `review.json` keeps all extracted behaviors, candidate ledgers, and routing
  decisions.
- `manifest.json` pins the source, taxonomy, model revisions, and artifact
  checksums.

The three compressed vector files total 85 MB and are omitted from Git. Their
hashes remain in `manifest.json`; rerunning the command rebuilds them.
