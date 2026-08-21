# Sources and evaluation boundary

The example downloads two immutable public sources and verifies each file by
SHA-256 before use.

## MITRE ATT&CK Enterprise 19.2

- Source: `mitre-attack/attack-stix-data`
- Release: Enterprise ATT&CK 19.2, modified August 5, 2026
- Commit: `6cda5ad8462c79e14fbb872f4e09059b18e0cfc4`
- Terms: MITRE ATT&CK terms of use. ATT&CK must be attributed to The MITRE
  Corporation.

## AnnoCTR

- Source: `boschresearch/anno-ctr-lrec-coling-2024`
- Commit: `d510b6949e1938d47c93a43eedd562dc538439dc`
- License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) for
  the `AnnoCTR/` corpus
- Paper: Lukas Lange et al., [“AnnoCTR: A Dataset for Detecting and Linking
  Entities, Tactics, and Techniques in Cyber Threat
  Reports”](https://aclanthology.org/2024.lrec-main.103/), LREC-COLING 2024.

Proofpoint contributed “MFA PSA, Oh My!” to AnnoCTR and remains the original
publisher of the complete report used in `verified-run/parsed-report.md`.

The benchmark uses AnnoCTR's published train, dev, and test files. Rows that
share one annotated span are grouped into a single multi-label case. Historical
linking is scored against the 578-technique MITRE entity snapshot distributed
with AnnoCTR. Full-report review uses active ATT&CK 19.2. Keeping the catalogs
separate prevents a correct current mapping from being marked wrong because an
older technique definition or label changed.

The legacy `benchmark` command starts from an annotated behavior span. The
`full-benchmark` command starts from complete report text and evaluates behavior
extraction separately from ATT&CK linking.
