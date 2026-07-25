# Unsupported structured-output probe

The pipeline does not call GLiNER2 with `output_schema`.

During verification on July 25, 2026 at 01:37 UTC, an exact-source array
schema was tested against the three ranked detector paragraphs. The server ran
public SIE commit `9d6ca6b00f788b6ab19f8d6dc9506e1b31dad2f0` with
`fastino/gliner2-large-v1`.

The model returned exact text for the Salem recipient, crew notification, and
camera observation. It returned empty arrays for the Sebring alert status,
Salem alert level, and East Palestine alarm. That response cannot support the
required record.

The complete unedited response is in
`unsupported-output-schema-response.json`. The verified pipeline uses labeled
source-span extraction and deterministic exact-source mapping instead.
