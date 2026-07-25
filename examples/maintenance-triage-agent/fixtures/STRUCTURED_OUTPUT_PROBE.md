# Recorded structured-output probe

The pipeline does not call GLiNER2 with `output_schema`.

During development, an exact-source array schema was tested against the three
ranked detector paragraphs. The request record includes the exact input,
schema, endpoint, runtime commit, timestamp, and call options:
`output-schema-probe-request.json`.

That one response contains exact text for the Salem recipient, crew
notification, and camera observation. It contains empty arrays for the Sebring
alert status, Salem alert level, and East Palestine alarm. The response cannot
support this example's required record. It does not establish a general model
capability boundary.

The complete unedited response is in `output-schema-probe-response.json`. The
verified pipeline uses labeled source-span extraction and deterministic
exact-source mapping instead.
