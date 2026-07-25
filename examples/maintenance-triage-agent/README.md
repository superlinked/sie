# Reconstruct a bearing failure from three detector readings

This example processes the NTSB's illustrated account of the East Palestine
derailment. Its only case input is the published page 4–5 spread from the
NTSB's `SPC-24-06` digest.

The result reconstructs one sequence:

```text
Sebring          7:37 p.m.   38°F above ambient   no alert
Salem            8:13 p.m.  103°F above ambient   noncritical alert to the Wayside Help Desk, not the crew
East Palestine  ~8:52 p.m.  253°F above ambient   critical alarm in the locomotive cab
```

It retains the NTSB's published cause statement. It does not infer a new root
cause, recommend an operational response, or write to a control system.

## What runs on SIE

| Step | Model | Output |
|---|---|---|
| Parse the illustrated PDF spread | `docling-project/docling` | Markdown from the original page layout |
| Retrieve detector and outcome passages | `BAAI/bge-m3` | Dense vectors and cosine ranking |
| Rerank against the sequence question | `Qwen/Qwen3-Reranker-4B` | Ordered source evidence with scores |
| Verify locations, times, alerts, the bearing, and railcar count | `urchade/gliner_multi-v2.1` | Exact source spans |
| Recover detector, alert, and outcome spans | `fastino/gliner2-large-v1` | Labeled exact-source spans |

Every stage consumes the prior stage. GLiNER2 runs once for each detector
paragraph, then on the shortest ranked paragraphs for the cause, engineer
action, and derailment. The review stops if the model omits the event times,
detector readings, alert recipient, camera observation, railcar count, or
bearing-failure span.

Ordinary code then maps exact text from the ranked NTSB paragraphs. It does not
ask GLiNER2 for a JSON schema. One recorded schema probe returned three empty
required fields for this input. Its exact
[request](fixtures/output-schema-probe-request.json) and unedited
[response](fixtures/output-schema-probe-response.json) are preserved. That
single result is diagnostic evidence, not a general model capability claim.

Python performs only transparent calculations: `103 - 38 = 65`,
`253 - 103 = 150`, and `253 - 38 = 215`. It also converts the NTSB's “hopper
car and 37 others” into a total of 38 derailed cars.

## Run it

```bash
cd examples/maintenance-triage-agent
cp .env.example .env
uv sync

uv run triage-fault --run-id local
uv run eval-triage runs/local
```

Set `SIE_CLUSTER_URL` and `SIE_API_KEY` to use SIE Cloud. The default points to
a local server at `http://localhost:8080`.

## Evidence bundle

```text
runs/<run-id>/manifest.json                  endpoint, model IDs, source hash, latency
runs/<run-id>/raw/parse.json                 complete Docling response
runs/<run-id>/raw/retrieve.json              embeddings and cosine ranking
runs/<run-id>/raw/rerank.json                complete reranker response
runs/<run-id>/raw/entities.json              combined GLiNER entity spans
runs/<run-id>/raw/gliner2-sebring.json       GLiNER2 Sebring spans
runs/<run-id>/raw/gliner2-salem.json         GLiNER2 Salem spans
runs/<run-id>/raw/gliner2-east-palestine.json GLiNER2 East Palestine spans
runs/<run-id>/raw/gliner2-cause.json         GLiNER2 bearing-failure span
runs/<run-id>/raw/gliner2-engineer.json      GLiNER2 engineer-action spans
runs/<run-id>/raw/gliner2-derailment.json    GLiNER2 derailed-railcar spans
runs/<run-id>/raw/mapped.json                validated exact-fragment mapping
runs/<run-id>/parsed.md                      parsed NTSB spread used downstream
runs/<run-id>/review.json                    detector trend and explicit boundary
runs/<run-id>/evaluation.json                deterministic checks
```

See [fixtures/SOURCES.md](fixtures/SOURCES.md) for the original NTSB URL,
extraction method, and checksums.
