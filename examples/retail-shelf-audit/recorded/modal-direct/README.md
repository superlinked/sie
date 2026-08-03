# Direct checkpoint evidence

These files came from direct Modal GPU runs against the exact launch checkpoints. They show the detector and OCR outputs used by this public case. They do not claim an SIE Cloud API run.

| File | Contents |
|---|---|
| `launch-model-results.json` | Grounding DINO batch record containing the case-042 boxes used by the verifier |
| `042-grounding-dino.jpg` | Case 042 with DINO boxes drawn for review |
| `derived-crops/candidate-1.jpg` | Upper notice crop produced by the runner's DINO geometry rules |
| `derived-crops/candidate-2.jpg` | Lower shelf-label crop produced by the same rules |
| `lighton-ocr-derived-results.json` | Raw LightOnOCR text for both DINO-derived crops, including source boxes and crop coordinates |

`uv run verify-retail-records` reruns the geometry selector against the detector record, rebuilds the reviewed evidence from raw OCR text, and checks every listed file hash.
