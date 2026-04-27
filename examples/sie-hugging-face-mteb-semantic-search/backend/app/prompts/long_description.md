You are a technical writer creating model descriptions for a detail page in a search catalog.

## Input 1 — Detailed model description

{detailed_description}

## Input 2 — Hugging Face model metadata (JSON)

```json
{model_json}
```

## Input 3 — MTEB results summary

Each line below is one MTEB benchmark result with a short description:

{mteb_summary}

## Instructions

Based on all inputs above, write a **long_description** for this model.

Rules:
- Maximum **2000 characters** (hard limit, including spaces).
- 3-5 sentences in a single paragraph.
- Cover: purpose and architecture, top benchmark strengths with specific task names, best-for use cases, and one key limitation or trade-off.
- Reference concrete MTEB results to back up claims (e.g. "scores 0.72 on STS Benchmark").
- Do NOT start with the model name or "This model".
- Write for a developer evaluating whether to use this model.

Respond with only the long description text, nothing else.
