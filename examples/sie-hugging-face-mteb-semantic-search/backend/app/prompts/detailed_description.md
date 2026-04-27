You are an expert in machine learning embedding models and benchmarking.

Your task is to write a comprehensive, detailed description of the model **{model_id}**.

## Input 1 — Hugging Face model metadata (JSON)

```json
{model_json}
```

## Input 2 — Model README (documentation)

The model's own documentation from its Hugging Face README, in markdown format (truncated to 4 000 characters):

{model_readme}

## Input 3 — MTEB benchmark results summary

The model was evaluated on the following MTEB tasks (averaged main score per task). For each result, provide a short contextual summary (max 100 characters) that explains what the score means in practice (e.g. "Strong retrieval on scientific papers" or "Average clustering on news topics").

{mteb_summary}

## Instructions

Using all inputs above, write a detailed description of this model (maximum 6000 characters).

Structure your description as follows:

1. **Model overview** — What it is, who made it, what architecture it is based on (e.g. BERT, RoBERTa, T5), embedding dimensions, max sequence length, supported languages.

2. **MTEB benchmark analysis** — For every MTEB task result provided, include:
   - The task name and type (retrieval, classification, clustering, STS, etc.).
   - The score and a short contextual description (max 100 characters) that puts the score in perspective.
   - How this result compares to typical performance for this task type.

3. **Best suited for** — Based on the benchmark results and model characteristics, clearly state the use cases where this model excels. Be specific: name concrete scenarios (e.g. "semantic search over legal documents", "short-text similarity for product matching", "multilingual FAQ retrieval").

4. **Limitations and trade-offs** — Model size vs speed, domains where performance drops, language coverage gaps, max token constraints.

5. **Usage recommendations** — Practical guidance: recommended frameworks (sentence-transformers, transformers), pooling strategy, whether normalization helps, batch size tips.

Write in a factual, information-dense style. Every sentence should add value. Do not pad with generic filler.
