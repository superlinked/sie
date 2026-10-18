"""Use the OpenAI Decisions client against a running SIE server."""

from __future__ import annotations

import os

from openai import OpenAI


def main() -> None:
    with OpenAI(
        base_url=os.environ.get("SIE_OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
        api_key=os.environ.get("SIE_API_KEY", "local"),
    ) as client:
        result = client.decisions.create(
            model=os.environ.get("SIE_DECISION_MODEL", "fastino/GLiNER2.5-Decide"),
            input="I was charged twice. Please refund one charge today.",
            questions=[
                {"type": "predicate", "name": "refund", "instructions": "Does the customer request a refund?"},
                {
                    "type": "choice",
                    "name": "team",
                    "instructions": "Which team should handle this?",
                    "choices": [{"value": "billing"}, {"value": "technical"}],
                },
                {
                    "type": "score",
                    "name": "urgency",
                    "instructions": "How urgent is the request?",
                    "levels": [{"label": "can wait"}, {"label": "today"}],
                },
            ],
        )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
