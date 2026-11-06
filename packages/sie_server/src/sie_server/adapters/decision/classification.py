"""Translate the native label classification contract to typed decisions."""

from __future__ import annotations

from string import Formatter
from typing import Any

from sie_server.adapters.decision.questions import MAX_QUESTION_CHARS, MAX_QUESTIONS, Question
from sie_server.core.extract_cost import MAX_EXTRACT_LABELS
from sie_server.types.inputs import InvalidInputError
from sie_server.types.responses import Classification


def classification_schema(labels: list[str], options: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(labels, list)
        or not 1 <= len(labels) <= MAX_EXTRACT_LABELS
        or any(not isinstance(label, str) or not label for label in labels)
    ):
        raise InvalidInputError(f"Classification requires 1 to {MAX_EXTRACT_LABELS} nonempty string labels")
    multi_label = options.get("multi_label", False)
    if not isinstance(multi_label, bool):
        raise InvalidInputError("multi_label must be a boolean")
    independent = multi_label or len(labels) == 1
    if independent and len(labels) > MAX_QUESTIONS:
        raise InvalidInputError(f"Independent classification supports at most {MAX_QUESTIONS} labels")
    template = options.get("hypothesis_template", "This text is about {}.")
    if not isinstance(template, str) or len(template) + sum(map(len, labels)) > MAX_QUESTION_CHARS:
        raise InvalidInputError("Classification labels and template exceed the character budget")
    try:
        fields = [field for field in Formatter().parse(template) if field[1] is not None]
        if not fields or any(name not in {"", "0"} or spec or conversion for _, name, spec, conversion in fields):
            raise ValueError
        if len(labels) * len(template) + len(fields) * sum(map(len, labels)) > MAX_QUESTION_CHARS:
            raise ValueError
        hypotheses = [template.format(label) for label in labels]
    except (ValueError, IndexError, KeyError) as exc:
        raise InvalidInputError(
            "hypothesis_template must use plain {} or {0} placeholders within the character budget"
        ) from exc
    if independent:
        return {str(index): {"type": "noul", "instructions": hypothesis} for index, hypothesis in enumerate(hypotheses)}
    return {
        "0": {
            "type": "choice",
            "instructions": "Choose the best matching classification for the text.",
            "criteria": {str(index): hypothesis for index, hypothesis in enumerate(hypotheses)},
        }
    }


def classifications(
    data: list[dict[str, Any]], labels: list[str], questions: tuple[Question, ...]
) -> list[list[Classification]]:
    independent = questions[0].kind == "noul"
    result: list[list[Classification]] = []
    for item in data:
        row = (
            [
                Classification(
                    label=label,
                    score=item[str(index)]["noul"] if independent else item["0"]["probabilities"][str(index)],
                )
                for index, label in enumerate(labels)
            ]
            if item
            else []
        )
        row.sort(key=lambda classification: classification["score"], reverse=True)
        result.append(row)
    return result
