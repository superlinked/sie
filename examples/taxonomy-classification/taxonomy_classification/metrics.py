from __future__ import annotations

from typing import Sequence

from taxonomy_classification.taxonomy import CategoryPath, ancestor_set


def category_prefix(category: CategoryPath, level: int) -> CategoryPath | None:
    if len(category) < level:
        return None
    return category[:level]


def hierarchical_precision(
    predictions: Sequence[CategoryPath], references: Sequence[CategoryPath]
) -> float:
    overlap_count = 0
    predicted_count = 0

    for prediction, reference in zip(predictions, references, strict=True):
        predicted_ancestors = ancestor_set(prediction)
        reference_ancestors = ancestor_set(reference)
        overlap_count += len(predicted_ancestors & reference_ancestors)
        predicted_count += len(predicted_ancestors)

    if predicted_count == 0:
        return 0.0

    return overlap_count / predicted_count


def hierarchical_recall(
    predictions: Sequence[CategoryPath], references: Sequence[CategoryPath]
) -> float:
    overlap_count = 0
    reference_count = 0

    for prediction, reference in zip(predictions, references, strict=True):
        predicted_ancestors = ancestor_set(prediction)
        reference_ancestors = ancestor_set(reference)
        overlap_count += len(predicted_ancestors & reference_ancestors)
        reference_count += len(reference_ancestors)

    if reference_count == 0:
        return 0.0

    return overlap_count / reference_count


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hierarchical_metrics(
    predictions: Sequence[CategoryPath], references: Sequence[CategoryPath]
) -> dict[str, float]:
    precision = hierarchical_precision(predictions, references)
    recall = hierarchical_recall(predictions, references)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


def macro_metrics(
    predictions: Sequence[CategoryPath | None],
    references: Sequence[CategoryPath | None],
) -> dict[str, float]:
    labels = sorted(
        {label for label in predictions if label is not None}
        | {label for label in references if label is not None}
    )

    if not labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for label in labels:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for prediction, reference in zip(predictions, references, strict=True):
            if prediction == label and reference == label:
                true_positives += 1
            elif prediction == label and reference != label:
                false_positives += 1
            elif prediction != label and reference == label:
                false_negatives += 1

        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1_score(precision, recall)

    label_count = len(labels)
    return {
        "precision": precision_sum / label_count,
        "recall": recall_sum / label_count,
        "f1": f1_sum / label_count,
    }


def level_metrics(
    predictions: Sequence[CategoryPath], references: Sequence[CategoryPath], level: int
) -> dict[str, float]:
    level_predictions: list[CategoryPath | None] = []
    level_references: list[CategoryPath] = []

    for prediction, reference in zip(predictions, references, strict=True):
        reference_label = category_prefix(reference, level)
        if reference_label is None:
            continue
        level_predictions.append(category_prefix(prediction, level))
        level_references.append(reference_label)

    return macro_metrics(level_predictions, level_references)


def exact_lenient_references(
    predictions: Sequence[CategoryPath],
    ground_truth_categories: Sequence[CategoryPath],
    potential_product_categories: Sequence[list[CategoryPath]],
) -> list[CategoryPath]:
    references: list[CategoryPath] = []

    for prediction, ground_truth, potential_categories in zip(
        predictions,
        ground_truth_categories,
        potential_product_categories,
        strict=True,
    ):
        acceptable_categories = {ground_truth, *potential_categories}
        if prediction in acceptable_categories:
            references.append(prediction)
        else:
            references.append(ground_truth)

    return references


def evaluate_predictions(
    predictions: Sequence[CategoryPath], references: Sequence[CategoryPath]
) -> dict[str, object]:
    return {
        "hierarchical": hierarchical_metrics(predictions, references),
        "macro": {
            "l1": level_metrics(predictions, references, 1),
            "l2": level_metrics(predictions, references, 2),
            "l3": level_metrics(predictions, references, 3),
        },
    }


def score_predictions(
    predictions: Sequence[CategoryPath],
    ground_truth_categories: Sequence[CategoryPath],
    potential_product_categories: Sequence[list[CategoryPath]],
) -> dict[str, object]:
    strict_references = list(ground_truth_categories)
    lenient_references = exact_lenient_references(
        predictions,
        ground_truth_categories,
        potential_product_categories,
    )

    return {
        "strict": evaluate_predictions(predictions, strict_references),
        "lenient": evaluate_predictions(predictions, lenient_references),
    }
