"""Source-aware answer formats; never inspect the target or the answering model."""

import re

VERSION = "bbeh-source-formats-v2"
SINGLE_CHOICE_TASKS = frozenset({
    "boolean_expressions", "disambiguation_qa", "geometric_shapes",
    "movie_recommendation", "nycc", "shuffled_objects",
})
CHOICE = re.compile(r"^\(([A-Z])\)[ \t]+", re.MULTILINE)
GEOMETRY_SUFFIX = "\nCoordinates have been rounded to 5 decimal places so ignore slight differences."


def text_key(value):
    # Whitespace is presentational for these choice texts, including multiline
    # captions. Do not remove words, quotes, numbers, brackets or negations.
    return " ".join(value.split()).casefold().removesuffix(".")


def options_from_question(request, task):
    matches = list(CHOICE.finditer(request))
    options = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(request)
        text = request[match.end():end].strip()
        if task == "geometric_shapes" and index == len(matches) - 1:
            text = text.removesuffix(GEOMETRY_SUFFIX).strip()
        label = match.group(1)
        if not text:
            raise ValueError("Single-choice source has an empty option")
        # One public source question repeats a label. Preserve the benchmark's
        # label semantics; never guess a repaired label or reference answer.
        options.setdefault(label, []).append(text)
    if len(options) < 2:
        raise ValueError("Single-choice source has no unambiguous option block")
    return options


def source_flags(request, task):
    if task not in SINGLE_CHOICE_TASKS:
        return []
    options = options_from_question(request, task)
    flags = []
    if any(len(texts) != 1 for texts in options.values()):
        flags.append("duplicate_source_option_label")
    texts_to_labels = {}
    for label, texts in options.items():
        for text in texts:
            texts_to_labels.setdefault(text_key(text), set()).add(label)
    if any(len(labels) > 1 for labels in texts_to_labels.values()):
        flags.append("identical_source_option_text_under_distinct_labels")
    return flags


def canonicalize(answer, request, task):
    """Return (canonical answer or None, rule), without access to gold labels.

    A label followed by text is accepted only if the entire text agrees with
    that source option. Text alone must identify exactly one option. Unknown,
    contradictory or multiple-choice responses fail closed for single-choice
    tasks. All other task-specific punctuation is preserved.
    """
    if answer is None:
        return None, "missing"
    value = answer.strip()
    if task in SINGLE_CHOICE_TASKS:
        options = options_from_question(request, task)
        bare = re.fullmatch(r"(?:\(([A-Za-z])\)|([A-Za-z]))", value)
        if bare:
            label = (bare.group(1) or bare.group(2)).upper()
            return (f"({label})", "choice_label") if label in options else (None, "invalid_choice_label")
        labelled = re.fullmatch(r"\(([A-Za-z])\)\s+(.+)", value, flags=re.DOTALL)
        if labelled:
            label, text = labelled.group(1).upper(), labelled.group(2)
            if label in options and text_key(text) in {text_key(option) for option in options[label]}:
                return f"({label})", "choice_label_and_exact_text"
            return None, "choice_label_text_disagreement"
        matching = [label for label, texts in options.items() if text_key(value) in {text_key(text) for text in texts}]
        if len(matching) == 1:
            return f"({matching[0]})", "unique_exact_choice_text"
        return None, "unrecognized_or_ambiguous_choice"
    if task == "web_of_lies":
        # The public task itself demonstrates <unknown, no, yes> etc. These
        # delimiters mark the whole three-answer tuple, not reasoning syntax.
        marker = "Your answer should be a list of three words"
        if marker not in request:
            raise ValueError("Unrecognized source truth-tuple contract")
        instruction = request.rsplit(marker, 1)[1]
        terms = "yes|no|unknown" if "unknown" in instruction else "yes|no"
        inner = value[1:-1].strip() if value.startswith("<") and value.endswith(">") else value
        matched = re.fullmatch(rf"({terms})\s*,\s*({terms})\s*,\s*({terms})", inner, flags=re.IGNORECASE)
        if matched:
            return ", ".join(part.lower() for part in matched.groups()), "source_truth_tuple"
        return None, "invalid_truth_tuple"
    return value, "unchanged"
