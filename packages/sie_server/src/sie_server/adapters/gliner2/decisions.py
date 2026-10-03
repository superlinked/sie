"""Typed decisions for GLiNER2.5-Decide: request parsing, the model's task schema, and answers.

GLiNER2.5-Decide reads a *classification task* as a name, an optional prompt,
and a label set, where each label may carry a description. This module maps
the three request shapes the other SIE decision models accept onto such tasks,
and turns the model's per-label logits back into their answers:

* ``output_schema``: Laya's typed question mapping ``{question_id: {"type",
  "instructions", "criteria", "labels"?}}``, validated by Laya's own rules. The
  question id is the task name and ``instructions`` its prompt.

  - ``choice``: one label per criterion; a criterion's description (dict form)
    becomes the label's description.
  - ``score``: levels ``"0"`` .. ``"k-1"`` (the ordinal labels the model reads),
    each described by its criterion.
  - ``noul``: the labels ``"yes"`` and ``"no"`` (or the question's ``labels``
    wording), described by ``criteria["true"]`` / ``criteria["false"]``.

* ``options.label_groups``: GLiClass's ``{group: [label, ...]}``; the group name is
  the task name and a request ``instruction`` the prompt of every group.
  ``options.classification_type`` ("single-label" or "multi-label") applies to
  every group.
* ``labels``: one task named ``options.classification_task`` (default
  ``"label"``), with ``instruction`` as its prompt.

Single-label tasks are a softmax over their labels and multi-label tasks an
independent sigmoid per label, as ``gliner2.classification`` computes them.
Nothing here imports torch or gliner2.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from sie_server.adapters.laya.questions import (
    Question,
    confidence_from_probs,
    parse_questions,
    render_criterion,
    resolve_noul_labels,
    shown,
)
from sie_server.types.inputs import InvalidInputError
from sie_server.types.responses import Classification

TaskKind = Literal["choice", "score", "noul", "group", "labels"]
RequestMode = Literal["questions", "groups", "labels"]

# Tasks (questions or label groups) per request. Every task is encoded in each
# item's single row, so the row's token budget binds long before this does.
MAX_TASKS = 64
# Labels (answer options) per task: gliner2.classification's default candidate
# cap per task (ClassificationConfig.max_candidates_per_task).
MAX_LABELS_PER_TASK = 64
# Labels across all tasks of a request.
MAX_LABELS = 1024
# Characters in a task name (question id, group name) and in a label.
MAX_NAME_CHARS = 128
MAX_LABEL_CHARS = 256
# Characters in one free-text field: a prompt (instructions) or a label description.
MAX_TEXT_CHARS = 2048
# Characters of every string the model reads in its task schema together. All
# of them are checked against these bounds before anything is tokenized.
MAX_SCHEMA_CHARS = 65_536

DEFAULT_LABELS_TASK = "label"
NOUL_WORDS = ("no", "yes")  # (false, true) labels the model reads for a noul question
DEFAULT_MULTI_LABEL_THRESHOLD = 0.5
GROUP_LABEL_SEPARATOR = "."  # "group.label" classifications, as the GLiClass adapter names them

# Structural tokens of the GLiNER2 prompt. A task name, label, prompt, or
# description containing one would be read as structure, so they are refused.
# (The document cannot inject them: it is split into lowercased words first.)
MARKERS = (
    "[P]",
    "[L]",
    "[C]",
    "[E]",
    "[R]",
    "[DESCRIPTION]",
    "[EXAMPLE]",
    "[OUTPUT]",
    "[SEP_STRUCT]",
    "[SEP_TEXT]",
)

_ERR_NO_TASK = (
    "GLiNER2.5-Decide requires typed questions in output_schema ({question_id: {type, instructions, criteria}}), "
    "options.label_groups, or labels"
)
_ERR_SCHEMA_AND_LABELS = "GLiNER2.5-Decide takes either output_schema (typed questions) or labels, not both"
_ERR_SCHEMA_AND_GROUPS = (
    "GLiNER2.5-Decide takes either output_schema (typed questions) or options.label_groups, not both"
)
_ERR_GROUPS_AND_LABELS = "GLiNER2.5-Decide takes either labels or options.label_groups, not both"
_ERR_SCHEMA_AND_INSTRUCTION = (
    "GLiNER2.5-Decide reads each question's own 'instructions' from output_schema; "
    "the request-level instruction applies only to labels and label_groups"
)


@dataclass(frozen=True, slots=True)
class DecisionTask:
    """One classification task the model reads, and how its answer is reported.

    Attributes:
        key: The answer's key in ``data`` (question id or group name) or, in
            labels mode, the task name.
        name: Task name the model reads.
        kind: ``choice``/``score``/``noul`` (output_schema), ``group``
            (label_groups), or ``labels``.
        labels: Label names the model reads, in order.
        keys: Output key of each label: the caller's label, or the level index
            of a score question.
        descriptions: Each label's description, or None.
        prompt: The task prompt, or None.
        multi_label: Independent sigmoid per label instead of a softmax.
        legend: A score question's criteria, index 0 first.
        true_index: Position of the "true" label of a noul question.
    """

    key: str
    name: str
    kind: TaskKind
    labels: tuple[str, ...]
    keys: tuple[str, ...]
    descriptions: tuple[str | None, ...]
    prompt: str | None = None
    multi_label: bool = False
    legend: tuple[Any, ...] = ()
    true_index: int = 0

    def model_entry(self) -> dict[str, Any]:
        """This task as a ``classifications`` entry of a gliner2 schema dict.

        The same entry ``gliner2.classification.compile_schema`` emits for a
        task with these labels, descriptions, and instruction.
        """
        entry: dict[str, Any] = {
            "task": self.name,
            "labels": list(self.labels),
            "true_label": ["N/A"],  # gliner2 2.0 reads it unconditionally
            "multi_label": self.multi_label,
            "cls_threshold": 0.5,
            "class_act": "auto",
        }
        if self.prompt:
            entry["prompt"] = self.prompt
        descriptions = {
            label: description for label, description in zip(self.labels, self.descriptions, strict=True) if description
        }
        if descriptions:
            entry["label_descriptions"] = descriptions
        return entry

    def free_texts(self) -> Iterator[str]:
        """The caller's free text this task puts in front of every document (billed)."""
        if self.prompt:
            yield self.prompt
        yield from (description for description in self.descriptions if description)


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """A validated request: its tasks, in order, and how answers are reported."""

    mode: RequestMode
    tasks: tuple[DecisionTask, ...]
    threshold: float = 0.0

    def model_schema(self) -> dict[str, Any]:
        """The gliner2 schema dict for every task (classifications only)."""
        return {
            "json_structures": [],
            "classifications": [task.model_entry() for task in self.tasks],
            "entities": {},
            "relations": [],
            "json_descriptions": {},
            "entity_descriptions": {},
        }

    @property
    def label_count(self) -> int:
        return sum(len(task.labels) for task in self.tasks)


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


def parse_request(
    *,
    labels: list[str] | None,
    output_schema: dict[str, Any] | None,
    instruction: str | None,
    options: Mapping[str, Any],
) -> DecisionRequest:
    """Validate a request and build its tasks. Every bound is checked before anything is tokenized.

    Raises:
        InvalidInputError: For any malformed or out-of-bounds request.
    """
    groups = options.get("label_groups")
    _check_unsupported_options(options)
    if output_schema is not None:
        if labels:
            raise InvalidInputError(_ERR_SCHEMA_AND_LABELS)
        if groups is not None:
            raise InvalidInputError(_ERR_SCHEMA_AND_GROUPS)
        if instruction is not None:
            raise InvalidInputError(_ERR_SCHEMA_AND_INSTRUCTION)
        request = DecisionRequest(mode="questions", tasks=_question_tasks(output_schema))
    else:
        prompt = _free_text(instruction, "instruction")
        multi_label = multi_label_option(options)
        threshold = threshold_option(options)
        if groups is not None:
            if labels:
                raise InvalidInputError(_ERR_GROUPS_AND_LABELS)
            tasks = _group_tasks(groups, prompt=prompt, multi_label=multi_label)
            request = DecisionRequest(mode="groups", tasks=tasks, threshold=threshold)
        elif labels:
            task = _labels_task(labels, options.get("classification_task"), prompt=prompt, multi_label=multi_label)
            request = DecisionRequest(mode="labels", tasks=(task,), threshold=threshold)
        else:
            raise InvalidInputError(_ERR_NO_TASK)
    _check_totals(request)
    return request


def _check_unsupported_options(options: Mapping[str, Any]) -> None:
    encoding = options.get("group_encoding")
    if encoding is not None and encoding != "joint":
        raise InvalidInputError(
            "GLiNER2.5-Decide scores every question and label group of an item in one encoder row "
            "(group_encoding 'joint'); other encodings are not supported"
        )
    if options.get("examples") is not None:
        raise InvalidInputError("GLiNER2.5-Decide does not take few-shot examples")


def _question_tasks(output_schema: Any) -> tuple[DecisionTask, ...]:
    if not isinstance(output_schema, Mapping):
        raise InvalidInputError("GLiNER2.5-Decide output_schema must map question ids to question definitions")
    if not output_schema:
        raise InvalidInputError(_ERR_NO_TASK)
    if len(output_schema) > MAX_TASKS:
        raise InvalidInputError(
            f"GLiNER2.5-Decide accepts at most {MAX_TASKS} questions per request, got {len(output_schema)}"
        )
    for qid, qdef in output_schema.items():
        declared = _declared_options(qdef)
        if declared > MAX_LABELS_PER_TASK:
            raise InvalidInputError(
                f"question {shown(qid)}: GLiNER2.5-Decide accepts at most {MAX_LABELS_PER_TASK} options "
                f"per question, got {declared}"
            )
    questions = parse_questions(output_schema)
    tasks = tuple(_question_task(q) for q in questions)
    _check_unique_names(tasks)
    return tasks


def _declared_options(qdef: Any) -> int:
    """Options a raw question definition declares, counted without walking its criteria."""
    if not isinstance(qdef, Mapping) or qdef.get("type") == "noul":
        return 0
    criteria = qdef.get("criteria")
    return len(criteria) if isinstance(criteria, (Mapping, list)) else 0


def _question_task(q: Question) -> DecisionTask:
    where = f"question {shown(q.qid)}"
    name = _name(q.qid, f"{where}: the question id", MAX_NAME_CHARS)
    prompt = _free_text(q.instructions, f"{where}: instructions")
    if q.qtype == "choice":
        criteria: dict[str, Any] = q.criteria
        keys = tuple(criteria)
        labels = _labels([str(key) for key in keys], where)
        descriptions = tuple(_description(value, where) for value in criteria.values())
        return DecisionTask(
            key=q.qid, name=name, kind="choice", labels=labels, keys=keys, descriptions=descriptions, prompt=prompt
        )
    if q.qtype == "score":
        levels: list[Any] = q.criteria
        labels = tuple(str(i) for i in range(len(levels)))
        descriptions = tuple(
            None if _is_label_itself(level, label) else _description(level, where)
            for label, level in zip(labels, levels, strict=True)
        )
        return DecisionTask(
            key=q.qid,
            name=name,
            kind="score",
            labels=labels,
            keys=labels,
            descriptions=descriptions,
            prompt=prompt,
            legend=tuple(levels),
        )
    false_word, true_word = resolve_noul_labels(q.labels) if q.labels is not None else NOUL_WORDS
    labels = _labels([true_word, false_word], where)
    criteria = q.criteria or {}
    descriptions = (_description(criteria.get("true"), where), _description(criteria.get("false"), where))
    return DecisionTask(
        key=q.qid,
        name=name,
        kind="noul",
        labels=labels,
        keys=("true", "false"),
        descriptions=descriptions,
        prompt=prompt,
        true_index=0,
    )


def _is_label_itself(level: Any, label: str) -> bool:
    """A score level that just repeats its own index ("0", "1", ...) needs no description."""
    return level in (None, "") or (isinstance(level, str) and level.strip() == label)


def _group_tasks(groups: Any, *, prompt: str | None, multi_label: bool) -> tuple[DecisionTask, ...]:
    if not isinstance(groups, Mapping) or not groups:
        raise InvalidInputError(
            "GLiNER2.5-Decide label_groups must be a non-empty object mapping group names to label lists"
        )
    if len(groups) > MAX_TASKS:
        raise InvalidInputError(
            f"GLiNER2.5-Decide accepts at most {MAX_TASKS} label groups per request, got {len(groups)}"
        )
    tasks: list[DecisionTask] = []
    for group, group_labels in groups.items():
        where = f"label_groups[{shown(group)}]"
        name = _name(group, f"{where}: the group name", MAX_NAME_CHARS)
        if not isinstance(group_labels, list) or not group_labels:
            raise InvalidInputError(f"GLiNER2.5-Decide {where} must be a non-empty list of labels")
        if len(group_labels) > MAX_LABELS_PER_TASK:
            raise InvalidInputError(
                f"GLiNER2.5-Decide {where} may have at most {MAX_LABELS_PER_TASK} labels, got {len(group_labels)}"
            )
        labels = _labels(group_labels, where)
        tasks.append(
            DecisionTask(
                key=group,
                name=name,
                kind="group",
                labels=labels,
                keys=tuple(group_labels),
                descriptions=(None,) * len(labels),
                prompt=prompt,
                multi_label=multi_label,
            )
        )
    _check_unique_names(tasks)
    return tuple(tasks)


def _labels_task(labels: Any, task_name: Any, *, prompt: str | None, multi_label: bool) -> DecisionTask:
    if not isinstance(labels, list):
        raise InvalidInputError("GLiNER2.5-Decide labels must be a list of non-empty strings")
    if len(labels) > MAX_LABELS_PER_TASK:
        raise InvalidInputError(
            f"GLiNER2.5-Decide accepts at most {MAX_LABELS_PER_TASK} labels per task, got {len(labels)}"
        )
    name = DEFAULT_LABELS_TASK if task_name is None else _name(task_name, "classification_task", MAX_NAME_CHARS)
    return DecisionTask(
        key=name,
        name=name,
        kind="labels",
        labels=_labels(labels, "labels"),
        keys=tuple(labels),
        descriptions=(None,) * len(labels),
        prompt=prompt,
        multi_label=multi_label,
    )


def _labels(values: Sequence[Any], where: str) -> tuple[str, ...]:
    labels = tuple(_name(value, f"{where}: a label", MAX_LABEL_CHARS) for value in values)
    if len(set(labels)) != len(labels):
        raise InvalidInputError(
            f"GLiNER2.5-Decide {where}: labels must be unique once surrounding whitespace is removed"
        )
    return labels


def _check_unique_names(tasks: Sequence[DecisionTask]) -> None:
    names = [task.name for task in tasks]
    if len(set(names)) != len(names):
        raise InvalidInputError(
            "GLiNER2.5-Decide question ids and label group names must be unique once surrounding whitespace is removed"
        )


def _name(value: Any, what: str, limit: int) -> str:
    """A task name or label as the model reads it: stripped, non-empty, bounded, no markers."""
    if not isinstance(value, str) or not value.strip():
        raise InvalidInputError(f"GLiNER2.5-Decide {what} must be a non-empty string")
    if len(value) > limit:
        raise InvalidInputError(f"GLiNER2.5-Decide {what} may have at most {limit} characters, got {len(value)}")
    _check_markers(value, what)
    return value.strip()


def _free_text(value: Any, what: str) -> str | None:
    """A prompt as the model reads it: None when empty, else bounded and without markers."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidInputError(f"GLiNER2.5-Decide {what} must be a string")
    if len(value) > MAX_TEXT_CHARS:
        raise InvalidInputError(
            f"GLiNER2.5-Decide {what} may have at most {MAX_TEXT_CHARS} characters, got {len(value)}"
        )
    _check_markers(value, what)
    return value.strip() or None


def _description(value: Any, where: str) -> str | None:
    """A criterion as a label description: strings as given, other values as compact JSON (as Laya renders them)."""
    if value is None or value == "":
        return None
    what = f"{where}: a criterion"
    try:
        text = render_criterion(value, limit=4 * MAX_TEXT_CHARS + 4)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"GLiNER2.5-Decide {what} must be a string or JSON-serializable") from exc
    return _free_text(text, what)


def _check_markers(value: str, what: str) -> None:
    for marker in MARKERS:
        if marker in value:
            raise InvalidInputError(
                f"GLiNER2.5-Decide {what} may not contain {marker!r}, a structural token of the model's prompt"
            )


def _check_totals(request: DecisionRequest) -> None:
    if request.label_count > MAX_LABELS:
        raise InvalidInputError(
            f"GLiNER2.5-Decide accepts at most {MAX_LABELS} labels or answer options per request, "
            f"got {request.label_count}"
        )
    chars = 0
    for task in request.tasks:
        chars += len(task.name) + sum(len(label) for label in task.labels)
        chars += sum(len(text) for text in task.free_texts())
    if chars > MAX_SCHEMA_CHARS:
        raise InvalidInputError(
            f"GLiNER2.5-Decide questions, labels, prompts, and descriptions may total at most {MAX_SCHEMA_CHARS} "
            f"characters, got {chars}"
        )


def multi_label_option(options: Mapping[str, Any]) -> bool:
    """``classification_type`` (GLiClass spelling) or ``multi_label`` (GLiNER2 spelling); default single-label."""
    classification_type = options.get("classification_type")
    multi_label = options.get("multi_label")
    if multi_label is not None and not isinstance(multi_label, bool):
        raise InvalidInputError("GLiNER2.5-Decide multi_label must be a boolean")
    if classification_type is None:
        return bool(multi_label)
    if classification_type not in ("single-label", "multi-label"):
        raise InvalidInputError("GLiNER2.5-Decide classification_type must be 'single-label' or 'multi-label'")
    resolved = classification_type == "multi-label"
    if multi_label is not None and multi_label != resolved:
        raise InvalidInputError("GLiNER2.5-Decide multi_label contradicts classification_type")
    return resolved


def threshold_option(options: Mapping[str, Any]) -> float:
    value = options.get("threshold", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError("GLiNER2.5-Decide threshold must be a finite number between 0 and 1")
    try:
        threshold = float(value)
    except OverflowError:  # an int too large for a float
        threshold = math.inf
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise InvalidInputError("GLiNER2.5-Decide threshold must be a finite number between 0 and 1")
    return threshold


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


def probabilities(task: DecisionTask, logits: np.ndarray) -> np.ndarray:
    """Softmax over a single-label task's logits, or a sigmoid per label of a multi-label task (float64)."""
    z = np.asarray(logits, dtype=np.float64)
    if task.multi_label:
        return np.exp(-np.logaddexp(0.0, -z))
    e = np.exp(z - z.max())
    return e / e.sum()


def answer(task: DecisionTask, p: np.ndarray, *, threshold: float = 0.0) -> dict[str, Any]:
    """One task's answer in the shape the Laya adapter (questions) or GLiClass adapter (groups) returns.

    ``confidence`` is ``1 - H(p) / log(k)`` for choice, score, and single-label
    groups, and ``max(p, 1 - p)`` for noul questions (Laya's definitions).
    """
    k = len(task.labels)
    probs = {key: float(value) for key, value in zip(task.keys, p, strict=True)}
    if task.kind == "score":
        return {
            "type": "score",
            "score": float(np.dot(np.arange(k, dtype=np.float64), p)),
            "legend": {str(i): level for i, level in enumerate(task.legend)},
            "probabilities": probs,
            "confidence": confidence_from_probs(p, k),
        }
    if task.kind == "noul":
        p_true = float(p[task.true_index])
        p_false = float(p[1 - task.true_index])
        return {"type": "noul", "noul": p_true, "answer": p_true > p_false, "confidence": max(p_true, 1.0 - p_true)}
    if task.multi_label:
        selection = threshold if threshold > 0.0 else DEFAULT_MULTI_LABEL_THRESHOLD
        return {"labels": [key for key, value in probs.items() if value >= selection], "probabilities": probs}
    return {
        "type": "choice",
        "choice": task.keys[int(np.argmax(p))],
        "probabilities": probs,
        "confidence": confidence_from_probs(p, k),
    }


def classifications(request: DecisionRequest, per_task: Sequence[np.ndarray], *, grouped: bool) -> list[Classification]:
    """Every label with its probability, highest first, filtered by the request threshold.

    Label groups name their labels ``"group.label"``, as the GLiClass adapter does.
    """
    result: list[Classification] = []
    for task, p in zip(request.tasks, per_task, strict=True):
        for key, value in zip(task.keys, p, strict=True):
            label = f"{task.key}{GROUP_LABEL_SEPARATOR}{key}" if grouped else key
            if float(value) >= request.threshold:
                result.append(Classification(label=label, score=float(value)))
    result.sort(key=lambda c: c["score"], reverse=True)
    return result


def split_logits(request: DecisionRequest, logits: np.ndarray) -> list[np.ndarray]:
    """A row's label logits, one array per task (tasks' labels are consecutive in the row)."""
    parts: list[np.ndarray] = []
    offset = 0
    for task in request.tasks:
        parts.append(logits[offset : offset + len(task.labels)])
        offset += len(task.labels)
    return parts


def schema_chars(output_schema: Any, labels: Any, options: Mapping[str, Any], limit: int) -> int:
    """Approximate characters of a request's task schema, capped at ``limit``; never raises (batching cost)."""
    try:
        if output_schema is not None:
            return approx_json_chars(output_schema, limit)
        total = approx_json_chars(options.get("label_groups"), limit) if options.get("label_groups") else 0
        if isinstance(labels, list):
            total += approx_json_chars(labels, limit)
        return min(total, limit)
    except Exception:  # noqa: BLE001 -- a cost estimate must never fail a request
        return 0


def approx_json_chars(value: Any, limit: int) -> int:
    """Approximate ``len(json.dumps(value))``, visiting at most ``limit`` characters' worth of values."""
    total = 0
    stack: list[Iterator[Any]] = [iter((value,))]
    while stack and total < limit:
        node = next(stack[-1], _END)
        if node is _END:
            stack.pop()
        elif isinstance(node, str):
            total += len(node) + 4
        elif isinstance(node, Mapping):
            total += 2
            stack.append(_flatten_items(node))
        elif isinstance(node, (list, tuple)):
            total += 2
            stack.append(iter(node))
        else:
            total += 6  # a number, boolean, or null
    return min(total, limit)


def _flatten_items(mapping: Mapping[Any, Any]) -> Iterator[Any]:
    for key, value in mapping.items():
        yield key
        yield value


_END = object()
