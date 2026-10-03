"""The typed questions each record is asked, and how each backend receives them.

Standard library only. Imported by build_inputs.py, run.py, score.py and the
tests, so the question a model saw and the question a score is computed for are
the same object.

A question is one of three types:

    choice   pick one option from a list; the answer is a distribution over options
    noul     a yes-or-no statement; the answer is the probability that it holds
    score    an ordered rubric; the answer is a distribution over levels, low to high

Every option has a stable `key` (what the gold label and the scorer use), a
short `name` and, where the source has one, a `description`. A backend never
sees the key unless the key is also what it is asked to return.

Three phrasings are defined for the vulnerability-triage set:

    short       options by name only
    described   options by name and description
    concrete    options by name, except where an option has a `concrete` name
                that spells out what happens (memory corruption, and both
                sides of the remote-without-login question)

Which phrasing each question gets is chosen per backend on the dev slice only
(tune.py), and the chosen mix is the `tuned` variant (tuning.py). Anywhere a
phrasing is taken, a dict of question id to phrasing is taken too. The
workflow set keeps its source's own questions, which always carry
descriptions, so it has one phrasing.

Two kinds of backend receive the questions:

    typed      a decision model that takes a state and a dict of typed questions
               in one call and answers all of them (`typed_questions`)
    labels     a zero-shot label scorer that takes a text and a list of label
               strings, one question per call (`label_options`)
"""

from __future__ import annotations

import json
from typing import Any

CHOICE = "choice"
NOUL = "noul"
SCORE = "score"
TYPES = (CHOICE, NOUL, SCORE)

SHORT = "short"
DESCRIBED = "described"
CONCRETE = "concrete"
VARIANTS = (SHORT, DESCRIBED, CONCRETE)

VULNERABILITY_TRIAGE = "vulnerability-triage"
WORKFLOWS = "workflows"
SETS = (VULNERABILITY_TRIAGE, WORKFLOWS)

# The vulnerability-triage question set. Gold answers come from the NVD
# analyst's CWE and CVSS v3.1 assessment of each record; build_inputs.py holds
# the mapping from NVD values to these keys.
VULNERABILITY_QUESTIONS: dict[str, dict[str, Any]] = {
    "weakness": {
        "type": CHOICE,
        "instructions": "Which kind of weakness does this vulnerability report describe?",
        "options": [
            {
                "key": "cross_site_scripting",
                "name": "cross-site scripting",
                "description": "attacker-supplied script runs in another user's browser",
            },
            {
                "key": "sql_injection",
                "name": "SQL injection",
                "description": "attacker input changes a database query",
            },
            {
                "key": "memory_corruption",
                "name": "memory corruption",
                "concrete": "buffer overflow or out-of-bounds memory access",
                "description": "buffer overflow, out-of-bounds read or write, or use after free",
            },
            {
                "key": "command_injection",
                "name": "command or code injection",
                "description": "attacker input runs as an operating system command or as code",
            },
            {
                "key": "path_traversal",
                "name": "path traversal",
                "description": "attacker-controlled file paths reach files outside the intended directory",
            },
            {
                "key": "request_forgery",
                "name": "cross-site request forgery",
                "description": "a logged-in user's browser is made to send a request the user did not intend",
            },
            {
                "key": "access_control",
                "name": "missing authentication or authorization",
                "description": "an action or resource lacks a proper login or permission check",
            },
            {
                "key": "file_upload",
                "name": "unrestricted file upload",
                "description": "dangerous file types, such as scripts, can be uploaded",
            },
        ],
    },
    "attack_vector": {
        "type": CHOICE,
        "instructions": "From where can an attacker exploit this vulnerability?",
        "options": [
            {
                "key": "network",
                "name": "remotely over the network",
                "description": "from anywhere the vulnerable service or page is reachable, such as the internet",
            },
            {
                "key": "adjacent",
                "name": "from an adjacent network",
                "description": "only from the same local network segment, or within Bluetooth or Wi-Fi range",
            },
            {
                "key": "local",
                "name": "with local access",
                "description": "only from the machine itself, as a logged-in user or through a file a user opens",
            },
            {
                "key": "physical",
                "name": "with physical access",
                "description": "only by physically touching or connecting to the device",
            },
        ],
    },
    "remote_unauthenticated": {
        "type": NOUL,
        "instructions": "Anyone who can reach the system over the network can exploit this without logging in.",
        "options": [
            {
                "key": "false",
                "name": "needs local access or a login",
                "concrete": "only a local user or a logged-in user can do it",
                "description": "the attacker needs access to the machine itself, or an account",
            },
            {
                "key": "true",
                "name": "exploitable remotely without a login",
                "concrete": "anyone on the network can do it without an account",
                "description": "an unauthenticated attacker can exploit it over the network",
            },
        ],
    },
}


# The questions a backend's call carries, where that is not all of them. A
# one-call model reads every question in the same prompt, and on dev GLiFormer
# confused "exploitable remotely without a login" with "remotely over the
# network": asked all three, its attack-vector answer collapsed to network. It
# is asked the two it answers, and the Fast lane leaves the third unanswered.
ASKED: dict[str, dict[str, tuple[str, ...]]] = {
    VULNERABILITY_TRIAGE: {"gliformer-large": ("weakness", "attack_vector")},
}


def asked(set_name: str, backend: str, questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The subset of `questions` a backend is asked on one set."""
    keep = ASKED.get(set_name, {}).get(backend)
    return questions if keep is None else {qid: q for qid, q in questions.items() if qid in keep}


# The severity rubric is not asked of any model; its level keys are the ones a
# computed severity uses. lanes.severity_answer composes that from a backend's
# answers to weakness, attack_vector and remote_unauthenticated with the CVSS
# v3.1 formula and the fixed tables in cvss.py; the LLM's comes from the full
# vector it returns. The gold is NVD's base severity.
SEVERITY_RUBRIC: dict[str, Any] = {
    "type": SCORE,
    "instructions": "How severe is this vulnerability?",
    "options": [
        {"key": "0", "name": "low", "description": "CVSS base score 0.1 to 3.9"},
        {"key": "1", "name": "medium", "description": "CVSS base score 4.0 to 6.9"},
        {"key": "2", "name": "high", "description": "CVSS base score 7.0 to 8.9"},
        {"key": "3", "name": "critical", "description": "CVSS base score 9.0 to 10.0"},
    ],
}


def humanize(key: str) -> str:
    """`escalate_to_human` -> `escalate to human`."""
    return key.replace("_", " ").strip()


def level_name(description: str) -> str:
    """The short name of a rubric level: the text before its first colon or semicolon."""
    for separator in (":", ";"):
        if separator in description:
            return description.split(separator, 1)[0].strip()
    return description.strip()


def from_source_question(question: dict[str, Any]) -> dict[str, Any]:
    """Convert one question in the workflow dataset's own shape to the shape used here.

    The source writes choice criteria as `{key: description}`, score criteria as
    an ordered list of level descriptions and noul criteria as an optional
    `{"false": ..., "true": ...}`. Nothing is reworded. A choice option's name is
    its source key, so `typed_questions(..., DESCRIBED)` gives back the source
    question exactly; a level's name is the text before its first colon or
    semicolon.
    """
    qtype = question["type"]
    criteria = question.get("criteria")
    if qtype == CHOICE:
        options = [{"key": key, "name": key, "description": text} for key, text in criteria.items()]
    elif qtype == SCORE:
        options = [
            {"key": str(index), "name": level_name(text), "description": text} for index, text in enumerate(criteria)
        ]
    elif qtype == NOUL:
        criteria = criteria or {}
        options = [
            {"key": "false", "name": "no", "description": criteria.get("false")},
            {"key": "true", "name": "yes", "description": criteria.get("true")},
        ]
    else:
        raise ValueError(f"unknown question type {qtype!r}")
    return {"type": qtype, "instructions": question["instructions"], "options": options}


def keys(question: dict[str, Any]) -> list[str]:
    return [option["key"] for option in question["options"]]


def _level_text(option: dict[str, Any]) -> str:
    """A rubric level with its description, without repeating a name the description already starts with."""
    description = option["description"] or ""
    if not description:
        return option["name"]
    if option["name"] == level_name(description):
        return description
    return f"{option['name']}: {description}"


Phrasing = str | dict[str, str]


def phrasing_of(variant: Phrasing, qid: str) -> str:
    """The phrasing one question gets: the variant itself, or its entry when the variant maps questions."""
    phrasing = variant[qid] if isinstance(variant, dict) else variant
    if phrasing not in VARIANTS:
        raise ValueError(f"unknown phrasing {phrasing!r}")
    return phrasing


def option_name(option: dict[str, Any], phrasing: str) -> str:
    """An option's name in one phrasing: `concrete` swaps in the concrete name where there is one."""
    if phrasing == CONCRETE:
        return option.get("concrete") or option["name"]
    return option["name"]


def typed_questions(
    questions: dict[str, dict[str, Any]], variant: Phrasing
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    """Render a question set for a typed decision model.

    Returns the question dict to send and, per question, the map from each
    option label the model can return to the option key the scorer uses.

    `described` sends `{name: description}` choice criteria, score levels with
    their descriptions and `true`/`false` noul descriptions where there are any.
    `short` sends choice options as a list of names, score levels by name and
    noul questions as the bare statement. `concrete` is `short` with concrete
    names, and sends a noul question's concrete names as its descriptions.
    Score and noul answers come back by level index and as P(true), so their
    map is the identity.
    """
    rendered: dict[str, dict[str, Any]] = {}
    label_maps: dict[str, dict[str, str]] = {}
    for qid, question in questions.items():
        phrasing = phrasing_of(variant, qid)
        qtype = question["type"]
        options = question["options"]
        body: dict[str, Any] = {"type": qtype, "instructions": question["instructions"]}
        if qtype == CHOICE:
            names = [option_name(option, phrasing) for option in options]
            if phrasing == DESCRIBED:
                body["criteria"] = {name: option["description"] for name, option in zip(names, options)}
            else:
                body["criteria"] = names
            label_maps[qid] = {name: option["key"] for name, option in zip(names, options)}
        elif qtype == SCORE:
            if phrasing == DESCRIBED:
                body["criteria"] = [_level_text(option) for option in options]
            else:
                body["criteria"] = [option_name(option, phrasing) for option in options]
            label_maps[qid] = {str(index): option["key"] for index, option in enumerate(options)}
        else:
            if phrasing == DESCRIBED:
                criteria = {option["key"]: option["description"] for option in options if option["description"]}
            elif phrasing == CONCRETE:
                criteria = {option["key"]: option["concrete"] for option in options if option.get("concrete")}
            else:
                criteria = {}
            if criteria:
                body["criteria"] = criteria
            label_maps[qid] = {"false": "false", "true": "true"}
        if len(label_maps[qid]) != len(options):
            raise ValueError(f"{qid}: two options share a name")
        rendered[qid] = body
    return rendered, label_maps


def negation(statement: str) -> str:
    """The false-side label for a yes-or-no statement that ships no description of its own."""
    statement = statement.strip()
    return f"Not the case: {statement[0].lower()}{statement[1:]}"


def label_options(question: dict[str, Any], variant: str) -> list[tuple[str, str]]:
    """Render one question for a label scorer, as `(key, label string)` pairs.

    A label scorer has no instruction channel, so each label has to carry its
    own meaning. `short` sends option names; `described` sends
    `"name: description"`, with underscores in a name read as spaces;
    `concrete` sends the concrete name where an option has one.

    A yes-or-no question whose options carry real names (the vulnerability set)
    sends those names under `short` and `concrete` and the descriptions under
    `described`. One whose options are only "yes" and "no" (the workflow set)
    always sends two full sentences, the true side first, because a bare "yes"
    or "no" says nothing without the question: the source description where
    there is one, otherwise the statement and its negation.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    qtype = question["type"]
    options = question["options"]
    if qtype == NOUL:
        by_key = {option["key"]: option for option in options}
        named = by_key["true"]["name"] not in ("yes", "no")
        if named and variant != DESCRIBED:
            pairs = [("true", option_name(by_key["true"], variant)), ("false", option_name(by_key["false"], variant))]
        else:
            true_text = by_key["true"]["description"] or question["instructions"]
            false_text = by_key["false"]["description"] or negation(question["instructions"])
            pairs = [("true", true_text), ("false", false_text)]
    elif qtype == SCORE:
        if variant == DESCRIBED:
            pairs = [(option["key"], humanize(_level_text(option))) for option in options]
        else:
            pairs = [(option["key"], humanize(option_name(option, variant))) for option in options]
    elif variant == DESCRIBED:
        pairs = [(option["key"], f"{humanize(option['name'])}: {option['description']}") for option in options]
    else:
        pairs = [(option["key"], humanize(option_name(option, variant))) for option in options]
    labels = [label for _, label in pairs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"two options render to the same label: {labels}")
    return pairs


def state_text(state: Any) -> str:
    """The text a label scorer receives: the state itself, or its JSON when it is structured.

    Serialised exactly the way the typed decision model serialises a dict state
    (`json.dumps(..., ensure_ascii=False)`), so both kinds of backend read the
    same characters.
    """
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)
