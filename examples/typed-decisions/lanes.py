"""How a lane answers a record: one model alone, or a fast model that escalates what it is unsure of.

Standard library only. score.py and tune.py read recordings into per-case
answers (after each backend's decision rules) and hand them here.

A lane in tuning.json:

    {"backend": "gliclass-large-v1"}
        every question answered by that backend

    {"backend": "gliformer-large", "escalate_to": "gliclass-large-v1",
     "cutoffs": {"weakness": 0.6, ...}}
        the first backend answers; where its top option's probability for a
        question is below that question's cut-off, the answer comes from the
        second backend instead. A question with no cut-off is never escalated;
        a question the first backend is not asked (questions.ASKED) always is.

Every lane also answers `severity`, computed from its own answers to
weakness, attack_vector and remote_unauthenticated with the CVSS v3.1 formula
and the tables in cvss.py (cvss.compose_severity).
"""

from __future__ import annotations

from typing import Any

from cvss import compose_severity

SEVERITY = "severity"
COMPONENTS = ("weakness", "attack_vector", "remote_unauthenticated")


def severity_answer(answers: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """The composed severity answer, or None when a component has no full distribution.

    A backend that returned a whole CVSS vector (the LLM) already carries its
    computed severity, which is used as it is.
    """
    if SEVERITY in answers:
        return answers[SEVERITY]
    parts = [answers.get(qid) for qid in COMPONENTS]
    if any(part is None or part["distribution"] is None for part in parts):
        return None
    distribution = compose_severity(*(part["distribution"] for part in parts))
    top = max(distribution, key=distribution.get)
    return {"distribution": distribution, "top": top, "top_p": distribution[top], "type": "score"}


def lane_answers(
    lane: dict[str, Any], by_backend: dict[str, dict[str, dict[str, dict[str, Any]]]]
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, bool]]]:
    """Per case, the lane's answer to every question plus `severity`, and which questions escalated."""
    first = by_backend[lane["backend"]]
    second = by_backend.get(lane.get("escalate_to", ""), {})
    cutoffs = lane.get("cutoffs", {})
    answers: dict[str, dict[str, dict[str, Any]]] = {}
    escalated: dict[str, dict[str, bool]] = {}
    for case, own in first.items():
        answers[case], escalated[case] = {}, {}
        qids = list(own) + [qid for qid in second.get(case, {}) if qid not in own and qid != SEVERITY]
        for qid in qids:
            answer = own.get(qid)
            cutoff = cutoffs.get(qid)
            # A question the first backend is not asked always goes to the second.
            up = answer is None or (cutoff is not None and answer["top_p"] < cutoff)
            if up and qid not in second.get(case, {}):
                raise SystemExit(f"{case}: {qid} escalates to {lane.get('escalate_to')}, which has no answer for it")
            answers[case][qid] = second[case][qid] if up else answer
            escalated[case][qid] = up
        composed = severity_answer(answers[case])
        if composed is not None:
            answers[case][SEVERITY] = composed
    return answers, escalated
