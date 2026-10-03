"""CVSS v3.1 base scores, and the NVD-derived gold for the questions built on them.

Standard library only. The formula follows the CVSS v3.1 specification from
FIRST (section 7.1, and Appendix A for Roundup). It reproduces NVD's recorded
base score for every record in the case file; tests/test_example.py checks a
set of published vectors.

Used two ways:
    build_inputs.py derives the gold answers of the questions that are read off
                    a record's CVSS vector (needs_account, remote_unauthenticated,
                    impact)
    score.py        composes a severity from a model's answers to the component
                    questions, with the same formula
"""

from __future__ import annotations

import math
from itertools import product

AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
AC = {"L": 0.77, "H": 0.44}
UI = {"N": 0.85, "R": 0.62}
CIA = {"H": 0.56, "L": 0.22, "N": 0.0}

# The impact patterns the `impact` question offers: (scope, confidentiality,
# integrity, availability). Together they are the exact vector of 200 of the
# 224 dev records; `impact_pattern` maps the rest to the nearest one.
IMPACT_PATTERNS = {
    "full_compromise": ("U", "H", "H", "H"),
    "data_exposure": ("U", "H", "N", "N"),
    "browser_script": ("C", "L", "L", "N"),
    "limited_change": ("U", "N", "L", "N"),
    "denial_of_service": ("U", "N", "N", "H"),
    "tampering": ("U", "N", "H", "N"),
}
SEVERITY_LEVELS = ("0", "1", "2", "3")  # low, medium, high, critical


def parse(vector: str) -> dict[str, str]:
    """`CVSS:3.1/AV:N/AC:L/...` -> {"AV": "N", "AC": "L", ...}."""
    return dict(part.split(":") for part in vector.split("/")[1:])


def roundup(value: float) -> float:
    """CVSS v3.1 Appendix A: the smallest number, to one decimal, not less than `value`."""
    integer = round(value * 100000)
    if integer % 10000 == 0:
        return integer / 100000.0
    return (math.floor(integer / 10000) + 1) / 10.0


def privileges_weight(privileges: str, scope: str) -> float:
    if privileges == "N":
        return 0.85
    if privileges == "L":
        return 0.68 if scope == "C" else 0.62
    return 0.5 if scope == "C" else 0.27


def impact_subscore(confidentiality: str, integrity: str, availability: str) -> float:
    return 1 - (1 - CIA[confidentiality]) * (1 - CIA[integrity]) * (1 - CIA[availability])


def base_score(metrics: dict[str, str]) -> float:
    iss = impact_subscore(metrics["C"], metrics["I"], metrics["A"])
    if metrics["S"] == "U":
        impact = 6.42 * iss
    else:
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
    exploitability = (
        8.22
        * AV[metrics["AV"]]
        * AC[metrics["AC"]]
        * privileges_weight(metrics["PR"], metrics["S"])
        * UI[metrics["UI"]]
    )
    if impact <= 0:
        return 0.0
    if metrics["S"] == "U":
        return roundup(min(impact + exploitability, 10))
    return roundup(min(1.08 * (impact + exploitability), 10))


def severity(score: float) -> str | None:
    """NVD's qualitative rating, as the level keys the severity rubric uses; None for a score of 0."""
    if score == 0:
        return None
    if score < 4.0:
        return "0"
    if score < 7.0:
        return "1"
    if score < 9.0:
        return "2"
    return "3"


def impact_pattern(metrics: dict[str, str]) -> str:
    """The impact pattern a vector has, or the nearest one.

    Nearest: the smallest difference in impact subscore, plus 0.5 when the
    scope differs, then the fewest differing C/I/A letters, then the order of
    IMPACT_PATTERNS.
    """
    record = (metrics["S"], metrics["C"], metrics["I"], metrics["A"])
    for key, pattern in IMPACT_PATTERNS.items():
        if pattern == record:
            return key
    record_iss = impact_subscore(*record[1:])

    def distance(item: tuple[int, tuple[str, tuple[str, ...]]]) -> tuple[float, int, int]:
        order, (_, pattern) = item
        gap = abs(impact_subscore(*pattern[1:]) - record_iss) + (0.5 if pattern[0] != record[0] else 0.0)
        letters = sum(a != b for a, b in zip(pattern[1:], record[1:]))
        return (round(gap, 9), letters, order)

    _, (key, _) = min(enumerate(IMPACT_PATTERNS.items()), key=distance)
    return key


def derived_gold(vector: str) -> dict[str, str]:
    """Gold answers read off a CVSS v3.1 vector."""
    metrics = parse(vector)
    return {
        "needs_account": "true" if metrics["PR"] != "N" else "false",
        "remote_unauthenticated": "true" if metrics["AV"] == "N" and metrics["PR"] == "N" else "false",
        "impact": impact_pattern(metrics),
    }


# Composition. A severity is computed from three asked answers: the weakness
# class, the attack vector, and whether the record can be exploited remotely
# without a login. The other base metrics come from fixed tables, fitted once on
# the 224 dev records' NVD vectors and frozen before any test record was sent:
#   user interaction  required for the two classes that need a victim to act
#                     (cross-site scripting 28 of 28 dev records, request
#                     forgery 26 of 28), none for every other class
#   impact            the pattern each weakness class most often has on dev
#                     (path traversal is a 12-12 tie, broken by IMPACT_PATTERNS
#                     order)
#   privileges        none when the record is exploitable remotely without a
#                     login, otherwise low (low or high on 100 of the 104 dev
#                     records that are not)
#   attack complexity low (221 of 224 dev records)
ATTACK_VECTOR = {"network": "N", "adjacent": "A", "local": "L", "physical": "P"}
USER_INTERACTION_BY_WEAKNESS = {
    "cross_site_scripting": "R",
    "sql_injection": "N",
    "memory_corruption": "N",
    "command_injection": "N",
    "path_traversal": "N",
    "request_forgery": "R",
    "access_control": "N",
    "file_upload": "N",
}
IMPACT_BY_WEAKNESS = {
    "cross_site_scripting": "browser_script",
    "sql_injection": "full_compromise",
    "memory_corruption": "full_compromise",
    "command_injection": "full_compromise",
    "path_traversal": "full_compromise",
    "request_forgery": "full_compromise",
    "access_control": "data_exposure",
    "file_upload": "full_compromise",
}


def compose_severity(
    weakness: dict[str, float], attack_vector: dict[str, float], remote_unauthenticated: dict[str, float]
) -> dict[str, float]:
    """A distribution over severity levels from the three answers' distributions.

    Every combination of answers is scored with the CVSS formula, the other
    metrics taken from the tables above, and weighted by the product of the
    answers' probabilities, as if the answers were independent.
    """
    levels = dict.fromkeys(SEVERITY_LEVELS, 0.0)
    for (w, p_w), (av, p_av), (remote, p_r) in product(
        weakness.items(), attack_vector.items(), remote_unauthenticated.items()
    ):
        weight = p_w * p_av * p_r
        if weight == 0:
            continue
        scope, confidentiality, integrity, availability = IMPACT_PATTERNS[IMPACT_BY_WEAKNESS[w]]
        metrics = {
            "AV": ATTACK_VECTOR[av],
            "AC": "L",
            "PR": "N" if remote == "true" else "L",
            "UI": USER_INTERACTION_BY_WEAKNESS[w],
            "S": scope,
            "C": confidentiality,
            "I": integrity,
            "A": availability,
        }
        levels[severity(base_score(metrics))] += weight
    total = sum(levels.values())
    return {level: value / total for level, value in levels.items()}
