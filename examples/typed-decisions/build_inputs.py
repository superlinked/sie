#!/usr/bin/env python3
"""Build the two case files from their public sources, by a rule fixed before any model call.

    uv sync --group build
    uv run python build_inputs.py            # writes data/inputs/<set>/cases.json

Readers normally never run this: once the evidence is published, fetch.py
downloads the same files at a pinned revision. It is here so the selection can
be audited and rebuilt.

Two sets.

vulnerability-triage
    CVE records from the NVD API 2.0, with the NVD analyst's own assessment as
    the gold answer: the primary CWE decides `weakness`, the CVSS v3.1 vector
    decides `attack_vector`, `user_interaction` and `privileges_required`, and
    the CVSS base severity decides `severity`. `needs_account`,
    `remote_unauthenticated` and `impact` are read off the same vector
    (cvss.py). The model sees the English description and nothing else.

    Three windows, by publication date (UTC):

        2023-10   2023-10-01 to 2023-10-31   dev, 20 per weakness class
        2023-11a  2023-11-01 to 2023-11-15   dev, 8 per weakness class
        2023-12   2023-12-01 to 2023-12-31   test, 20 per weakness class

    Everything that is tuned (phrasing, yes-or-no cut-offs, choice prior
    corrections) is chosen on the 224 dev records only. The December records
    are the held-out test slice: they were added after the October slice had
    been looked at, and no setting is chosen on them.

    A record is eligible when it is not rejected; has exactly one CVSS v3.1
    metric of type Primary from nvd@nist.gov; has exactly one CWE from
    nvd@nist.gov and that CWE is in CWE_CLASS below; has exactly one English
    description of 12 to 150 words, ASCII only. Eligible records are ordered by
    sha256("sie-typed-decisions-nvd-v1:" + CVE id) and the first N per weakness
    class are taken from each window. No record is chosen or dropped by hand.

    Each case also carries the vendor and product of every vulnerable CPE NVD
    lists for it, which the GLiFormer multi-task run (multitask.py) scores
    against.

    NVD records keep changing after publication. The rebuilt file records each
    record's `lastModified` and a digest of the fields used, so drift between a
    rebuild and the published inputs shows up as a different digest rather than
    as a different score.

workflows
    The `test` split of the typed-decisions benchmark on Hugging Face, pinned to
    a commit, four workflows. Each case carries its own state and five questions
    in the source's exact shape, and its gold is the benchmark's: the mean of
    three samples from a teacher model, reduced to the argmax label. A score here
    measures agreement with that teacher, not correctness.

    All 400 test cases are taken, 100 per workflow, ordered by
    sha256("sie-typed-decisions-workflows-v1:" + case id).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cvss import derived_gold
from questions import (
    VULNERABILITY_QUESTIONS,
    VULNERABILITY_TRIAGE,
    WORKFLOWS,
    from_source_question,
    keys,
    typed_questions,
)

NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
# window id -> (split, first publication date, last publication date, records per weakness class)
NVD_WINDOWS = {
    "2023-10": ("dev", "2023-10-01T00:00:00.000", "2023-10-31T23:59:59.999", 20),
    "2023-11a": ("dev", "2023-11-01T00:00:00.000", "2023-11-15T23:59:59.999", 8),
    "2023-12": ("test", "2023-12-01T00:00:00.000", "2023-12-31T23:59:59.999", 20),
}
NVD_SALT = "sie-typed-decisions-nvd-v1:"
NVD_PAGE_SIZE = 2000
# NVD asks unauthenticated clients for no more than 5 requests in 30 seconds.
NVD_PAUSE_S = 6.5

CWE_CLASS = {
    "CWE-79": "cross_site_scripting",
    "CWE-89": "sql_injection",
    "CWE-119": "memory_corruption",
    "CWE-120": "memory_corruption",
    "CWE-121": "memory_corruption",
    "CWE-122": "memory_corruption",
    "CWE-125": "memory_corruption",
    "CWE-416": "memory_corruption",
    "CWE-787": "memory_corruption",
    "CWE-77": "command_injection",
    "CWE-78": "command_injection",
    "CWE-94": "command_injection",
    "CWE-22": "path_traversal",
    "CWE-352": "request_forgery",
    "CWE-285": "access_control",
    "CWE-287": "access_control",
    "CWE-306": "access_control",
    "CWE-639": "access_control",
    "CWE-862": "access_control",
    "CWE-863": "access_control",
    "CWE-434": "file_upload",
}
ATTACK_VECTOR = {"NETWORK": "network", "ADJACENT_NETWORK": "adjacent", "LOCAL": "local", "PHYSICAL": "physical"}
USER_INTERACTION = {"NONE": "false", "REQUIRED": "true"}
PRIVILEGES = {"NONE": "0", "LOW": "1", "HIGH": "2"}
SEVERITY = {"LOW": "0", "MEDIUM": "1", "HIGH": "2", "CRITICAL": "3"}
MIN_WORDS, MAX_WORDS = 12, 150

WORKFLOW_DATASET = "LocalLLaMA/typed-decisions"
WORKFLOW_REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
WORKFLOW_FILE = "all/test-00000-of-00001.parquet"
WORKFLOW_SALT = "sie-typed-decisions-workflows-v1:"
WORKFLOW_PER_WORKFLOW = 100


def http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": "*/*", "User-Agent": "sie-typed-decisions-example"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
        return response.read()


def sha256(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def nvd_window(start: str, end: str) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    urls: list[str] = []
    index = 0
    while True:
        query = urllib.parse.urlencode(
            {"pubStartDate": start, "pubEndDate": end, "resultsPerPage": NVD_PAGE_SIZE, "startIndex": index}
        )
        url = f"{NVD_API}?{query}"
        urls.append(url)
        page = json.loads(http_get(url))
        records.extend(entry["cve"] for entry in page["vulnerabilities"])
        index += page["resultsPerPage"]
        if index >= page["totalResults"]:
            return records, urls
        time.sleep(NVD_PAUSE_S)


def nvd_case(record: dict[str, Any]) -> dict[str, Any] | None:
    """One eligible record as a case, or None when the rule excludes it."""
    if record.get("vulnStatus") == "Rejected":
        return None
    metrics = [
        metric
        for metric in record.get("metrics", {}).get("cvssMetricV31", [])
        if metric.get("type") == "Primary" and metric.get("source") == "nvd@nist.gov"
    ]
    if len(metrics) != 1:
        return None
    cwes = sorted(
        {
            description["value"]
            for weakness in record.get("weaknesses", [])
            if weakness.get("source") == "nvd@nist.gov"
            for description in weakness["description"]
        }
    )
    if len(cwes) != 1 or cwes[0] not in CWE_CLASS:
        return None
    english = [entry["value"] for entry in record.get("descriptions", []) if entry.get("lang") == "en"]
    if len(english) != 1:
        return None
    text = re.sub(r"\s+", " ", english[0]).strip()
    if not text.isascii() or not MIN_WORDS <= len(text.split()) <= MAX_WORDS:
        return None
    cvss = metrics[0]["cvssData"]
    used = {"description": text, "cwe": cwes[0], "cvss": cvss}
    cpes = sorted(
        {
            tuple(match["criteria"].split(":")[3:5])
            for node in (configuration.get("nodes", []) for configuration in record.get("configurations", []))
            for part in node
            for match in part.get("cpeMatch", [])
            if match.get("vulnerable") and match.get("criteria", "").startswith("cpe:2.3:")
        }
    )
    return {
        "slug": record["id"],
        "state": text,
        "gold": {
            "weakness": CWE_CLASS[cwes[0]],
            "attack_vector": ATTACK_VECTOR[cvss["attackVector"]],
            "user_interaction": USER_INTERACTION[cvss["userInteraction"]],
            "privileges_required": PRIVILEGES[cvss["privilegesRequired"]],
            "severity": SEVERITY[cvss["baseSeverity"]],
            **derived_gold(cvss["vectorString"]),
        },
        "source": {
            "url": f"https://nvd.nist.gov/vuln/detail/{record['id']}",
            "published": record["published"],
            "last_modified": record["lastModified"],
            "cwe": cwes[0],
            "cvss_vector": cvss["vectorString"],
            "cvss_base_score": cvss["baseScore"],
            "used_fields_sha256": sha256(canonical(used)),
            "cpe_vendor_product": [list(pair) for pair in cpes],
        },
    }


def build_vulnerability_triage() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    queries: dict[str, list[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    for window, (split, start, end, per_class) in NVD_WINDOWS.items():
        records, urls = nvd_window(start, end)
        queries[window] = urls
        eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            case = nvd_case(record)
            if case is not None:
                eligible[case["gold"]["weakness"]].append(case)
        counts[window] = {"retrieved": len(records), **{key: len(value) for key, value in sorted(eligible.items())}}
        for weakness in keys(VULNERABILITY_QUESTIONS["weakness"]):
            ordered = sorted(eligible[weakness], key=lambda case: sha256(NVD_SALT + case["slug"]))
            if len(ordered) < per_class:
                raise SystemExit(f"{window}: only {len(ordered)} eligible {weakness} records")
            for case in ordered[:per_class]:
                cases.append({**case, "split": split, "window": window})
        time.sleep(NVD_PAUSE_S)
    return {
        "set": VULNERABILITY_TRIAGE,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {
            "name": "National Vulnerability Database (NVD), CVE API 2.0",
            "queries": queries,
            "eligible_counts": counts,
            "terms": (
                "This product uses data from the NVD API but is not endorsed or certified by the NVD. "
                "CVE descriptions are used under the CVE Program Terms of Use."
            ),
        },
        "rule": {
            "salt": NVD_SALT,
            "windows": {
                window: {"split": v[0], "from": v[1], "to": v[2], "per_class": v[3]}
                for window, v in NVD_WINDOWS.items()
            },
            "words": [MIN_WORDS, MAX_WORDS],
            "cwe_class": CWE_CLASS,
            "gold_maps": {
                "attack_vector": ATTACK_VECTOR,
                "user_interaction": USER_INTERACTION,
                "privileges_required": PRIVILEGES,
                "severity": SEVERITY,
            },
        },
        "questions": VULNERABILITY_QUESTIONS,
        "cases": cases,
    }


def build_workflows() -> dict[str, Any]:
    # Optional dependency: only this builder reads parquet.
    import pyarrow.parquet as pq  # noqa: PLC0415

    url = f"https://huggingface.co/datasets/{WORKFLOW_DATASET}/resolve/{WORKFLOW_REVISION}/{WORKFLOW_FILE}"
    payload = http_get(url)
    rows = pq.read_table(io.BytesIO(payload)).to_pylist()

    workflow_questions: dict[str, dict[str, Any]] = {}
    by_workflow: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_questions = json.loads(row["questions"])
        converted = {qid: from_source_question(question) for qid, question in source_questions.items()}
        rendered, _ = typed_questions(converted, "described")
        if rendered != source_questions:
            raise SystemExit(f"{row['id']}: questions do not round-trip through the converter")
        known = workflow_questions.setdefault(row["workflow"], source_questions)
        if known != source_questions:
            raise SystemExit(f"{row['id']}: questions differ from the rest of {row['workflow']}")
        gold = json.loads(row["gold"])
        by_workflow[row["workflow"]].append(
            {
                "slug": row["id"],
                "split": "test",
                "workflow": row["workflow"],
                "state": json.loads(row["state"]),
                "gold": {qid: str(answer["label"]) for qid, answer in gold.items()},
                "gold_probabilities": {qid: answer["probabilities"] for qid, answer in gold.items()},
            }
        )

    cases: list[dict[str, Any]] = []
    for workflow in sorted(by_workflow):
        ordered = sorted(by_workflow[workflow], key=lambda case: sha256(WORKFLOW_SALT + case["slug"]))
        cases.extend(ordered[:WORKFLOW_PER_WORKFLOW])
    return {
        "set": WORKFLOWS,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {
            "name": "typed-decisions benchmark, test split",
            "dataset": WORKFLOW_DATASET,
            "revision": WORKFLOW_REVISION,
            "file": WORKFLOW_FILE,
            "file_sha256": sha256(payload),
            "license": "Apache-2.0",
            "gold": "mean of three teacher-model samples; the argmax label is scored",
        },
        "rule": {"salt": WORKFLOW_SALT, "per_workflow": WORKFLOW_PER_WORKFLOW},
        "questions": {workflow: workflow_questions[workflow] for workflow in sorted(workflow_questions)},
        "cases": cases,
    }


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}: {len(payload['cases'])} cases")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="output directory (default: data)")
    parser.add_argument("--set", choices=[VULNERABILITY_TRIAGE, WORKFLOWS], action="append", help="build one set")
    args = parser.parse_args()
    wanted = args.set or [VULNERABILITY_TRIAGE, WORKFLOWS]
    data_dir = Path(args.data)
    if VULNERABILITY_TRIAGE in wanted:
        write(data_dir / f"inputs/{VULNERABILITY_TRIAGE}/cases.json", build_vulnerability_triage())
    if WORKFLOWS in wanted:
        write(data_dir / f"inputs/{WORKFLOWS}/cases.json", build_workflows())
    return 0


if __name__ == "__main__":
    sys.exit(main())
