from __future__ import annotations

import json
import zipfile

import pytest

from threat_mapper.data import _safe_extract, find_annoctr_report, load_linking_cases


def test_linking_rows_with_same_span_become_one_multilabel_case(tmp_path) -> None:
    path = tmp_path / "root" / "AnnoCTR" / "linking_mitre_only"
    path.mkdir(parents=True)
    rows = [
        {
            "mention": "stolen session cookie",
            "_context_left": "used a ",
            "_context_right": " to sign in",
            "sentence_left": "before",
            "sentence_right": "after",
            "label_link": "https://attack.mitre.org/techniques/T1539",
            "entity_class": "CE",
            "entity_type": "TECHNIQUE",
            "document": "report",
        },
        {
            "mention": "stolen session cookie",
            "_context_left": "used a ",
            "_context_right": " to sign in",
            "sentence_left": "before",
            "sentence_right": "after",
            "label_link": "https://attack.mitre.org/techniques/T1550/004",
            "entity_class": "CI",
            "entity_type": "TECHNIQUE",
            "document": "report",
        },
    ]
    (path / "test.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n\n",
        encoding="utf-8",
    )

    cases = load_linking_cases(tmp_path, "test")

    assert len(cases) == 1
    assert cases[0].evidence == "used a stolen session cookie to sign in"
    assert cases[0].gold_ids == ("T1539", "T1550.004")
    assert cases[0].annotation_classes == ("CE", "CI")


def test_safe_extract_rejects_zip_slip(tmp_path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "bad")

    with pytest.raises(RuntimeError, match="escapes"):
        _safe_extract(archive, tmp_path / "output")


def test_find_annoctr_report_rejects_path_traversal(tmp_path) -> None:
    with pytest.raises(ValueError, match="safe AnnoCTR document"):
        find_annoctr_report(tmp_path, "test", "../report")
