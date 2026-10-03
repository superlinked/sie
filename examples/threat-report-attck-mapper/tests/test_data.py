from __future__ import annotations

import json
import zipfile

import pytest

from threat_mapper.data import (
    _safe_extract,
    find_annoctr_report,
    load_gold_mentions,
    load_linking_cases,
    load_training_examples,
)


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


def test_training_examples_exclude_negative_spans_and_keep_source_context(tmp_path) -> None:
    path = tmp_path / "root" / "AnnoCTR" / "linking_mitre_only"
    path.mkdir(parents=True)
    rows = [
        {
            "mention": "downloaded another payload",
            "_context_left": "The loader ",
            "_context_right": ".",
            "label_link": "https://attack.mitre.org/techniques/T1105",
            "entity_class": "CI",
            "entity_type": "TECHNIQUE",
            "document": "train-report",
        },
        {
            "mention": "malware",
            "context_left": "The seller advertised ",
            "context_right": ".",
            "label_link": "No Annotation",
            "entity_class": "",
            "entity_type": "TECHNIQUE",
            "document": "train-report",
        },
    ]
    (path / "train.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    examples = load_training_examples(tmp_path)

    assert len(examples) == 1
    assert examples[0].technique_id == "T1105"
    assert examples[0].embedding_text == (
        "Span: downloaded another payload\nSentence: The loader downloaded another payload."
    )


def test_safe_extract_rejects_zip_slip(tmp_path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "bad")

    with pytest.raises(RuntimeError, match="escapes"):
        _safe_extract(archive, tmp_path / "output")


def test_find_annoctr_report_rejects_path_traversal(tmp_path) -> None:
    with pytest.raises(ValueError, match="safe AnnoCTR document"):
        find_annoctr_report(tmp_path, "test", "../report")


def test_gold_mentions_use_context_to_align_a_repeated_span(tmp_path) -> None:
    root = tmp_path / "root" / "AnnoCTR"
    linking = root / "linking_mitre_only"
    reports = root / "text" / "test"
    linking.mkdir(parents=True)
    reports.mkdir(parents=True)
    report = "The tool used a cookie before. Later it stole a cookie to log in."
    (reports / "report.txt").write_text(report, encoding="utf-8")
    row = {
        "mention": "cookie",
        "_context_left": "Later it stole a ",
        "_context_right": " to log in.",
        "label_link": "https://attack.mitre.org/techniques/T1539",
        "entity_class": "CI",
        "entity_type": "TECHNIQUE",
        "document": "report",
    }
    (linking / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    mentions = load_gold_mentions(tmp_path, "test")

    assert len(mentions) == 1
    assert mentions[0].source_start == report.rindex("cookie")
    assert report[mentions[0].source_start : mentions[0].source_end] == mentions[0].quote
