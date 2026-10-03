from __future__ import annotations

import json

from threat_mapper.catalog import catalog_by_id, load_annoctr_catalog, load_catalog


def test_load_catalog_keeps_active_techniques_and_cleans_description(tmp_path) -> None:
    bundle = {
        "objects": [
            {
                "type": "attack-pattern",
                "id": "attack-pattern--one",
                "name": "Adversary-in-the-Middle",
                "description": "Adversaries use <code>proxy</code>. (Citation: Example) See [cookies](https://example.test).",
                "modified": "2026-08-05T00:00:00Z",
                "external_references": [
                    {
                        "source_name": "mitre-attack",
                        "external_id": "T1557",
                        "url": "https://attack.mitre.org/techniques/T1557",
                    }
                ],
                "kill_chain_phases": [{"phase_name": "credential-access"}],
                "x_mitre_platforms": ["Windows"],
                "x_mitre_is_subtechnique": False,
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--old",
                "name": "Old",
                "revoked": True,
                "external_references": [{"source_name": "mitre-attack", "external_id": "T1000"}],
            },
        ]
    }
    path = tmp_path / "attack.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")

    techniques = load_catalog(path)

    assert len(techniques) == 1
    assert techniques[0].technique_id == "T1557"
    assert techniques[0].description == "Adversaries use proxy. See cookies."
    assert "credential access" in techniques[0].candidate_text
    assert catalog_by_id(techniques)["T1557"].name == "Adversary-in-the-Middle"


def test_load_annoctr_catalog_preserves_its_historical_label_space(tmp_path) -> None:
    path = tmp_path / "mitre_entity.jsonl"
    path.write_text(
        json.dumps(
            {
                "idx": "https://attack.mitre.org/techniques/T1550/004",
                "title": "Web Session Cookie",
                "text": "Use a stolen cookie. (Citation: old)",
                "entity_type": "techniques",
                "is_subtechnique": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    techniques = load_annoctr_catalog(path)

    assert techniques[0].technique_id == "T1550.004"
    assert techniques[0].description == "Use a stolen cookie."
