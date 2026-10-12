from __future__ import annotations

from pathlib import Path

import pytest
import yaml

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping key repeated in the same mapping.

    PyYAML keeps the last value of a repeated key without complaint, so a
    duplicated block can silently hide an edit. Stricter consumers of these
    files reject the duplicate outright.
    """


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen: set[object] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


@pytest.mark.parametrize("path", sorted(MODELS_DIR.glob("*.yaml")), ids=lambda path: path.name)
def test_model_yaml_has_no_duplicate_keys(path: Path) -> None:
    yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)  # noqa: S506
