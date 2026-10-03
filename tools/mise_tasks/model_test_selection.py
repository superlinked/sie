"""Select handwritten model tests by their exact model-ID literals."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MODEL_TEST_PATH = Path("packages/sie_server/tests/test_all_models.py")
MODEL_CALLS = {
    "_get_adapter",
    "_assert_dense",
    "_assert_dense_image",
    "_assert_sparse",
    "_assert_multivector",
    "_assert_multivector_image",
    "_assert_score",
    "_assert_extract",
}


def model_id_literals(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    model_ids: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else None
        first_argument = node.args[0]
        if name in MODEL_CALLS and isinstance(first_argument, ast.Constant) and isinstance(first_argument.value, str):
            model_ids.add(first_argument.value)
    return model_ids


def select_node_ids(model_id: str, source: str, path: str = str(MODEL_TEST_PATH)) -> list[str]:
    tree = ast.parse(source, filename=path)
    selected: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
            continue
        model_ids = model_id_literals(node)
        if len(model_ids) > 1:
            rendered = ", ".join(sorted(repr(value) for value in model_ids))
            raise ValueError(f"{path}::{node.name} maps to multiple model IDs: {rendered}")
        if model_ids == {model_id}:
            selected.append(f"{path}::{node.name}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("--test-file", type=Path, default=MODEL_TEST_PATH)
    args = parser.parse_args()
    try:
        selected = select_node_ids(args.model_id, args.test_file.read_text(), str(args.test_file))
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not selected:
        print(f"ERROR: no handwritten model tests map exactly to {args.model_id!r}", file=sys.stderr)
        return 1
    print("\n".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
