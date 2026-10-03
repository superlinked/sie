"""The JSON number digit bound applied by the SGLang child's sitecustomize hook."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_COMPAT_DIR = Path(__file__).resolve().parents[2] / "src/sie_server/adapters/sglang/_compat"

# ``str(Grammar.from_json_schema(...))`` from XGrammar 0.2.1 for a schema with an
# unbounded integer, an unbounded number, and a digit-only string pattern.
_XGRAMMAR_EBNF = r"""basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
root_prop_2 ::= (("\"" root_prop_2_1 "\""))
root ::= (("{" "\"n\"" ": " basic_integer ", " "\"x\"" ": " basic_number ", " "\"s\"" ": " root_prop_2 "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
root_prop_2_1 ::= (([0-9] root_prop_2_1) | ([0-9]))
"""


def _load_sitecustomize() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sie_sglang_sitecustomize", _COMPAT_DIR / "sitecustomize.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bound_json_number_rules_bounds_only_the_shared_number_rules() -> None:
    sitecustomize = _load_sitecustomize()

    bounded = sitecustomize.bound_json_number_rules(_XGRAMMAR_EBNF, 19)

    rules = dict(line.split(" ::= ", 1) for line in bounded.splitlines())
    assert rules["basic_integer"] == '(("0") | ("-"? [1-9] [0-9]{0,18}))'
    assert rules["basic_number"] == (r'(("-"? ("0" | [1-9] [0-9]{0,18}) ("." [0-9]{1,19})? ([eE] [+\-]? [0-9]{1,3})?))')
    # A string pattern compiles to the same digit recursion as a number; only
    # the shared numeric rules may change.
    original = dict(line.split(" ::= ", 1) for line in _XGRAMMAR_EBNF.splitlines())
    for name in original.keys() - {"basic_integer", "basic_number"}:
        assert rules[name] == original[name], name


def test_bound_json_number_rules_bounds_the_injected_definition_rules() -> None:
    sitecustomize = _load_sitecustomize()
    ebnf = (
        "root ::= ((defs_SieBoundedJsonInteger defs_SieBoundedJsonNumber))\n"
        'defs_SieBoundedJsonInteger ::= (("0") | (defs_SieBoundedJsonInteger_1 [1-9] [0-9]*))\n'
        "defs_SieBoundedJsonNumber ::= ((defs_SieBoundedJsonNumber_1 defs_SieBoundedJsonNumber_7))\n"
    )

    rules = dict(line.split(" ::= ", 1) for line in sitecustomize.bound_json_number_rules(ebnf, 5).splitlines())

    assert rules["defs_SieBoundedJsonInteger"] == '(("0") | ("-"? [1-9] [0-9]{0,4}))'
    assert rules["defs_SieBoundedJsonNumber"].startswith('(("-"? ("0" | [1-9] [0-9]{0,4}) ("." [0-9]{1,5})?')


_INT = {"$ref": "#/$defs/SieBoundedJsonInteger"}
_NUM = {"$ref": "#/$defs/SieBoundedJsonNumber"}
_DEFS = {"SieBoundedJsonInteger": {"type": "integer"}, "SieBoundedJsonNumber": {"type": "number"}}


def test_bound_json_schema_numbers_references_every_unbounded_numeric_schema() -> None:
    sitecustomize = _load_sitecustomize()
    schema = {
        "type": "object",
        "properties": {
            "year": {"type": ["integer", "null"], "description": "four digits"},
            "price": {"type": "number", "description": "USD"},
            "tags": {"type": "array", "items": {"type": "number"}},
            "either": {"anyOf": [{"$ref": "#/$defs/Count"}, {"type": "string"}]},
        },
        "$defs": {"Count": {"type": "integer", "title": "count"}},
    }
    original = json.loads(json.dumps(schema))

    bounded = sitecustomize.bound_json_schema_numbers(schema, 19)

    assert schema == original
    assert bounded["properties"]["year"] == {"anyOf": [_INT, {"type": "null", "description": "four digits"}]}
    assert bounded["properties"]["price"] == _NUM
    assert bounded["properties"]["tags"] == {"type": "array", "items": _NUM}
    assert bounded["properties"]["either"] == schema["properties"]["either"]
    assert bounded["$defs"] == {"Count": _INT} | _DEFS


@pytest.mark.parametrize(
    ("numeric", "expected"),
    [
        ({"type": "integer", "minimum": 0}, {"type": "integer", "minimum": 0, "maximum": 2**63 - 1}),
        (
            {"type": "integer", "exclusiveMaximum": 10},
            {"type": "integer", "exclusiveMaximum": 10, "minimum": -(2**63 - 1)},
        ),
        ({"type": "number", "minimum": 0}, {"type": "number", "minimum": 0, "maximum": 1e15}),
        ({"type": "number", "exclusiveMinimum": 0}, {"type": "number", "exclusiveMinimum": 0, "maximum": 1e15}),
        ({"type": "number", "maximum": 100}, {"type": "number", "maximum": 100}),
        ({"type": "integer", "minimum": 1, "maximum": 5}, {"type": "integer", "minimum": 1, "maximum": 5}),
        ({"type": "number", "enum": [1.5, 2.5]}, {"type": "number", "enum": [1.5, 2.5]}),
        ({"type": "integer", "const": 3}, {"type": "integer", "const": 3}),
        ({"type": "string", "pattern": "^[0-9]+$"}, {"type": "string", "pattern": "^[0-9]+$"}),
    ],
)
def test_bound_json_schema_numbers_handles_bounded_and_finite_schemas(numeric: dict, expected: dict) -> None:
    sitecustomize = _load_sitecustomize()

    bounded = sitecustomize.bound_json_schema_numbers({"type": "object", "properties": {"v": numeric}}, 19)

    assert bounded == {"type": "object", "properties": {"v": expected}}


def test_bound_json_schema_numbers_integer_bound_follows_the_digit_limit() -> None:
    sitecustomize = _load_sitecustomize()

    bounded = sitecustomize.bound_json_schema_numbers({"type": "integer", "minimum": 0}, 4)

    assert bounded == {"type": "integer", "minimum": 0, "maximum": 9999}


def test_bound_json_schema_numbers_rewrites_a_numeric_root() -> None:
    sitecustomize = _load_sitecustomize()

    assert sitecustomize.bound_json_schema_numbers({"type": "number"}, 19) == _NUM | {
        "$defs": {"SieBoundedJsonNumber": {"type": "number"}}
    }
    assert sitecustomize.bound_json_schema_numbers(True, 19) is True


def test_bound_json_number_rules_leaves_a_grammar_without_numbers_alone() -> None:
    sitecustomize = _load_sitecustomize()
    ebnf = 'root ::= (("{" "\\"a\\"" ": " basic_string "}"))\nbasic_string ::= (("\\"" [a-z]* "\\""))\n'

    assert sitecustomize.bound_json_number_rules(ebnf, 19) == ebnf


def _fake_xgrammar(root: Path) -> None:
    package = root / "xgrammar"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from .compiler import GrammarCompiler\n", encoding="utf-8")
    (package / "grammar.py").write_text(
        f"""EBNF = {_XGRAMMAR_EBNF!r}
CALLS = []


class Grammar:
    @classmethod
    def from_json_schema(cls, schema, **kwargs):
        CALLS.append(("from_json_schema", schema, kwargs))
        return cls()

    def __str__(self):
        return EBNF
""",
        encoding="utf-8",
    )
    (package / "compiler.py").write_text(
        """from .grammar import CALLS, Grammar


class GrammarCompiler:
    def compile_json_schema(
        self, schema, *, any_whitespace=True, indent=None, separators=None, strict_mode=True, max_whitespace_cnt=None
    ):
        CALLS.append(("compile_json_schema", schema, {"any_whitespace": any_whitespace, "max_whitespace_cnt": max_whitespace_cnt}))
        return "json"

    def compile_grammar(self, ebnf):
        CALLS.append(("compile_grammar", ebnf, {}))
        return "ebnf"
""",
        encoding="utf-8",
    )


def _run_child(tmp_path: Path, script: str, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    fake_root = tmp_path / "fake"
    _fake_xgrammar(fake_root)
    env = {k: v for k, v in os.environ.items() if k != "SIE_SGLANG_JSON_NUMBER_MAX_DIGITS"}
    env["PYTHONPATH"] = os.pathsep.join((str(_COMPAT_DIR), str(fake_root)))
    env.update(env_overrides)
    return subprocess.run(  # noqa: S603 - executes the fixed local interpreter
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


_CHILD_SCRIPT = """import json
from xgrammar import GrammarCompiler
from xgrammar.grammar import CALLS

compiler = GrammarCompiler()
results = [
    compiler.compile_json_schema(schema='{"type": "object"}', any_whitespace=False),
    compiler.compile_json_schema('{"type": "object"}', any_whitespace=True, max_whitespace_cnt=4),
]
print(json.dumps({
    "patched": bool(getattr(GrammarCompiler, "_sie_json_number_max_digits", False)),
    "results": results,
    "calls": CALLS,
}))
"""


def _child_report(completed: subprocess.CompletedProcess[str]) -> dict:
    assert completed.returncode == 0, completed.stderr
    assert "Error in sitecustomize" not in completed.stderr
    return json.loads(completed.stdout)


def test_json_number_hook_is_inert_without_configuration(tmp_path: Path) -> None:
    report = _child_report(_run_child(tmp_path, _CHILD_SCRIPT))

    assert report["patched"] is False
    assert report["results"] == ["json", "json"]
    assert [call[0] for call in report["calls"]] == ["compile_json_schema", "compile_json_schema"]


def test_json_number_hook_compiles_the_digit_bounded_grammar(tmp_path: Path) -> None:
    report = _child_report(_run_child(tmp_path, _CHILD_SCRIPT, SIE_SGLANG_JSON_NUMBER_MAX_DIGITS="19"))

    assert report["patched"] is True
    assert report["results"] == ["ebnf", "ebnf"]
    # The schema and every formatting option reach XGrammar's own converter.
    conversions = [call for call in report["calls"] if call[0] == "from_json_schema"]
    assert [call[1] for call in conversions] == ['{"type": "object"}', '{"type": "object"}']
    assert [call[2] for call in conversions] == [
        {"any_whitespace": False, "indent": None, "separators": None, "strict_mode": True, "max_whitespace_cnt": None},
        {"any_whitespace": True, "indent": None, "separators": None, "strict_mode": True, "max_whitespace_cnt": 4},
    ]
    compiled = [call[1] for call in report["calls"] if call[0] == "compile_grammar"]
    assert compiled == [sitecustomize_bound(19), sitecustomize_bound(19)]


def sitecustomize_bound(max_digits: int) -> str:
    return _load_sitecustomize().bound_json_number_rules(_XGRAMMAR_EBNF, max_digits)


@pytest.mark.parametrize("value", ["0", "-3", "many", "1.5"])
def test_json_number_hook_rejects_invalid_bounds(tmp_path: Path, value: str) -> None:
    completed = _run_child(tmp_path, "import xgrammar\n", SIE_SGLANG_JSON_NUMBER_MAX_DIGITS=value)

    assert "Error in sitecustomize" in completed.stderr
    assert "SIE_SGLANG_JSON_NUMBER_MAX_DIGITS must be a positive integer" in completed.stderr
