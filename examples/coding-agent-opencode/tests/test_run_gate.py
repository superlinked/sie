"""run_gate.sh end to end, with a scripted stand-in for ``opencode`` on PATH.

Everything but the agent is real: the script copies the fixture, runs the
seeded tests before and after, checks the tests were left alone and prints
the verdict. Each stand-in plays one kind of agent run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1]

FIXED_AMOUNTS = """
from decimal import Decimal


def parse_amount(text: str) -> Decimal:
    cleaned = text.strip()
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    value = Decimal(cleaned.strip("()").replace("$", "").replace(",", ""))
    return -value if negative else value


def total(cells: list[str]) -> Decimal:
    return sum((parse_amount(cell) for cell in cells), Decimal(0))
"""

DESELECT_FAILING_TESTS = (
    'import os; os.environ["PYTEST_ADDOPTS"] = '
    '"--deselect tests/test_amounts.py::test_credit_in_parentheses_is_negative '
    '--deselect tests/test_amounts.py::test_total_nets_credits"'
)


FIX = f"cat > invoice_totals/amounts.py <<'PY'\n{FIXED_AMOUNTS}\nPY"
NO_CHANGE = 'echo \'{"type":"text","text":"nothing to do"}\''


def _executable(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\nset -eu\n{body}\n")
    path.chmod(0o755)


def _gate(tmp_path: Path, agent: str, extra_env: dict[str, str] | None = None) -> tuple[int, list[str]]:
    root = tmp_path / "example"
    if not root.exists():
        root.mkdir()
        shutil.copy(EXAMPLE / "run_gate.sh", root)
        ignored = shutil.ignore_patterns(".venv", "uv.lock", "__pycache__", ".pytest_cache")
        shutil.copytree(EXAMPLE / "fixture", root / "fixture", ignore=ignored)
        shutil.copytree(EXAMPLE / "recipes", root / "recipes")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _executable(bin_dir / "opencode", agent)
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env.update(extra_env or {})
    run = subprocess.run(
        ["bash", str(root / "run_gate.sh"), "laptop"], check=False, capture_output=True, text=True, env=env, timeout=600
    )
    return run.returncode, run.stdout.strip().splitlines()


def test_a_real_fix_passes(tmp_path: Path) -> None:
    code, lines = _gate(tmp_path, FIX)
    assert code == 0, lines
    assert lines[-1].startswith("GATE PASS")


def test_runs_started_in_the_same_second_judge_their_own_copies(tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    _executable(tmp_path / "bin" / "date", "echo 20260101T000000Z")
    first_code, first = _gate(tmp_path, FIX)
    second_code, second = _gate(tmp_path, NO_CHANGE)
    assert first_code == 0, first
    assert second_code == 1
    assert second[-1].startswith("GATE FAIL (")
    assert first[-1].rsplit("(", 1)[1] != second[-1].rsplit("(", 1)[1]


def test_an_untouched_fixture_fails(tmp_path: Path) -> None:
    code, lines = _gate(tmp_path, NO_CHANGE)
    assert code == 1
    assert lines[-1].startswith("GATE FAIL (")


def test_a_failed_opencode_run_still_prints_a_verdict(tmp_path: Path) -> None:
    code, lines = _gate(tmp_path, "exit 3")
    assert code == 1
    assert lines[-1].startswith("GATE FAIL: opencode run exited 3")


def test_changed_tests_fail_before_pytest_is_asked(tmp_path: Path) -> None:
    code, lines = _gate(tmp_path, "printf 'def test_nothing():\\n    assert True\\n' > tests/test_amounts.py")
    assert code == 1
    assert lines[-1].startswith("GATE FAIL: the tests or pyproject.toml were changed")


def test_an_environment_planted_in_the_worked_copy_is_not_used(tmp_path: Path) -> None:
    agent = (
        "[ -d .venv ] || uv venv --quiet --python 3.12 .venv\n"
        "site=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')\n"
        f"echo '{DESELECT_FAILING_TESTS}' > \"$site/zz_gate.pth\""
    )
    code, lines = _gate(tmp_path, agent)
    assert code == 1
    assert lines[-1].startswith("GATE FAIL (")
    assert any("2 failed" in line for line in lines)


@pytest.mark.parametrize("variable", ["PYTEST_ADDOPTS", "PYTEST_PLUGINS"])
def test_the_callers_pytest_environment_cannot_deselect_the_seeded_tests(tmp_path: Path, variable: str) -> None:
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "gate_deselect.py").write_text(
        "def pytest_collection_modifyitems(config, items):\n"
        "    items[:] = [item for item in items if 'credit' not in item.name]\n"
    )
    caller = {
        "PYTEST_ADDOPTS": {"PYTEST_ADDOPTS": "-k 'not credit'"},
        "PYTEST_PLUGINS": {"PYTEST_PLUGINS": "gate_deselect", "PYTHONPATH": str(plugins)},
    }[variable]
    code, lines = _gate(tmp_path, NO_CHANGE, caller)
    assert code == 1, lines
    assert lines[-1].startswith("GATE FAIL (")
    assert any("2 failed" in line for line in lines)


def test_the_build_agent_gets_the_step_budget(tmp_path: Path) -> None:
    record = tmp_path / "opencode-call.txt"
    agent = f'printf \'%s\\n\' "$@" "$OPENCODE_CONFIG_CONTENT" > {record}\n{FIX}'
    code, lines = _gate(tmp_path, agent, {"GATE_STEPS": "7"})
    assert code == 0, lines
    *args, config = record.read_text().splitlines()
    assert args[args.index("--agent") + 1] == "build"
    assert json.loads(config) == {"agent": {"build": {"steps": 7}}}


@pytest.mark.parametrize("ignores_term", [False, True])
def test_a_run_past_its_deadline_is_stopped_and_fails(tmp_path: Path, ignores_term: bool) -> None:
    child = tmp_path / "agent-child.pid"
    prelude = "trap '' TERM\n" if ignores_term else ""
    agent = f"{prelude}sleep 120 &\necho $! > {child}\nwait"
    started = time.monotonic()
    code, lines = _gate(tmp_path, agent, {"GATE_DEADLINE_SECONDS": "2"})
    assert code == 1, lines
    assert lines[-1].startswith("GATE FAIL: opencode run passed its 2s deadline")
    assert time.monotonic() - started < 60
    pid = int(child.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"the agent's child process {pid} outlived the deadline")


@pytest.mark.parametrize(
    ("recipe", "needs_sie_variables"), [("gpu", False), ("openrouter", False), ("cluster", True), ("cloud", True)]
)
def test_only_a_recipe_that_reads_the_sie_variables_requires_them(
    tmp_path: Path, recipe: str, needs_sie_variables: bool
) -> None:
    root = tmp_path / "example"
    root.mkdir()
    shutil.copy(EXAMPLE / "run_gate.sh", root)
    ignored = shutil.ignore_patterns(".venv", "uv.lock", "__pycache__", ".pytest_cache")
    shutil.copytree(EXAMPLE / "fixture", root / "fixture", ignore=ignored)
    shutil.copytree(EXAMPLE / "recipes", root / "recipes")
    record = tmp_path / "opencode-config.txt"
    (tmp_path / "bin").mkdir()
    _executable(tmp_path / "bin" / "opencode", f'echo "$OPENCODE_CONFIG" > {record}\n{FIX}')
    unset = ("VIRTUAL_ENV", "SIE_BASE_URL", "SIE_API_KEY")
    env = {key: value for key, value in os.environ.items() if key not in unset}
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{env['PATH']}"
    run = subprocess.run(
        ["bash", str(root / "run_gate.sh"), recipe], check=False, capture_output=True, text=True, env=env, timeout=600
    )
    if needs_sie_variables:
        assert run.returncode == 2
        assert run.stderr.strip() == "SIE_BASE_URL is not set"
        assert not record.exists()
    else:
        assert run.returncode == 0, run.stdout
        assert run.stdout.strip().splitlines()[-1].startswith("GATE PASS")
        assert Path(record.read_text().strip()).parts[-3:] == ("recipes", recipe, "opencode.json")
