"""run_gate.sh end to end, with a scripted stand-in for ``claude`` on PATH.

Everything but the agent is real: the script copies the shared fixture, sources
the recipe, runs the seeded tests before and after, judges the stream-json
events, checks the tests were left alone and prints the verdict. Each stand-in
plays one kind of Claude Code run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import gate_verdict

EXAMPLE = Path(__file__).resolve().parents[1]
FIXTURE = EXAMPLE.parent / "coding-agent-opencode" / "fixture"

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

FIX = f"cat > invoice_totals/amounts.py <<'PY'\n{FIXED_AMOUNTS}\nPY"


def _events(turns: int = 4, input_tokens: int = 4000, is_error: bool = False) -> str:
    lines = [
        {"type": "system", "subtype": "init", "tools": ["Bash", "Edit", "Read"]},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read", "input": {}}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Edit", "input": {}}]}},
        {
            "type": "result",
            "subtype": "success",
            "is_error": is_error,
            "num_turns": turns,
            "duration_ms": 1200,
            "result": "done",
            "usage": {"input_tokens": input_tokens, "cache_read_input_tokens": 0, "output_tokens": 20},
        },
    ]
    return "\n".join(json.dumps(line) for line in lines)


def _stand_in(body: str) -> str:
    return f"""#!/usr/bin/env bash
set -eu
printf '%s\\n' "$@" > "$GATE_TEST_RECORD/args"
env | grep -E '^(ANTHROPIC|CLAUDE_CODE|CLAUDE_CONFIG_DIR|API_TIMEOUT_MS)' | sort > "$GATE_TEST_RECORD/env" || true
{body}
"""


def _gate(
    tmp_path: Path, agent: str, recipe: str = "cloud", extra_env: dict[str, str] | None = None
) -> tuple[int, str, Path]:
    root = tmp_path / "examples"
    example = root / "coding-agent-claude-code"
    if not example.exists():
        example.mkdir(parents=True)
        for name in ("run_gate.sh", "gate_verdict.py"):
            shutil.copy(EXAMPLE / name, example)
        shutil.copytree(EXAMPLE / "recipes", example / "recipes")
        ignored = shutil.ignore_patterns(".venv", "uv.lock", "__pycache__", ".pytest_cache")
        shutil.copytree(FIXTURE, root / "coding-agent-opencode" / "fixture", ignore=ignored)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    claude = bin_dir / "claude"
    claude.write_text(_stand_in(agent))
    claude.chmod(0o755)
    record = tmp_path / "record"
    record.mkdir(exist_ok=True)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "SIE_", "GATE_", "PYTEST_"))
    }
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "SIE_BASE_URL": "https://sie.example",
            "SIE_API_KEY": "sk-test",
            "ANTHROPIC_API_KEY": "must-not-reach-claude",
            "ANTHROPIC_CUSTOM_HEADERS": "x-forward: elsewhere",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "GATE_TEST_RECORD": str(record),
            **(extra_env or {}),
        }
    )
    completed = subprocess.run(
        ["bash", str(example / "run_gate.sh"), recipe],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    return completed.returncode, completed.stdout + completed.stderr, record


def test_a_fixing_run_passes_with_the_recipe_and_bare_flags(tmp_path: Path) -> None:
    status, output, record = _gate(tmp_path, f"{FIX}\ncat <<'EOF'\n{_events()}\nEOF")
    assert status == 0, output
    assert "GATE PASS" in output
    args = (record / "args").read_text().splitlines()
    assert args[:2] == ["--bare", "-p"]
    assert "Make the tests pass." in "\n".join(args[2 : args.index("--settings")])
    assert args[args.index("--settings") + 1] == '{"apiKeyHelper": "printenv SIE_API_KEY"}'
    assert args[args.index("--permission-mode") + 1] == "acceptEdits"
    assert args[args.index("--allowedTools") + 1] == "Read,Edit,Bash(uv run pytest *)"
    assert args[args.index("--output-format") + 1] == "stream-json"
    assert args[args.index("--max-turns") + 1] == "50"
    env = dict(line.split("=", 1) for line in (record / "env").read_text().splitlines())
    assert env["ANTHROPIC_BASE_URL"] == "https://sie.example"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-test"
    assert env["ANTHROPIC_MODEL"] == "zai-org/GLM-5.3-Flash:relay"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "zai-org/GLM-5.3-Flash:relay"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32768"
    assert env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] == "1"
    assert env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_CUSTOM_HEADERS" not in env
    assert "CLAUDE_CODE_USE_BEDROCK" not in env
    assert "/run-output/gate-" in env["CLAUDE_CONFIG_DIR"]


def test_the_cluster_recipe_takes_the_model_and_window_from_the_environment(tmp_path: Path) -> None:
    extra = {"SIE_MODEL": "org/model:long", "SIE_CONTEXT_TOKENS": "65536", "SIE_OUTPUT_TOKENS": "8192"}
    status, output, record = _gate(tmp_path, f"{FIX}\ncat <<'EOF'\n{_events()}\nEOF", "cluster", extra)
    assert status == 0, output
    env = dict(line.split("=", 1) for line in (record / "env").read_text().splitlines())
    assert env["ANTHROPIC_MODEL"] == "org/model:long"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "org/model:long"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "8192"


@pytest.mark.parametrize("missing", ["SIE_MODEL", "SIE_CONTEXT_TOKENS", "SIE_OUTPUT_TOKENS"])
def test_the_cluster_recipe_needs_the_model_and_window(tmp_path: Path, missing: str) -> None:
    extra = {"SIE_MODEL": "m", "SIE_CONTEXT_TOKENS": "65536", "SIE_OUTPUT_TOKENS": "8192", missing: ""}
    status, output, _ = _gate(tmp_path, "exit 0", "cluster", extra)
    assert status == 2
    assert f"{missing} is not set" in output


def test_an_unknown_recipe_is_refused(tmp_path: Path) -> None:
    status, output, _ = _gate(tmp_path, "exit 0", "laptop")
    assert status == 2
    assert "unknown recipe 'laptop'" in output


def test_no_change_fails_on_the_seeded_tests(tmp_path: Path) -> None:
    status, output, _ = _gate(tmp_path, f"cat <<'EOF'\n{_events()}\nEOF")
    assert status == 1
    assert output.rstrip().splitlines()[-1].startswith("GATE FAIL (")


def test_a_failed_run_fails(tmp_path: Path) -> None:
    status, output, _ = _gate(tmp_path, "exit 3")
    assert status == 1
    assert "GATE FAIL: claude run exited 3" in output


@pytest.mark.parametrize(
    ("events", "problem"),
    [
        (_events(is_error=True), "the run ended in an error"),
        (_events(input_tokens=0), "read no input tokens"),
        (_events(turns=1), "fewer than 3"),
        ("", "no result event"),
    ],
)
def test_an_unclean_run_fails_before_pytest(tmp_path: Path, events: str, problem: str) -> None:
    status, output, _ = _gate(tmp_path, f"{FIX}\ncat <<'EOF'\n{events}\nEOF")
    assert status == 1
    assert "GATE FAIL: the Claude Code run did not finish cleanly" in output
    assert problem in output


def test_edited_tests_fail(tmp_path: Path) -> None:
    edit = "echo 'def test_nothing(): pass' > tests/test_amounts.py"
    status, output, _ = _gate(tmp_path, f"{FIX}\n{edit}\ncat <<'EOF'\n{_events()}\nEOF")
    assert status == 1
    assert "GATE FAIL: the tests or pyproject.toml were changed" in output


def test_a_run_past_its_deadline_is_stopped(tmp_path: Path) -> None:
    status, output, _ = _gate(tmp_path, "sleep 60", extra_env={"GATE_DEADLINE_SECONDS": "2"})
    assert status == 1
    assert "passed its 2s deadline" in output


def test_the_verdict_counts_invalid_request_retries() -> None:
    lines = _events().splitlines()
    retry = {"type": "system", "subtype": "api_retry", "error_status": 400, "error": "invalid_request_error"}
    passed, summary = gate_verdict.judge([json.dumps(retry), *lines, "not json"])
    assert not passed
    assert summary["api_retries"] == 1
    assert summary["unreadable_lines"] == 1
    assert summary["tool_uses"] == ["Read", "Edit"]
    assert "1 request(s) refused as invalid were retried" in summary["problems"]


def test_the_verdict_accepts_a_busy_retry() -> None:
    retry = {"type": "system", "subtype": "api_retry", "error_status": 503, "error": "overloaded_error"}
    passed, summary = gate_verdict.judge([json.dumps(retry), *_events().splitlines()])
    assert passed, summary
    assert summary["usage"]["input_tokens"] == 4000
