"""run_gate.sh end to end, with a scripted stand-in for ``npx`` on PATH.

Everything but the agent is real: the script copies the shared fixture, syncs
its environment, sources the recipe, which writes Codex's config and runs Codex
through ``npx``, runs the seeded tests before and after, judges the JSON
events, checks the tests were left alone and prints the verdict. Each stand-in
plays one kind of Codex run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

import gate_verdict

EXAMPLE = Path(__file__).resolve().parents[1]
FIXTURE = EXAMPLE.parent / "coding-agent-opencode" / "fixture"
PINNED = ["-y", "@openai/codex@0.167.0"]
METADATA_WARNING = (
    "Model metadata for `m` not found. Defaulting to fallback metadata; this can degrade performance and cause issues."
)

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


def _command(n: int) -> dict[str, Any]:
    item = {"id": f"item_{n}", "type": "command_execution", "command": "ls", "exit_code": 0, "status": "completed"}
    return {"type": "item.completed", "item": item}


def _events(commands: int = 2, input_tokens: int = 4000, ending: str = "completed") -> str:
    lines: list[dict[str, Any]] = [
        {"type": "thread.started", "thread_id": "thread"},
        {"type": "item.completed", "item": {"id": "item_0", "type": "error", "message": METADATA_WARNING}},
        {"type": "turn.started"},
        *[_command(n + 1) for n in range(commands)],
        {"type": "item.completed", "item": {"id": "item_9", "type": "agent_message", "text": "Done."}},
    ]
    if ending == "completed":
        usage = {"input_tokens": input_tokens, "cached_input_tokens": 0, "output_tokens": 40}
        lines.append({"type": "turn.completed", "usage": usage})
    elif ending == "failed":
        message = '{"error": {"message": "tools[4].type: \'namespace\' tools are not supported"}}'
        lines += [{"type": "error", "message": message}, {"type": "turn.failed", "error": {"message": message}}]
    elif ending == "error":
        lines += [{"type": "error", "message": "unexpected status 500"}, {"type": "turn.completed", "usage": {}}]
    return "\n".join(json.dumps(line) for line in lines)


def _stand_in(body: str) -> str:
    return f"""#!/usr/bin/env bash
set -eu
printf '%s\\n' "$@" > "$GATE_TEST_RECORD/args"
env | grep -E '^(CODEX_HOME|SIE_[A-Z_]+|UV_CACHE_DIR)=' | sort > "$GATE_TEST_RECORD/env" || true
cp "$CODEX_HOME/config.toml" "$GATE_TEST_RECORD/config.toml"
if [ -x .venv/bin/python ]; then echo synced > "$GATE_TEST_RECORD/environment"; fi
{body}
"""


def _env(tmp_path: Path, agent: str, extra_env: dict[str, str] | None, unset: tuple[str, ...]) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    npx = bin_dir / "npx"
    npx.write_text(_stand_in(agent))
    npx.chmod(0o755)
    record = tmp_path / "record"
    record.mkdir(exist_ok=True)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CODEX_", "SIE_", "GATE_", "PYTEST_", "OPENAI_"))
    }
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "SIE_BASE_URL": "https://sie.example",
            "SIE_API_KEY": "sk-test",
            "GATE_TEST_RECORD": str(record),
            **(extra_env or {}),
        }
    )
    for name in unset:
        env.pop(name, None)
    return env


def _gate(
    tmp_path: Path,
    agent: str,
    recipe: str = "cloud",
    extra_env: dict[str, str] | None = None,
    unset: tuple[str, ...] = (),
) -> tuple[int, str, Path]:
    root = tmp_path / "examples"
    example = root / "coding-agent-codex"
    if not example.exists():
        example.mkdir(parents=True)
        for name in ("run_gate.sh", "gate_verdict.py"):
            shutil.copy(EXAMPLE / name, example)
        shutil.copytree(EXAMPLE / "recipes", example / "recipes")
        ignored = shutil.ignore_patterns(".venv", "uv.lock", "__pycache__", ".pytest_cache")
        shutil.copytree(FIXTURE, root / "coding-agent-opencode" / "fixture", ignore=ignored)
    completed = subprocess.run(
        ["bash", str(example / "run_gate.sh"), recipe],
        capture_output=True,
        text=True,
        env=_env(tmp_path, agent, extra_env, unset),
        timeout=300,
        check=False,
    )
    return completed.returncode, completed.stdout + completed.stderr, tmp_path / "record"


def _recorded_env(record: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in (record / "env").read_text().splitlines())


def _recorded_config(record: Path) -> dict[str, Any]:
    return tomllib.loads((record / "config.toml").read_text())


def test_a_fixing_run_passes_with_the_pinned_codex_and_the_recipe(tmp_path: Path) -> None:
    status, output, record = _gate(tmp_path, f"{FIX}\ncat <<'EOF'\n{_events()}\nEOF")
    assert status == 0, output
    assert "GATE PASS" in output
    args = (record / "args").read_text().split("\n")
    assert args[:7] == [*PINNED, "exec", "--json", "--skip-git-repo-check", "--sandbox", "workspace-write"]
    assert "Make the tests pass." in "\n".join(args[7:])
    config = _recorded_config(record)
    assert config["model"] == "zai-org/GLM-5.3-Flash:relay"
    assert config["model_provider"] == "sie"
    assert config["model_context_window"] == 131072
    assert config["web_search"] == "disabled"
    assert config["features"] == {"multi_agent": False, "goals": False, "plugins": False}
    assert config["analytics"] == {"enabled": False}
    assert config["shell_environment_policy"] == {"exclude": ["SIE_API_KEY"]}
    assert config["model_providers"]["sie"] == {
        "name": "SIE",
        "base_url": "https://sie.example/v1",
        "env_key": "SIE_API_KEY",
        "wire_api": "responses",
        "stream_idle_timeout_ms": 1800000,
    }
    env = _recorded_env(record)
    run = Path(env["CODEX_HOME"]).parent
    assert env["CODEX_HOME"] == str(run / "codex-home")
    assert "/run-output/gate-" in env["CODEX_HOME"]
    assert env["UV_CACHE_DIR"] == str(run / "fixture" / ".uv-cache")
    assert env["SIE_API_KEY"] == "sk-test"
    assert (record / "environment").read_text().strip() == "synced"


def test_the_cluster_recipe_takes_the_model_and_window_from_the_environment(tmp_path: Path) -> None:
    extra = {"SIE_MODEL": "org/model:long", "SIE_CONTEXT_TOKENS": "65536", "SIE_BASE_URL": "http://gateway:8080"}
    status, output, record = _gate(tmp_path, f"{FIX}\ncat <<'EOF'\n{_events()}\nEOF", "cluster", extra)
    assert status == 0, output
    config = _recorded_config(record)
    assert config["model"] == "org/model:long"
    assert config["model_context_window"] == 65536
    assert config["model_providers"]["sie"]["base_url"] == "http://gateway:8080/v1"


@pytest.mark.parametrize("missing", ["SIE_MODEL", "SIE_CONTEXT_TOKENS"])
def test_the_cluster_recipe_needs_the_model_and_window(tmp_path: Path, missing: str) -> None:
    extra = {"SIE_MODEL": "m", "SIE_CONTEXT_TOKENS": "65536", missing: ""}
    status, output, _ = _gate(tmp_path, "exit 0", "cluster", extra)
    assert status == 2
    assert f"{missing} is not set" in output


def test_the_laptop_recipe_defaults_to_the_local_server_and_its_32k_profile(tmp_path: Path) -> None:
    unset = ("SIE_BASE_URL", "SIE_API_KEY")
    status, output, record = _gate(tmp_path, f"{FIX}\ncat <<'EOF'\n{_events()}\nEOF", "laptop", unset=unset)
    assert status == 0, output
    config = _recorded_config(record)
    assert config["model"] == "Qwen/Qwen3.5-4B:long-context"
    assert config["model_context_window"] == 32768
    assert config["model_providers"]["sie"]["base_url"] == "http://localhost:8080/v1"
    assert _recorded_env(record)["SIE_API_KEY"] == "local"


def test_the_laptop_recipe_takes_the_server_key_and_model_from_the_environment(tmp_path: Path) -> None:
    extra = {
        "SIE_BASE_URL": "http://127.0.0.1:9000",
        "SIE_API_KEY": "server-token",
        "SIE_MODEL": "sie-fake:chat",
        "SIE_CONTEXT_TOKENS": "131072",
    }
    status, output, record = _gate(tmp_path, f"{FIX}\ncat <<'EOF'\n{_events()}\nEOF", "laptop", extra)
    assert status == 0, output
    config = _recorded_config(record)
    assert config["model"] == "sie-fake:chat"
    assert config["model_context_window"] == 131072
    assert config["model_providers"]["sie"]["base_url"] == "http://127.0.0.1:9000/v1"
    assert _recorded_env(record)["SIE_API_KEY"] == "server-token"


def test_an_unknown_recipe_is_refused(tmp_path: Path) -> None:
    status, output, _ = _gate(tmp_path, "exit 0", "desktop")
    assert status == 2
    assert "unknown recipe 'desktop'; pick cloud, cluster or laptop" in output


def test_no_change_fails_on_the_seeded_tests(tmp_path: Path) -> None:
    status, output, _ = _gate(tmp_path, f"cat <<'EOF'\n{_events()}\nEOF")
    assert status == 1
    assert output.rstrip().splitlines()[-1].startswith("GATE FAIL (")


def test_a_failed_run_fails(tmp_path: Path) -> None:
    status, output, _ = _gate(tmp_path, "exit 3")
    assert status == 1
    assert "GATE FAIL: codex run exited 3" in output


@pytest.mark.parametrize(
    ("events", "problem"),
    [
        (_events(ending="failed"), "the turn failed: "),
        (_events(ending="error"), "1 error event(s): unexpected status 500"),
        (_events(input_tokens=0), "read no input tokens"),
        (_events(commands=0), "0 commands run, fewer than 1"),
        ("", "no turn.completed event"),
    ],
)
def test_an_unclean_run_fails_before_pytest(tmp_path: Path, events: str, problem: str) -> None:
    status, output, _ = _gate(tmp_path, f"{FIX}\ncat <<'EOF'\n{events}\nEOF")
    assert status == 1
    assert "GATE FAIL: the Codex run did not finish cleanly" in output
    assert problem in output


def test_edited_tests_fail(tmp_path: Path) -> None:
    edit = "echo 'def test_nothing(): pass' > tests/test_amounts.py"
    status, output, _ = _gate(tmp_path, f"{FIX}\n{edit}\ncat <<'EOF'\n{_events()}\nEOF")
    assert status == 1
    assert "GATE FAIL: the tests or pyproject.toml were changed" in output


def test_an_added_conftest_fails(tmp_path: Path) -> None:
    plant = "printf 'import pytest\\n' > conftest.py"
    status, output, _ = _gate(tmp_path, f"{FIX}\n{plant}\ncat <<'EOF'\n{_events()}\nEOF")
    assert status == 1
    assert "GATE FAIL: a conftest.py was added" in output


def test_a_run_past_its_deadline_is_stopped(tmp_path: Path) -> None:
    status, output, _ = _gate(tmp_path, "sleep 60", extra_env={"GATE_DEADLINE_SECONDS": "2"})
    assert status == 1
    assert "passed its 2s deadline" in output


def test_the_verdict_counts_reconnects_and_keeps_warnings() -> None:
    reconnect = {"type": "error", "message": "Reconnecting... 1/5 (stream disconnected before completion)"}
    passed, summary = gate_verdict.judge([json.dumps(reconnect), *_events().splitlines(), "not json"])
    assert passed, summary
    assert summary["reconnects"] == 1
    assert summary["unreadable_lines"] == 1
    assert summary["commands"] == 2
    assert summary["items"] == ["error", "command_execution", "command_execution", "agent_message"]
    assert summary["warnings"] == [METADATA_WARNING]
    assert summary["usage"] == {"input_tokens": 4000, "cached_input_tokens": 0, "output_tokens": 40}


def test_sourcing_a_recipe_writes_its_own_codex_home_and_runs_the_pinned_codex(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    env = _env(tmp_path, "", {"HOME": str(home)}, ())
    recipe = str(EXAMPLE / "recipes" / "cloud.sh")
    command = ["bash", "-c", 'source "$1" && codex --version', "bash", recipe]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    record = tmp_path / "record"
    assert (record / "args").read_text().split("\n")[:3] == [*PINNED, "--version"]
    assert _recorded_env(record)["CODEX_HOME"] == str(home / ".codex-sie")
    assert (home / ".codex-sie" / "config.toml").read_text() == (record / "config.toml").read_text()
    assert not (home / ".codex").exists()
