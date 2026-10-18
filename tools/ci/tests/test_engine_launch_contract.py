from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from tools.ci import engine_launch_contract, engine_launch_probe

SNAPSHOT_0_5_20 = engine_launch_contract.SNAPSHOT_DIR / "sglang-0.5.20.txt"


def report(*, version="0.5.20", flags=None, routes=None):
    flags = sorted(flags if flags is not None else engine_launch_contract.read_snapshot(SNAPSHOT_0_5_20))
    return {
        "schema": 1,
        "bundle": "sglang-cu130",
        "engine": "sglang",
        "engine_version": version,
        "distributions": {"sglang": version},
        "flags": flags,
        "flags_sha256": engine_launch_probe.flag_set_sha256(flags),
        "routes": routes if routes is not None else [{"route": "m", "argv": ["--port", "1"], "error": None}],
    }


def engine_parser(*options: str) -> argparse.ArgumentParser:
    parser = engine_launch_probe._EngineParser()
    for option in options:
        parser.add_argument(option, type=float if option == "--mem-fraction-static" else str)
    return parser


def test_probe_runs_offline_inside_the_image_without_a_gpu():
    command = engine_launch_contract.probe_command("sie-server:pr", "sglang")

    assert command[:8] == ["docker", "run", "--rm", "-i", "--pull", "never", "--network", "none"]
    assert "HF_HUB_OFFLINE=1" in command
    assert "TRANSFORMERS_OFFLINE=1" in command
    assert command[-6:] == ["--entrypoint", "python", "sie-server:pr", "-", "--bundle", "sglang"]
    assert not {"push", "login", "--gpus"} & set(command)


def test_a_stalled_probe_is_a_contract_failure(monkeypatch):
    def stalled(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(engine_launch_contract.subprocess, "run", stalled)

    with pytest.raises(RuntimeError, match="did not finish within 5s"):
        engine_launch_contract.run_probe("sie-server:pr", "sglang", timeout_s=5)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--mamba-scheduler-strategy", "extra_buffer"], "not declared by the engine: --mamba-scheduler-strategy"),
        (["--cuda-graph-max-bs", "8"], "not declared by the engine: --cuda-graph-max-bs"),
        (["--disable-piecewise-cuda-graph"], "not declared by the engine: --disable-piecewise-cuda-graph"),
        (["--tp", "2"], "not declared by the engine: --tp"),
    ],
)
def test_a_flag_the_engine_does_not_declare_exactly_fails(argv, expected):
    parser = engine_parser(
        "--mamba-radix-cache-strategy",
        "--cuda-graph-max-bs-decode",
        "--cuda-graph-max-bs-prefill",
        "--disable-prefill-cuda-graph",
        "--tp-size",
        "--tensor-parallel-size",
    )

    assert engine_launch_probe.check_argv(parser, argv) == expected


def test_the_engine_parser_rejects_values_it_cannot_parse():
    parser = engine_parser("--mem-fraction-static")

    assert engine_launch_probe.check_argv(parser, ["--mem-fraction-static", "0.85"]) is None
    assert "invalid float value" in engine_launch_probe.check_argv(parser, ["--mem-fraction-static", "most"])


def test_a_clean_report_matching_its_snapshot_passes():
    assert engine_launch_contract.evaluate(report()) == []


def test_a_bundle_without_an_engine_parser_has_nothing_to_check():
    assert engine_launch_contract.evaluate({"engine": None, "routes": []}) == []


def test_every_failing_route_is_reported():
    routes = [
        {"route": "a", "argv": [], "error": "not declared by the engine: --mamba-scheduler-strategy"},
        {"route": "b", "argv": [], "error": None},
        {"route": "c", "argv": None, "error": "render failed: boom"},
    ]

    assert engine_launch_contract.evaluate(report(routes=routes)) == [
        "a: not declared by the engine: --mamba-scheduler-strategy",
        "c: render failed: boom",
    ]


def test_an_image_whose_engine_drifted_from_its_snapshot_fails():
    flags = [*engine_launch_contract.read_snapshot(SNAPSHOT_0_5_20), "--brand-new-flag"]
    flags.remove("--mamba-radix-cache-strategy")

    (problem,) = engine_launch_contract.evaluate(report(flags=flags))

    assert "removed ['--mamba-radix-cache-strategy']" in problem
    assert "added ['--brand-new-flag']" in problem


def test_an_engine_bump_without_a_snapshot_prints_the_flag_diff(tmp_path):
    (tmp_path / "sglang-0.5.10.post1.txt").write_text("--a\n--old\n")
    (tmp_path / "sglang-0.5.20.txt").write_text("--a\n--old-but-renamed\n")
    (tmp_path / "sglang-0.6.0.txt").write_text("--newer\n")

    (problem,) = engine_launch_contract.evaluate(report(version="0.5.21", flags=["--a", "--new"]), tmp_path)

    assert "commit sglang-0.5.21.txt" in problem
    assert "Since sglang-0.5.20.txt: removed ['--old-but-renamed'], added ['--new']" in problem


def test_the_first_snapshot_of_an_engine_is_requested(tmp_path):
    (problem,) = engine_launch_contract.evaluate(report(version="0.5.20", flags=["--a"]), tmp_path)

    assert problem == "no snapshot of sglang 0.5.20's declared flags; commit sglang-0.5.20.txt"


@pytest.mark.parametrize(
    ("older", "newer"),
    [("0.5.10.post1", "0.5.20"), ("0.5.9", "0.5.10"), ("0.5.10", "0.5.10.post1"), ("0.5.20", "0.6.0")],
)
def test_snapshot_versions_order_numerically(older, newer):
    assert engine_launch_contract._version_key(older) < engine_launch_contract._version_key(newer)


def test_check_writes_the_engine_snapshot_for_a_bump_and_fails(tmp_path, capsys):
    bumped = report(version="0.5.21", flags=["--a", "--b"])

    with pytest.raises(ValueError, match="commit sglang-0.5.21.txt"):
        engine_launch_contract.check(
            "sie-server:pr",
            "sglang-cu130",
            report_dir=tmp_path / "out",
            probe=lambda image, bundle: bumped,
        )

    assert (tmp_path / "out/sglang-0.5.21.txt").read_text() == "--a\n--b\n"
    assert json.loads((tmp_path / "out/sglang-cu130.json").read_text()) == bumped
    assert capsys.readouterr().out == ("::group::conformance/engine-flags/sglang-0.5.21.txt\n--a\n--b\n::endgroup::\n")


def test_check_passes_a_clean_image(capsys):
    engine_launch_contract.check("sie-server:pr", "sglang-cu130", probe=lambda image, bundle: report())

    assert "1 routed profiles parsed by sglang 0.5.20" in capsys.readouterr().out


@pytest.mark.parametrize("path", sorted(engine_launch_contract.SNAPSHOT_DIR.glob("*.txt")), ids=lambda path: path.name)
def test_committed_snapshots_are_sorted_unique_option_strings(path: Path):
    flags = path.read_text().splitlines()

    assert flags == sorted(set(flags))
    assert all(flag.startswith("-") and flag.strip() == flag for flag in flags)
