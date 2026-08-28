from __future__ import annotations

from tools.ci.classify_paths import GROUPS, all_groups, classify


def test_push_runs_every_group() -> None:
    assert classify(["README.md"], event="push") == all_groups()


def test_docs_only_pull_request_skips_expensive_groups() -> None:
    assert classify(["README.md", "docs/guide.md"], event="pull_request") == dict.fromkeys(GROUPS, False)


def test_each_owned_path_selects_its_group() -> None:
    cases = {
        "packages/sie_server/src/sie_server/api.py": "python",
        "packages/sie_ts_sdk/src/index.ts": "typescript",
        "packages/sie_gateway/src/main.rs": "rust",
        "tests/parity/run_batch_empty.json": "contracts",
        "deploy/helm/sie-cluster/values.yaml": "helm",
    }
    for path, group in cases.items():
        selected = classify([path], event="pull_request")
        assert selected[group]


def test_global_and_unknown_source_paths_fail_open() -> None:
    assert classify(["mise.toml"], event="pull_request") == all_groups()
    assert classify(["new-runtime/source.wasm"], event="pull_request") == all_groups()
    assert classify(["tools/ci/classify_paths.py"], event="pull_request") == all_groups()


def test_cross_language_paths_union_consumers() -> None:
    selected = classify(
        ["packages/sie_server/src/sie_server/ipc_types.py", "packages/sie_gateway/src/ipc_types.rs"],
        event="pull_request",
    )
    assert selected["python"]
    assert selected["rust"]
    assert selected["contracts"]
