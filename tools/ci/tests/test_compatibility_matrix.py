from __future__ import annotations

import copy
import json
import re
from functools import cache

import pytest

from tools.ci import compatibility_matrix as matrix

DOCS = matrix.ROOT / "docs/compatibility"


def floors() -> dict:
    return json.loads(matrix.DRIVER_FLOORS.read_text())


@cache
def generated() -> dict:
    return matrix.build(matrix.ROOT, "0.0.0", floors=floors())


def test_cuda_runtimes_come_from_each_lanes_final_stage():
    runtimes = matrix.cuda_runtimes(matrix.ROOT)

    assert runtimes["cuda12", "default"] == "12.4.1"
    assert runtimes["cuda12", "sglang"] == "12.9.1"
    assert runtimes["cuda12", "sglang-embedding"] == runtimes["cuda12", "sglang"]
    assert runtimes["cuda12", "sglang-vision-extract"] == runtimes["cuda12", "sglang"]
    assert runtimes["cuda13", "*"] == "13.0.0"
    assert runtimes["cuda12", matrix.RUST_BUNDLE] == "12.4.1"


@pytest.mark.parametrize(("runtime", "driver"), [("12.4.1", "525"), ("12.9.1", "525"), ("13.0.0", "580")])
def test_driver_floor_follows_the_cuda_major(runtime, driver):
    assert matrix.driver_floor(floors(), runtime)["driver_min"] == driver


def test_an_unrecorded_cuda_major_fails_rather_than_guessing():
    with pytest.raises(ValueError, match="CUDA 14"):
        matrix.driver_floor(floors(), "14.0.0")


def test_driver_floors_cite_nvidia():
    table = floors()

    assert {floor["cuda_major"] for floor in table["floors"]} == {"12", "13"}
    for floor in table["floors"]:
        assert floor["source"].startswith("https://docs.nvidia.com/")


@pytest.mark.parametrize(
    ("bundle", "engine"),
    [
        ({"deps": {"sglang": {"version": "==0.5.10.post1", "marker": "linux"}}}, ("sglang", "0.5.10.post1")),
        ({"deps": {"sglang": "==0.5.20", "xgrammar": "==0.2.1"}}, ("sglang", "0.5.20")),
        ({"deps": {"tensorrt-llm": "==1.3.0rc24", "transformers": "==5.5.4"}}, ("tensorrt-llm", "1.3.0rc24")),
        ({"deps": {"transformers": ">=5.14,<6"}}, ("transformers", None)),
        ({"engine": "candle", "deps": {}}, ("candle", None)),
    ],
)
def test_bundle_engine_reports_only_exact_pins(bundle, engine):
    assert matrix.bundle_engine(bundle) == engine


@pytest.mark.parametrize(
    ("profile", "gpu"),
    [
        ("h100-256k", "h100"),
        ("h200-256k-no-spec", "h200"),
        ("a100-40gb", "a100-40"),
        ("rtx-pro-6000-256k", "rtx-pro-6000"),
        ("l4-smoke", "l4"),
        ("long-context", None),
        ("default-no-admission", None),
        ("default", None),
    ],
)
def test_gpu_class_comes_from_the_profile_name(profile, gpu):
    assert matrix.gpu_class(profile) == gpu


def test_every_release_lane_and_the_rust_worker_appear_once():
    lanes = [lane["lane"] for lane in generated()["lanes"]]
    expected = [f"{platform}-{bundle}" for platform, bundle in matrix.release_targets(matrix.ROOT)]

    assert lanes == [*expected, matrix.RUST_LANE]
    rust = generated()["lanes"][-1]
    assert rust["image"] == f"{matrix.REGISTRY}/sie-server-rust:v0.0.0-{matrix.RUST_LANE}"
    assert rust["compute_capability"] == "8.9"


def test_every_cell_has_a_known_status_and_a_reason():
    result = generated()

    assert result["cells"]
    for cell in result["cells"]:
        assert cell["status"] in matrix.STATUSES
        assert cell["reason"]
    assert sum(result["summary"].values()) == len(result["cells"])


def test_cuda_only_engines_are_unsupported_on_cpu_lanes():
    cpu_sglang = [cell for cell in generated()["cells"] if cell["lane"] == "cpu-sglang"]

    assert cpu_sglang
    assert {cell["status"] for cell in cpu_sglang} == {"unsupported"}
    assert {cell["gpu_class"] for cell in cpu_sglang} == {"cpu"}


def test_sglang_embeddings_have_a_declared_cuda_release_lane():
    embedding = [cell for cell in generated()["cells"] if cell["bundle"] == "sglang-embedding"]

    assert embedding
    assert {(cell["lane"], cell["status"]) for cell in embedding} == {("cuda12-sglang-embedding", "declared")}


def test_a_bundle_without_a_published_image_is_unsupported(monkeypatch):
    release_targets = matrix.release_targets
    monkeypatch.setattr(
        matrix,
        "release_targets",
        lambda root: [target for target in release_targets(root) if target != ("cuda12", "sglang-embedding")],
    )
    result = matrix.build(matrix.ROOT, "0.0.0", floors=floors())
    embedding = [cell for cell in result["cells"] if cell["bundle"] == "sglang-embedding"]

    assert embedding
    assert {(cell["lane"], cell["status"]) for cell in embedding} == {(None, "unsupported")}


def test_profile_variants_carry_their_profile_and_width():
    cells = {(cell["lane"], cell["variant_id"]): cell for cell in generated()["cells"]}

    variant = cells["cuda13-sglang-cu130", "Qwen/Qwen3.8-27B-FP8:h100-256k"]
    assert (variant["model"], variant["profile"], variant["gpu_class"]) == ("Qwen/Qwen3.8-27B-FP8", "h100-256k", "h100")
    assert cells["cuda13-sglang-cu130", "zai-org/GLM-5.3-Flash"]["tp_width"] == 8
    assert cells["cuda12-sglang", "Qwen/Qwen3-0.6B"]["profile"] == "default"
    assert cells["cuda12-default", "nvidia/NV-Embed-v2:h100"]["gpu_class"] == "h100"


def run(lane, route, image, result="passed", **extra):
    return {
        "lane": lane,
        "route": route,
        "image": image,
        "result": result,
        "gpu": "NVIDIA L4",
        "gpu_count": 1,
        "driver": "580.95.05",
        "cuda_driver_api": "13.0",
        "date": "2026-10-15",
        "test": "serve-and-call",
        **extra,
    }


def test_a_passing_run_on_the_lane_digest_marks_the_cell_tested():
    digest = "sha256:" + "a" * 64
    image = f"ghcr.io/superlinked/sie-server@{digest}"
    tested = {"runs": [run("cuda12-sglang", "Qwen/Qwen3-0.6B", image)]}
    result = matrix.build(
        matrix.ROOT, "0.0.0", floors=floors(), metadata={"cuda12-sglang": {"digest": digest}}, tested=tested
    )
    cells = {(cell["lane"], cell["variant_id"]): cell for cell in result["cells"]}

    cell = cells["cuda12-sglang", "Qwen/Qwen3-0.6B"]
    assert cell["status"] == "tested"
    assert cell["evidence"]["driver"] == "580.95.05"
    assert result["summary"]["tested"] == 1


def test_a_run_on_another_digest_is_not_evidence_for_this_release():
    tested = {"runs": [run("cuda12-sglang", "Qwen/Qwen3-0.6B", "ghcr.io/superlinked/sie-server@sha256:" + "b" * 64)]}
    result = matrix.build(
        matrix.ROOT,
        "0.0.0",
        floors=floors(),
        metadata={"cuda12-sglang": {"digest": "sha256:" + "a" * 64}},
        tested=tested,
    )

    assert result["summary"]["tested"] == 0


def test_a_run_on_the_lane_digest_counts_whatever_other_images_were_run():
    digest = "sha256:" + "a" * 64
    other = "ghcr.io/superlinked/sie-server@sha256:" + "b" * 64
    tested = {
        "runs": [
            run("cuda12-sglang", "Qwen/Qwen3-0.6B", f"ghcr.io/superlinked/sie-server@{digest}"),
            run("cuda12-sglang", "Qwen/Qwen3-0.6B", other),
        ]
    }
    result = matrix.build(
        matrix.ROOT, "0.0.0", floors=floors(), metadata={"cuda12-sglang": {"digest": digest}}, tested=tested
    )

    assert result["summary"]["tested"] == 1


def test_a_conclusive_failure_outranks_a_pass_on_the_same_image():
    digest = "sha256:" + "a" * 64
    image = f"ghcr.io/superlinked/sie-server@{digest}"
    tested = {
        "runs": [
            run("cuda12-sglang", "Qwen/Qwen3-0.6B", image),
            run("cuda12-sglang", "Qwen/Qwen3-0.6B", image, result="failed", conclusive=True, detail="crash"),
        ]
    }
    result = matrix.build(
        matrix.ROOT, "0.0.0", floors=floors(), metadata={"cuda12-sglang": {"digest": digest}}, tested=tested
    )
    cells = {(cell["lane"], cell["variant_id"]): cell for cell in result["cells"]}

    assert cells["cuda12-sglang", "Qwen/Qwen3-0.6B"]["status"] == "unsupported"
    assert result["summary"]["tested"] == 0


def test_runs_without_a_recorded_lane_digest_classify_nothing():
    image = "ghcr.io/superlinked/sie-server@sha256:" + "a" * 64
    tested = {
        "runs": [
            run("cuda12-sglang", "Qwen/Qwen3-0.6B", image),
            run("cuda12-sglang", "Qwen/Qwen3-4B-Instruct-2507", image, result="failed", conclusive=True, detail="x"),
        ]
    }
    result = matrix.build(matrix.ROOT, "0.0.0", floors=floors(), tested=tested)
    cells = {(cell["lane"], cell["variant_id"]): cell for cell in result["cells"]}

    assert result["summary"]["tested"] == 0
    assert cells["cuda12-sglang", "Qwen/Qwen3-4B-Instruct-2507"]["status"] == "declared"


def test_a_conclusive_failure_is_unsupported_with_its_cause():
    digest = "sha256:" + "a" * 64
    image = f"ghcr.io/superlinked/sie-server@{digest}"
    tested = {
        "runs": [
            run("cuda12-sglang", "Qwen/Qwen3-0.6B", image, result="failed", conclusive=True, detail="out of memory"),
            run("cuda12-sglang", "Qwen/Qwen3-4B-Instruct-2507", image, result="failed", conclusive=False, detail="x"),
        ]
    }
    metadata = {"cuda12-sglang": {"digest": digest}}
    cells = {
        (cell["lane"], cell["variant_id"]): cell
        for cell in matrix.build(matrix.ROOT, "0.0.0", floors=floors(), metadata=metadata, tested=tested)["cells"]
    }

    assert cells["cuda12-sglang", "Qwen/Qwen3-0.6B"]["status"] == "unsupported"
    assert "out of memory" in cells["cuda12-sglang", "Qwen/Qwen3-0.6B"]["reason"]
    assert cells["cuda12-sglang", "Qwen/Qwen3-4B-Instruct-2507"]["status"] == "declared"


def test_observed_image_versions_win_over_bundle_pins():
    metadata = {"cuda12-default": {"distributions": {"transformers": "4.57.1", "torch": "2.9.1"}}}
    lanes = {
        lane["lane"]: lane for lane in matrix.build(matrix.ROOT, "0.0.0", floors=floors(), metadata=metadata)["lanes"]
    }

    assert lanes["cuda12-default"]["engine_version"] == "4.57.1"
    assert lanes["cuda12-default"]["engine_version_source"] == "image"
    assert lanes["cuda13-sglang-cu130"]["engine_version_source"] == "bundle-pin"


def test_diff_lists_status_and_driver_changes():
    old = generated()
    new = copy.deepcopy(old)
    new["cells"][0]["status"] = "unsupported"
    new["lanes"][0]["driver_min"] = "580"

    changes = matrix.diff(old, new)

    assert changes[0].endswith(f"{old['cells'][0]['status']} -> unsupported")
    assert changes[-1] == f"{old['lanes'][0]['lane']} driver_min: {old['lanes'][0]['driver_min']} -> 580"
    assert matrix.diff(old, old) == []


@pytest.mark.parametrize("path", sorted(DOCS.glob("sie-compatibility-v*.json")), ids=lambda path: path.name)
def test_committed_matrices_carry_the_current_schema_and_their_release(path):
    committed = json.loads(path.read_text())

    assert committed["schema"] == matrix.SCHEMA
    assert path.name == f"sie-compatibility-v{committed['release']}.json"


def test_committed_docs_page_renders_from_the_matrix_it_names():
    page = (matrix.ROOT / "docs/compatibility.md").read_text()
    source = re.search(r"from (docs/compatibility/sie-compatibility-v[0-9.]+\.json)", page)

    assert source is not None
    assert matrix.render_markdown(json.loads((matrix.ROOT / source[1]).read_text())) == page
