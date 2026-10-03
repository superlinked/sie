from pathlib import Path


def test_runtime_identity_is_pinned_to_chart_security_context() -> None:
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text(encoding="utf-8")
    normalized = " ".join(dockerfile.split())

    assert "groupadd --gid 1000 sie" in normalized
    assert "useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash sie" in normalized
    user_instructions = [line.strip() for line in dockerfile.splitlines() if line.lstrip().startswith("USER ")]
    assert user_instructions == ["USER sie"]
