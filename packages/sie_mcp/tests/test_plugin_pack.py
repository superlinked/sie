import base64
import io
import os
import shlex
import subprocess
import zipfile
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

import pytest
from click import Group
from sie_mcp.cli import app
from sie_mcp.plugin_pack import build_plugin_pack, normalize_mcp_url, render_install_guide
from typer.main import get_command
from typer.testing import CliRunner

_SKILL_MD = """---
name: superlinked-docs
description: >-
  Offload document work to the Superlinked inference cluster: convert documents
  to clean markdown instead of ingesting the file directly.
---

# Body
"""

_PARSE_DOCUMENT_SKILL_MD = """---
name: parse-document
description: Parse documents through the Superlinked MCP edge.
---

# Parse
"""

_SUMMARIZE_DOCUMENT_SKILL_MD = """---
name: summarize-document
description: Summarize documents through the Superlinked MCP edge.
---

# Summarize
"""

_EXTRACT_ENTITIES_SKILL_MD = """---
name: extract-entities
description: Extract entities through the Superlinked MCP edge.
---

# Extract
"""

_REDACT_PII_SKILL_MD = """---
name: redact-pii
description: Redact PII through the Superlinked MCP edge.
---

# Redact
"""

_TEST_CONNECTOR_SECRET = "pack-secret/that+must?stay=local"  # noqa: S105 - inert test fixture


def _secret_encodings(secret: str) -> tuple[bytes, ...]:
    raw = secret.encode()
    return (
        raw,
        base64.b64encode(raw),
        base64.urlsafe_b64encode(raw),
        quote(secret, safe="").encode(),
        raw.hex().encode(),
    )


def _assert_secret_absent(value: bytes | str, secret: str = _TEST_CONNECTOR_SECRET) -> None:
    contents = value.encode() if isinstance(value, str) else value
    for encoded in _secret_encodings(secret):
        assert encoded not in contents


def _assert_pack_has_no_secret(out_dir: Path, secret: str = _TEST_CONNECTOR_SECRET) -> None:
    generated_files = tuple(path for path in out_dir.rglob("*") if path.is_file())
    assert generated_files
    for path in generated_files:
        _assert_secret_absent(path.read_bytes(), secret)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://mcp.example.com", "https://mcp.example.com/mcp"),
        ("https://mcp.example.com/", "https://mcp.example.com/mcp"),
        ("https://mcp.example.com/mcp/", "https://mcp.example.com/mcp"),
        ("https://mcp.example.com/api/mcp", "https://mcp.example.com/api/mcp"),
    ],
)
def test_normalize_mcp_url(raw: str, expected: str) -> None:
    assert normalize_mcp_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "mcp.example.com/mcp",
        "ftp://mcp.example.com/mcp",
        "https://mcp.example.com/not-mcp",
        "https://mcp.example.com/mcp?secret=nope",
        "https://mcp.example.com/mcp#frag",
    ],
)
def test_normalize_mcp_url_rejects_bad_values(raw: str) -> None:
    with pytest.raises(ValueError, match="MCP URL"):
        normalize_mcp_url(raw)


def test_build_plugin_pack_writes_install_artifacts(tmp_path: Path) -> None:
    report = build_plugin_pack(
        _SKILL_MD,
        mcp_url="https://mcp.example.com",
        connector_secret=_TEST_CONNECTOR_SECRET,
        claude_code_skill_mds=[
            _EXTRACT_ENTITIES_SKILL_MD,
            _PARSE_DOCUMENT_SKILL_MD,
            _REDACT_PII_SKILL_MD,
            _SUMMARIZE_DOCUMENT_SKILL_MD,
        ],
        cowork_guide_md="# Cowork install\n",
        out_dir=tmp_path,
    )

    assert report.mcp_url == "https://mcp.example.com/mcp"
    assert report.skill_name == "superlinked-docs"
    assert report.skill_zip == tmp_path / "superlinked-docs-skill.zip"
    assert report.claude_code_skill == tmp_path / "claude-code" / "superlinked-docs" / "SKILL.md"
    assert report.claude_code_skills == (
        tmp_path / "claude-code" / "superlinked-docs" / "SKILL.md",
        tmp_path / "claude-code" / "extract-entities" / "SKILL.md",
        tmp_path / "claude-code" / "parse-document" / "SKILL.md",
        tmp_path / "claude-code" / "redact-pii" / "SKILL.md",
        tmp_path / "claude-code" / "summarize-document" / "SKILL.md",
    )
    assert report.cowork_guide == tmp_path / "cowork" / "superlinked.md"
    assert report.install_guide == tmp_path / "INSTALL.md"

    assert report.claude_code_skill.read_text(encoding="utf-8") == _SKILL_MD
    parse_skill = tmp_path / "claude-code" / "parse-document" / "SKILL.md"
    assert parse_skill.read_text(encoding="utf-8") == _PARSE_DOCUMENT_SKILL_MD
    assert report.cowork_guide is not None
    assert report.cowork_guide.read_text(encoding="utf-8") == "# Cowork install\n"

    guide = report.install_guide.read_text(encoding="utf-8")
    assert "# Superlinked MCP plugin pack for sie-cluster" in guide
    assert "https://mcp.example.com/mcp" in guide
    _assert_pack_has_no_secret(tmp_path)
    _assert_secret_absent(repr(report))
    _assert_secret_absent(repr(asdict(report)))
    assert "${SIE_MCP_CONNECTOR_SECRET:?set it from your local secret store}" in guide
    assert "<connector-secret-from-local-secret-store>" in guide
    assert "claude mcp add --scope user --transport http superlinked-docs" in guide
    assert "cp -R claude-code/* ~/.claude/skills/" in guide
    assert "parse-document" in guide
    assert "summarize-document" in guide
    assert "extract-entities" in guide
    assert "redact-pii" in guide
    assert (
        "The included Claude Code skills (`superlinked-docs`, `extract-entities`, `parse-document`, "
        "`redact-pii`, `summarize-document`)" in guide
    )
    assert "superlinked-docs-skill.zip" in guide


def test_plugin_router_uses_shell_safe_example_values() -> None:
    router = Path(__file__).parents[1] / "plugin" / "superlinked.md"
    guidance = router.read_text(encoding="utf-8")

    assert "<mcp-host>" not in guidance
    assert "<secret>" not in guidance
    assert "https://mcp.example.com/mcp" in guidance
    assert "'replace-with-connector-secret'" in guidance
    assert "`SIE_MCP_CONNECTOR_SECRET`" in guidance
    assert "client-side shell variable" in guidance
    assert "`SIE_MCP_CONNECTOR_SECRETS` runtime map" in guidance


def test_public_mcp_command_examples_use_shell_safe_values() -> None:
    package_root = Path(__file__).parents[1]
    readme = (package_root / "README.md").read_text(encoding="utf-8")
    guidance = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (package_root / "README.md", package_root / "docs" / "examples" / "mcp_smoke.py")
    )

    for placeholder in (
        "<cluster-api-key>",
        "<cluster-gateway>",
        "<cluster-name>",
        "<connector-secret>",
        "<mcp-host>",
        "<secret>",
        "<tag>",
    ):
        assert placeholder not in guidance
    assert "https://mcp.example.com/mcp" in guidance
    assert "'replace-with-connector-secret'" in guidance
    assert "export SIE_MCP_PUBLIC_URL='https://mcp.example.com'" in readme


def test_helm_and_plugin_pack_examples_share_one_local_only_demo_secret() -> None:
    repo_root = Path(__file__).parents[3]
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    helm_values = (repo_root / "deploy/helm/sie-cluster/values.yaml").read_text(encoding="utf-8")

    assert "`local-dev-secret` as a local-only demo connector secret" in readme
    assert '--from-literal=connector-secrets="local-dev-secret:local-dev"' in readme
    assert "SIE_MCP_CONNECTOR_SECRET='local-dev-secret'" in readme
    assert "Local-only demo example: `local-dev-secret:local-dev`" in helm_values


def test_build_plugin_pack_keeps_connector_secret_out_of_skill_zip(tmp_path: Path) -> None:
    report = build_plugin_pack(
        _SKILL_MD,
        mcp_url="https://mcp.example.com/mcp",
        connector_secret=_TEST_CONNECTOR_SECRET,
        out_dir=tmp_path,
    )

    with zipfile.ZipFile(io.BytesIO(report.skill_zip.read_bytes())) as archive:
        names = archive.namelist()
        skill_text = archive.read("superlinked-docs/SKILL.md").decode()

    assert names == ["superlinked-docs/SKILL.md"]
    _assert_secret_absent(skill_text)
    _assert_pack_has_no_secret(tmp_path)


def test_build_plugin_pack_uses_placeholder_when_secret_omitted(tmp_path: Path) -> None:
    report = build_plugin_pack(_SKILL_MD, mcp_url="https://mcp.example.com/mcp", out_dir=tmp_path)

    guide = report.install_guide.read_text(encoding="utf-8")
    assert "Authorization: Bearer ${SIE_MCP_CONNECTOR_SECRET:?set it from your local secret store}" in guide
    assert "Authorization: Bearer <connector-secret-from-local-secret-store>" in guide
    assert "Claude Code skill files (`superlinked-docs`)" in guide
    assert "The included Claude Code skills (`superlinked-docs`)" in guide
    assert "parse-document" not in guide
    assert "summarize-document" not in guide
    assert "extract-entities" not in guide
    assert "redact-pii" not in guide


def test_generated_claude_command_cannot_run_with_unset_secret(tmp_path: Path) -> None:
    report = build_plugin_pack(_SKILL_MD, mcp_url="https://mcp.example.com/mcp", out_dir=tmp_path / "pack")
    guide = report.install_guide.read_text(encoding="utf-8")
    command = next(line for line in guide.splitlines() if line.startswith("claude mcp add "))
    invoked = tmp_path / "claude-invoked"
    script = (
        f"claude() {{ printf invoked > {shlex.quote(str(invoked))}; }}\nunset SIE_MCP_CONNECTOR_SECRET\n{command}\n"
    )
    env = os.environ.copy()
    env.pop("SIE_MCP_CONNECTOR_SECRET", None)

    result = subprocess.run(  # noqa: S603 - executes the generated command in an isolated shell fixture.
        ["/bin/bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert not invoked.exists()
    assert "SIE_MCP_CONNECTOR_SECRET" in result.stderr


def test_render_install_guide_omits_claude_code_skills_when_none_are_included() -> None:
    guide = render_install_guide(
        cluster_label="sie-cluster",
        mcp_url="https://mcp.example.com/mcp",
        skill_name_value="superlinked-docs",
        skill_zip_name="superlinked-docs-skill.zip",
        claude_code_skill_names=(),
        has_cowork_guide=False,
    )

    assert "claude-code/*/SKILL.md" not in guide
    assert "cp -R claude-code/* ~/.claude/skills/" not in guide
    assert "Claude Code skill files" not in guide
    assert "The included Claude Code skills" not in guide
    assert "Restart Claude Code after adding the server." in guide


def test_plugin_pack_cli_never_emits_or_persists_connector_secret(tmp_path: Path) -> None:
    result = CliRunner(mix_stderr=False).invoke(
        app,
        [
            "plugin-pack",
            "--mcp-url",
            "https://mcp.example.com/mcp",
            "--connector-secret",
            _TEST_CONNECTOR_SECRET,
            "--out-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    _assert_secret_absent(result.output)
    _assert_secret_absent(result.stdout)
    _assert_secret_absent(result.stderr)
    _assert_secret_absent(repr(result.exception))
    _assert_pack_has_no_secret(tmp_path)


def test_plugin_pack_cli_exception_never_emits_connector_secret(tmp_path: Path) -> None:
    result = CliRunner(mix_stderr=False).invoke(
        app,
        [
            "plugin-pack",
            "--mcp-url",
            "not-an-http-url",
            "--connector-secret",
            _TEST_CONNECTOR_SECRET,
            "--out-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    _assert_secret_absent(result.output)
    _assert_secret_absent(result.stdout)
    _assert_secret_absent(result.stderr)
    _assert_secret_absent(repr(result.exception))


def test_plugin_pack_help_describes_both_endpoint_types() -> None:
    root_command = get_command(app)
    assert isinstance(root_command, Group)
    mcp_url_option = next(
        parameter for parameter in root_command.commands["plugin-pack"].params if parameter.name == "mcp_url"
    )

    assert mcp_url_option.help == "Hosted or self-hosted MCP endpoint to install, e.g. https://mcp.example.com/mcp."


def test_build_plugin_pack_refreshes_claude_code_skills(tmp_path: Path) -> None:
    stale = tmp_path / "claude-code" / "stale" / "SKILL.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")

    build_plugin_pack(_SKILL_MD, mcp_url="https://mcp.example.com/mcp", out_dir=tmp_path)

    assert not stale.exists()
    assert (tmp_path / "claude-code" / "superlinked-docs" / "SKILL.md").exists()
