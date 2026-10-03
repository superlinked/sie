"""Build a ready-to-share plugin install pack for an SIE MCP edge."""

from __future__ import annotations

import shlex
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from sie_mcp.skill_zip import build_skill_zip, skill_name


@dataclass(frozen=True)
class PluginPackReport:
    """Paths written by ``build_plugin_pack``."""

    out_dir: Path
    mcp_url: str
    skill_name: str
    skill_zip: Path
    claude_code_skills: tuple[Path, ...]
    install_guide: Path
    cowork_guide: Path | None

    @property
    def claude_code_skill(self) -> Path:
        """Backward-compatible primary Claude Code skill path."""
        return self.claude_code_skills[0]


def normalize_mcp_url(raw: str) -> str:
    """Return a clean absolute MCP endpoint URL whose path ends with ``/mcp``."""
    value = raw.strip()
    if not value:
        raise ValueError("MCP URL is required")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("MCP URL must be an absolute http(s) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("MCP URL must not include a query string or fragment")

    path = parsed.path.rstrip("/")
    if not path:
        path = "/mcp"
    elif not path.endswith("/mcp"):
        raise ValueError("MCP URL path must end with /mcp")

    return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))


def build_plugin_pack(
    skill_md: str,
    *,
    mcp_url: str,
    out_dir: Path,
    connector_secret: str | None = None,
    cluster_label: str = "sie-cluster",
    claude_code_skill_mds: Iterable[str] = (),
    cowork_guide_md: str | None = None,
) -> PluginPackReport:
    """Write a plugin pack: skill ZIP, Claude Code skill, and install guide."""
    # Keep accepting the former keyword without using or persisting it. Users
    # provide connector credentials only at installation time.
    del connector_secret
    normalized_url = normalize_mcp_url(mcp_url)
    name = skill_name(skill_md)

    out_dir.mkdir(parents=True, exist_ok=True)

    skill_zip = out_dir / f"{name}-skill.zip"
    skill_zip.write_bytes(build_skill_zip(skill_md))

    claude_code_skills = _write_claude_code_skills(out_dir, [skill_md, *claude_code_skill_mds])

    cowork_guide = None
    if cowork_guide_md is not None:
        cowork_dir = out_dir / "cowork"
        cowork_dir.mkdir(parents=True, exist_ok=True)
        cowork_guide = cowork_dir / "superlinked.md"
        cowork_guide.write_text(cowork_guide_md, encoding="utf-8")

    install_guide = out_dir / "INSTALL.md"
    install_guide.write_text(
        render_install_guide(
            cluster_label=cluster_label,
            mcp_url=normalized_url,
            skill_name_value=name,
            skill_zip_name=skill_zip.name,
            claude_code_skill_names=tuple(path.parent.name for path in claude_code_skills),
            has_cowork_guide=cowork_guide is not None,
        ),
        encoding="utf-8",
    )

    return PluginPackReport(
        out_dir=out_dir,
        mcp_url=normalized_url,
        skill_name=name,
        skill_zip=skill_zip,
        claude_code_skills=tuple(claude_code_skills),
        install_guide=install_guide,
        cowork_guide=cowork_guide,
    )


def _write_claude_code_skills(out_dir: Path, skill_mds: Iterable[str]) -> list[Path]:
    """Write unique Claude Code skills in Claude's local skill layout."""
    claude_code_root = out_dir / "claude-code"
    if claude_code_root.exists():
        shutil.rmtree(claude_code_root)

    paths: list[Path] = []
    seen: set[str] = set()
    for skill_md in skill_mds:
        name = skill_name(skill_md)
        if name in seen:
            continue
        seen.add(name)
        claude_code_dir = claude_code_root / name
        claude_code_dir.mkdir(parents=True, exist_ok=True)
        skill_path = claude_code_dir / "SKILL.md"
        skill_path.write_text(skill_md, encoding="utf-8")
        paths.append(skill_path)
    return paths


def render_install_guide(
    *,
    cluster_label: str,
    mcp_url: str,
    skill_name_value: str,
    skill_zip_name: str,
    claude_code_skill_names: tuple[str, ...],
    has_cowork_guide: bool,
) -> str:
    """Render endpoint-specific install instructions for the generated plugin pack."""
    secret_placeholder = "<connector-secret-from-local-secret-store>"  # noqa: S105 - non-secret placeholder.
    auth_header = f"Authorization: Bearer {secret_placeholder}"
    claude_code_add = (
        "claude mcp add --scope user --transport http "
        f"{shlex.quote(skill_name_value)} {shlex.quote(mcp_url)} "
        '--header "Authorization: Bearer '
        '${SIE_MCP_CONNECTOR_SECRET:?set it from your local secret store}"'
    )
    claude_code_names = ", ".join(f"`{name}`" for name in claude_code_skill_names)
    claude_code_files = (
        f"- `claude-code/*/SKILL.md` - Claude Code skill files ({claude_code_names}).\n"
        if claude_code_skill_names
        else ""
    )
    claude_code_install = (
        "mkdir -p ~/.claude/skills\ncp -R claude-code/* ~/.claude/skills/\n" if claude_code_skill_names else ""
    )
    claude_code_skills_note = (
        "Restart Claude Code after adding the server and skills. "
        f"The included Claude Code skills ({claude_code_names}) call the MCP tools "
        "exposed by the edge directly. "
        if claude_code_skill_names
        else "Restart Claude Code after adding the server. "
    )
    redaction_note = (
        "The MCP redaction flow returns redacted text and counts, but not a local "
        "placeholder-to-original map; do not use it when de-redaction is required."
        if "redact-pii" in claude_code_skill_names
        else ""
    )
    cowork_note = "- `cowork/superlinked.md` - Cowork/plugin install notes.\n" if has_cowork_guide else ""

    return (
        f"# Superlinked MCP plugin pack for {cluster_label}\n\n"
        "This pack connects an agent surface to an already-running Superlinked MCP edge. "
        "Use it for any hosted or self-hosted cluster where an operator has given you "
        "an MCP URL and connector secret.\n\n"
        "## Files\n\n"
        f"- `{skill_zip_name}` - uploadable skill ZIP for claude.ai.\n"
        f"{claude_code_files}"
        f"{cowork_note}"
        "\n"
        "## Connector\n\n"
        f"- MCP endpoint: `{mcp_url}`\n"
        "- Connector secret: retrieve it from your connector configuration or local secret store.\n\n"
        "No connector secret is embedded in any file in this pack. Keep it in connector settings "
        "or a local secret store; never paste it into chat.\n\n"
        "## Claude Code\n\n"
        "Load the secret into a local environment variable from your secret store, then run "
        "these commands from this directory. Do not save the value in this pack:\n\n"
        "```bash\n"
        f"{claude_code_add}\n"
        f"{claude_code_install}"
        "unset SIE_MCP_CONNECTOR_SECRET\n"
        "```\n\n"
        f"{claude_code_skills_note}"
        f"{redaction_note}\n\n"
        "## claude.ai / Claude desktop app\n\n"
        "1. Add a custom connector with the MCP endpoint above.\n"
        "2. Click Connect and complete the OAuth sign-in. Enter the connector secret on the "
        "Superlinked authorize page, not in chat.\n"
        f"3. Upload `{skill_zip_name}` as the Superlinked document-offload skill.\n\n"
        "## Cowork\n\n"
        "Add a custom MCP connector with the endpoint above and this header:\n\n"
        "```text\n"
        f"{auth_header}\n"
        "```\n\n"
        "Then install the Superlinked document-offload skill from `packages/sie_mcp/plugin/SKILL.md` "
        "or the Cowork guide in this pack if present.\n"
    )
