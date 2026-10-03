"""CLI entry point for the SIE MCP edge service."""

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from sie_mcp.plugin_pack import build_plugin_pack
from sie_mcp.skill_zip import build_skill_zip, skill_name

app = typer.Typer(name="sie-mcp", help="SIE MCP edge service.", no_args_is_help=True)

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SKILL_MD = _PACKAGE_ROOT / "plugin" / "SKILL.md"
_DEFAULT_COWORK_GUIDE = _PACKAGE_ROOT / "plugin" / "superlinked.md"
_DEFAULT_CLAUDE_CODE_SKILLS_DIR = _PACKAGE_ROOT / "plugin" / "claude-code"


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", "-h", help="Host to bind to.")] = "0.0.0.0",  # noqa: S104 — intentional bind to all interfaces for the edge service
    port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on.")] = 8088,
    log_level: Annotated[
        str,
        typer.Option("--log-level", "-l", envvar="SIE_LOG_LEVEL", help="Log level."),
    ] = "info",
    reload: Annotated[
        bool,
        typer.Option("--reload", "-r", help="Auto-reload for development."),
    ] = False,
) -> None:
    """Start the MCP edge over remote streamable-HTTP."""
    _setup_logging(log_level)
    uvicorn.run(
        "sie_mcp.app:build_app",
        host=host,
        port=port,
        factory=True,
        reload=reload,
        log_level=log_level.lower(),
    )


@app.command("skill-zip")
def skill_zip(
    skill: Annotated[
        Path,
        typer.Option(
            "--skill", "-s", help="Path to the SKILL.md to package (default: the surface-agnostic plugin/SKILL.md)."
        ),
    ] = _DEFAULT_SKILL_MD,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Output .zip path (default: dist/<name>-skill.zip)."),
    ] = None,
) -> None:
    """Package the surface-agnostic Superlinked Agent Skill as a claude.ai-uploadable ZIP."""
    if not skill.is_file():
        raise typer.BadParameter(f"SKILL.md not found at {skill}")
    text = skill.read_text(encoding="utf-8")
    data = build_skill_zip(text)
    target = out or (_PACKAGE_ROOT / "dist" / f"{skill_name(text)}-skill.zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    typer.echo(f"wrote {target} ({len(data)} bytes)")


@app.command("plugin-pack")
def plugin_pack(
    mcp_url: Annotated[
        str,
        typer.Option(
            "--mcp-url",
            envvar="SIE_MCP_URL",
            help="Hosted or self-hosted MCP endpoint to install, e.g. https://mcp.example.com/mcp.",
        ),
    ],
    connector_secret: Annotated[
        str | None,
        typer.Option(
            "--connector-secret",
            envvar="SIE_MCP_CONNECTOR_SECRET",
            help=(
                "Deprecated compatibility input; ignored and never used, written, or printed. "
                "Provide the secret during connector installation."
            ),
        ),
    ] = None,
    cluster_label: Annotated[
        str,
        typer.Option("--cluster-label", help="Human-readable cluster label for the generated install guide."),
    ] = "sie-cluster",
    skill: Annotated[
        Path,
        typer.Option("--skill", "-s", help="Path to the surface-agnostic Superlinked SKILL.md to package."),
    ] = _DEFAULT_SKILL_MD,
    cowork_guide: Annotated[
        Path | None,
        typer.Option("--cowork-guide", help="Optional Cowork install guide to include in the pack."),
    ] = _DEFAULT_COWORK_GUIDE,
    claude_code_skills_dir: Annotated[
        Path | None,
        typer.Option(
            "--claude-code-skills-dir",
            help="Optional directory of Claude Code skill folders to include in the pack.",
        ),
    ] = _DEFAULT_CLAUDE_CODE_SKILLS_DIR,
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", "-o", help="Output directory for the plugin pack."),
    ] = _PACKAGE_ROOT / "dist" / "superlinked-docs-plugin",
) -> None:
    """Build a quick install pack for a hosted or self-hosted Superlinked MCP edge."""
    if not skill.is_file():
        raise typer.BadParameter(f"SKILL.md not found at {skill}")
    cowork_text = None
    if cowork_guide is not None:
        if not cowork_guide.is_file():
            raise typer.BadParameter(f"Cowork guide not found at {cowork_guide}")
        cowork_text = cowork_guide.read_text(encoding="utf-8")
    claude_code_skill_mds = []
    if claude_code_skills_dir is not None:
        if not claude_code_skills_dir.is_dir():
            raise typer.BadParameter(f"Claude Code skills directory not found at {claude_code_skills_dir}")
        claude_code_skill_mds = [
            skill_file.read_text(encoding="utf-8") for skill_file in sorted(claude_code_skills_dir.glob("*/SKILL.md"))
        ]
    try:
        report = build_plugin_pack(
            skill.read_text(encoding="utf-8"),
            mcp_url=mcp_url,
            connector_secret=connector_secret,
            cluster_label=cluster_label,
            claude_code_skill_mds=claude_code_skill_mds,
            cowork_guide_md=cowork_text,
            out_dir=out_dir,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"wrote plugin pack to {report.out_dir}")
    typer.echo(f"- install guide: {report.install_guide}")
    typer.echo(f"- claude.ai skill ZIP: {report.skill_zip}")
    for skill_path in report.claude_code_skills:
        typer.echo(f"- Claude Code skill: {skill_path}")


@app.command()
def version() -> None:
    """Show version information."""
    try:
        typer.echo(f"sie-mcp {_pkg_version('sie-mcp')}")
    except PackageNotFoundError:
        typer.echo("sie-mcp (version unknown)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
