"""Package the claude.ai Agent Skill as an uploadable ZIP.

claude.ai's two-piece install takes the remote MCP connector plus a **skill ZIP**.
The platform requires the archive to contain a single top-level folder named after
the skill, with ``SKILL.md`` inside it (``superlinked-docs/SKILL.md``) — never at the
archive root — and enforces ``name``/``description`` frontmatter constraints. This
module reads the surface-agnostic ``plugin/SKILL.md`` (one skill across Cowork /
claude.ai / the desktop app), validates it against those rules, and emits a
deterministic ZIP so the ``mcp-skill-zip`` task produces a byte-stable artifact.
"""

import io
import re
import zipfile

# claude.ai skill-name rules: lowercase letters/digits/hyphens, <=64 chars, and the
# words "anthropic"/"claude" are reserved.
_NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")
_RESERVED_NAME_WORDS = ("anthropic", "claude")
_MAX_DESCRIPTION_LEN = 1024
_BLOCK_SCALAR_INDICATORS = frozenset({">", ">-", ">+", "|", "|-", "|+", ""})
# Fixed timestamp keeps the archive reproducible across builds.
_ZIP_DATE_TIME = (2020, 1, 1, 0, 0, 0)
_KEY_RE = re.compile(r"^([A-Za-z0-9_-]+):(.*)$")


class SkillPackagingError(Exception):
    """Raised when ``SKILL.md`` violates the claude.ai skill-ZIP contract."""


def parse_frontmatter(skill_md: str) -> dict[str, str]:
    """Extract the leading YAML frontmatter block as a flat ``{key: value}`` map.

    Only the leaf cases the skill contract needs are handled: inline scalars and
    folded/literal block scalars (e.g. ``description: >-``).
    """
    lines = skill_md.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillPackagingError("SKILL.md must start with a '---' frontmatter block")
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise SkillPackagingError("SKILL.md frontmatter block is not terminated with '---'")

    fields: dict[str, str] = {}
    body = lines[1:end]
    i = 0
    while i < len(body):
        match = _KEY_RE.match(body[i])
        if not match:
            i += 1
            continue
        key, rest = match.group(1), match.group(2).strip()
        if rest in _BLOCK_SCALAR_INDICATORS:
            collected: list[str] = []
            i += 1
            while i < len(body) and (body[i].startswith((" ", "\t")) or not body[i].strip()):
                collected.append(body[i].strip())
                i += 1
            fields[key] = " ".join(part for part in collected if part)
        else:
            fields[key] = _scalar_value(rest)
            i += 1
    return fields


def _scalar_value(raw: str) -> str:
    """Resolve an inline YAML scalar: unwrap quotes, else strip a trailing ``#`` comment."""
    value = raw.strip()
    if value[:1] in ("'", '"'):
        return value.strip("\"'")
    comment = value.find(" #")
    if comment != -1:
        value = value[:comment].rstrip()
    return value


def skill_name(skill_md: str) -> str:
    """Validate ``SKILL.md`` against the claude.ai contract and return its skill name."""
    fields = parse_frontmatter(skill_md)
    name = fields.get("name", "")
    if not _NAME_RE.match(name):
        raise SkillPackagingError(f"skill name {name!r} must match ^[a-z0-9-]{{1,64}}$")
    if any(word in name for word in _RESERVED_NAME_WORDS):
        raise SkillPackagingError(f"skill name {name!r} must not contain a reserved word")
    description = fields.get("description", "")
    if not description:
        raise SkillPackagingError("skill description is required and must be non-empty")
    if len(description) > _MAX_DESCRIPTION_LEN:
        raise SkillPackagingError(f"skill description exceeds {_MAX_DESCRIPTION_LEN} characters")
    return name


def build_skill_zip(skill_md: str, *, extra: dict[str, str] | None = None) -> bytes:
    """Build the skill ZIP bytes, nesting every file under ``<name>/``."""
    name = skill_name(skill_md)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        _write(archive, f"{name}/SKILL.md", skill_md)
        for rel_path, content in sorted((extra or {}).items()):
            _write(archive, f"{name}/{rel_path}", content)
    return buffer.getvalue()


def _write(archive: zipfile.ZipFile, arcname: str, content: str) -> None:
    info = zipfile.ZipInfo(arcname, date_time=_ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, content)
