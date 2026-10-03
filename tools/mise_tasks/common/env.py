"""Environment variable parsing for mise task arguments.

Mise converts CLI flags to environment variables with the `usage_` prefix.
For example: `--port 8080` becomes `usage_port=8080`.
"""

from __future__ import annotations

import os
from pathlib import Path


def _get_usage_value(name: str) -> str:
    """Fetch usage_* value, supporting both '-' and '_' flag names."""
    value = os.environ.get(f"usage_{name}", "")
    if not value and "-" in name:
        value = os.environ.get(f"usage_{name.replace('-', '_')}", "")
    return value


def get_usage_flag(name: str, default: str | None = None) -> str | None:
    """Get a mise usage flag value from environment.

    Args:
        name: The flag name without 'usage_' prefix.
        default: Default value if not set.

    Returns:
        The flag value, or default if not set.

    Example:
        # For `--port 8080`:
        port = get_usage_flag("port", "8080")
    """
    value = _get_usage_value(name)
    return value or default


def get_usage_bool(name: str, default: bool = False) -> bool:
    """Get a boolean mise usage flag.

    Args:
        name: The flag name without 'usage_' prefix.
        default: Default value if not set.

    Returns:
        True if the flag is "true" (case-insensitive), default otherwise.

    Example:
        # For `--verbose`:
        verbose = get_usage_bool("verbose")
    """
    value = _get_usage_value(name)
    if not value:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def get_usage_int(name: str, default: int) -> int:
    """Get an integer mise usage flag.

    Args:
        name: The flag name without 'usage_' prefix.
        default: Default value if not set or invalid.

    Returns:
        The integer value, or default if not set or invalid.

    Example:
        # For `--port 8080`:
        port = get_usage_int("port", 8080)
    """
    value = _get_usage_value(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_usage_list(name: str, separator: str = ",") -> list[str]:
    """Get a list mise usage flag (comma-separated by default).

    Args:
        name: The flag name without 'usage_' prefix.
        separator: The separator character.

    Returns:
        List of values, empty list if not set.

    Example:
        # For `--models bge-m3,e5-base`:
        models = get_usage_list("models")  # ["bge-m3", "e5-base"]
    """
    value = _get_usage_value(name)
    if not value:
        return []
    return [item.strip() for item in value.split(separator) if item.strip()]


def resolve_project_root() -> Path:
    """Resolve the repo root for the current worktree."""
    cwd = Path.cwd()
    env_root = os.environ.get("MISE_PROJECT_ROOT")
    if env_root:
        env_path = Path(env_root)
        if cwd.is_relative_to(env_path):
            return env_path
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "mise.toml").exists():
            return candidate
    return cwd


def get_mise_env(project_root: Path | None = None) -> dict[str, str]:
    """Return environment with worktree-safe mise settings."""
    root = project_root or resolve_project_root()
    env = os.environ.copy()
    env.setdefault("MISE_PROJECT_ROOT", str(root))
    trusted = env.get("MISE_TRUSTED_CONFIG_PATHS", "")
    if trusted:
        paths = trusted.split(os.pathsep)
        if str(root) not in paths:
            env["MISE_TRUSTED_CONFIG_PATHS"] = os.pathsep.join([*paths, str(root)])
    else:
        env["MISE_TRUSTED_CONFIG_PATHS"] = str(root)
    # Prevent auto-install side effects during tasks.
    env.setdefault("MISE_AUTO_INSTALL", "0")
    env.setdefault("MISE_TASK_RUN_AUTO_INSTALL", "0")
    env.setdefault("MISE_EXEC_AUTO_INSTALL", "0")
    return env


def apply_mise_env(project_root: Path | None = None) -> None:
    """Apply worktree-safe mise settings to current process env."""
    os.environ.update(get_mise_env(project_root))
