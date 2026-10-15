from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "SIE_DATA_DIR"
_DIR_NAME = "sie"


def resolve_data_dir() -> Path | None:
    """The directory that holds the server token and the telemetry worker id.

    ``SIE_DATA_DIR`` wins. Otherwise it is a per-user directory:
    ``$XDG_DATA_HOME/sie``, or ``~/.local/share/sie``. ``None`` when neither
    ``SIE_DATA_DIR`` nor a home directory is available.
    """
    configured = os.environ.get(DATA_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    xdg_data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    if xdg_data_home and Path(xdg_data_home).is_absolute():
        return Path(xdg_data_home) / _DIR_NAME
    try:
        return Path.home() / ".local" / "share" / _DIR_NAME
    except RuntimeError:
        return None


def ensure_data_dir(path: Path) -> Path:
    """Create ``path`` readable by its owner only, leaving an existing directory as it is."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path
