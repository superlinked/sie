from __future__ import annotations

import os
import re
import stat
import time
from collections.abc import Iterator
from pathlib import Path

DATA_DIR_ENV = "SIE_DATA_DIR"
TEMPORARY_FILE_MAX_AGE_S = 60.0
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


def owned_files(directory: Path, name: re.Pattern[str]) -> Iterator[tuple[Path, os.stat_result]]:
    """Regular files in ``directory`` whose whole name matches ``name`` and that this user owns."""
    for entry in directory.iterdir():
        if not name.fullmatch(entry.name):
            continue
        try:
            info = entry.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid():
            yield entry, info


def remove_stale_files(directory: Path, name: re.Pattern[str]) -> None:
    """Remove the files ``owned_files`` finds that have one link and are too old to still be in use.

    A file younger than ``TEMPORARY_FILE_MAX_AGE_S`` may belong to a start
    that is still writing it, so it is left in place. Errors are ignored.
    """
    cutoff = time.time() - TEMPORARY_FILE_MAX_AGE_S
    try:
        for entry, info in owned_files(directory, name):
            if info.st_nlink == 1 and info.st_mtime < cutoff:
                entry.unlink(missing_ok=True)
    except OSError:
        return
