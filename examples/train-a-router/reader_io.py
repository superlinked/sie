"""Local durability and provenance helpers; no service or credential dependency."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sync_files(paths, root):
    directories = {Path(root)}
    for path in paths:
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), "Expected a regular output file")
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        directories.update(parent for parent in path.parents if parent == root or root in parent.parents)
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@contextmanager
def exclusive_run(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".collector.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another collector owns this output directory") from error
        yield


@contextmanager
def durable_writes(module, root):
    """Wrap file persistence only; request, retry and selection code is unchanged."""
    root, original, dirty = Path(root).resolve(), module.dump, set()

    def write(path, value):
        original(path, value)
        dirty.add(Path(path).resolve())

    def commit():
        # run() writes the immutable plan directly, before its first commit.
        plan = root / "plan.json"
        if plan.exists() and not commit.plan_synced:
            dirty.add(plan)
            commit.plan_synced = True
        if dirty:
            sync_files(dirty, root)
            dirty.clear()

    commit.plan_synced = False
    module.dump = write
    try:
        yield commit
    finally:
        try:
            commit()
        finally:
            module.dump = original


class ChangedFileSync:
    """Sync changed local feature files without re-reading or hashing prior bytes."""

    def __init__(self):
        self.seen = {}

    def __call__(self, root):
        root, changed = Path(root).resolve(), []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if self.seen.get(path) != signature:
                changed.append(path)
                self.seen[path] = signature
        if changed:
            sync_files(changed, root)
