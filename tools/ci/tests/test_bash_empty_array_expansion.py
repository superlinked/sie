"""An array that can be empty must not be expanded bare under ``set -u``.

macOS ships bash 3.2.57, where ``"${ARR[@]}"`` on an empty array is a hard
error under ``set -u`` rather than an empty expansion. A task script that
declares an array empty and fills it only on some paths, or seeds it from its
own arguments, then dies with ``ARR[@]: unbound variable`` on a stock macOS
shell before it does any work.

The portable form is ``${ARR[@]+"${ARR[@]}"}``, which expands to nothing when
the array is empty and is identical otherwise. ``"${ARR[@]:-}"`` is not a
substitute: it injects one empty argument.

This guards the class rather than the scripts that had it. Any array a task
script declares as possibly empty, either literally or from the script's own
arguments, must use the guarded form wherever it is expanded.
"""

from __future__ import annotations

import re
from pathlib import Path

TASKS = Path(__file__).resolve().parents[2] / "mise_tasks"
# An array that starts empty, or that starts as this script's arguments and so
# is empty whenever the task is run without any.
_POSSIBLY_EMPTY = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)=\(\s*(?:"\$@"\s*)?\)\s*$', re.MULTILINE)
_STRICT = ("set -u", "set -eu", "set -euo")


def _bare_expansions(source: str, name: str) -> list[int]:
    """Line numbers where ``name`` is expanded without the empty-array guard.

    Quoting is irrelevant to the crash. bash 3.2 refuses an empty array at
    ``${ARR[@]}`` just as it does at ``"${ARR[@]}"``, so the scan must not
    require the quotes.

    The guarded form contains the bare expansion inside itself, at a fixed
    offset, so a hit is excused only when the guard starts at exactly the
    position that would make this hit its inner expansion. Searching a window
    instead would excuse an unsafe expansion that merely sits beside a guarded
    one, as in ``${ARR[@]+"${ARR[@]}"}${ARR[@]}``, which bash still evaluates.
    """
    bare = f"${{{name}[@]}}"
    guarded = f'${{{name}[@]+"{bare}"}}'
    inner_offset = guarded.index(bare, 1)
    lines: list[int] = []
    for match in re.finditer(re.escape(bare), source):
        if source.startswith(guarded, match.start() - inner_offset):
            continue
        lines.append(source[: match.start()].count("\n") + 1)
    return lines


def test_no_task_script_expands_a_possibly_empty_array_bare() -> None:
    offenders: list[str] = []
    declared = 0
    scanned = 0
    for script in sorted(TASKS.glob("*.bash")):
        source = script.read_text(encoding="utf-8")
        if not any(flag in source for flag in _STRICT):
            continue
        scanned += 1
        for name in sorted(set(_POSSIBLY_EMPTY.findall(source))):
            declared += 1
            for line in _bare_expansions(source, name):
                offenders.append(f"{script.name}:{line} expands {name}[@] bare under set -u")

    assert offenders == [], offenders
    # Positive controls: a scan that matched no scripts, or no arrays inside
    # them, would pass while proving nothing.
    assert scanned >= 15, scanned
    assert declared >= 3, declared


def test_the_guard_recognises_the_crashing_form() -> None:
    """The detector must fire on the real shape and stay quiet on the fix."""
    broke = 'set -euo pipefail\nUV_BUNDLE_ARGS=()\nuv run "${UV_BUNDLE_ARGS[@]}" serve\n'
    fixed = 'set -euo pipefail\nUV_BUNDLE_ARGS=()\nuv run ${UV_BUNDLE_ARGS[@]+"${UV_BUNDLE_ARGS[@]}"} serve\n'

    assert _POSSIBLY_EMPTY.findall(broke) == ["UV_BUNDLE_ARGS"]
    assert _bare_expansions(broke, "UV_BUNDLE_ARGS") == [3]
    assert _bare_expansions(fixed, "UV_BUNDLE_ARGS") == []

    # Quoting is irrelevant to the crash: bash 3.2 refuses the unquoted form
    # too, so a detector that required the quotes would pass it silently.
    unquoted = "set -euo pipefail\nUV_BUNDLE_ARGS=()\nuv run ${UV_BUNDLE_ARGS[@]} serve\n"
    assert _bare_expansions(unquoted, "UV_BUNDLE_ARGS") == [3]

    # An unsafe expansion beside a guarded one is still unsafe: bash evaluates
    # the second, and a window search would have excused it.
    adjacent = 'set -euo pipefail\nA=()\nrun ${A[@]+"${A[@]}"}${A[@]}\n'
    assert _bare_expansions(adjacent, "A") == [3]

    # The guarded form alone stays quiet, so the exclusion still works.
    assert _bare_expansions('set -euo pipefail\nA=()\nrun ${A[@]+"${A[@]}"}\n', "A") == []

    # An array seeded from the task's own arguments is empty whenever the task
    # is run with none, which is how serve.bash reaches the same crash.
    from_argv = 'set -euo pipefail\nSERVER_ARGS=("$@")\nserve "${SERVER_ARGS[@]}"\n'
    assert _POSSIBLY_EMPTY.findall(from_argv) == ["SERVER_ARGS"]
