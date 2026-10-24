"""Interpreter contract for readers of the immutable UTF-8 experiment files."""

import os
import sys


def ensure_utf8():
    """Restart before artifact I/O; retain interpreter flags, CLI arguments and exit behavior."""
    if sys.flags.utf8_mode:
        return
    arguments = list(sys.orig_argv)
    arguments[0] = sys.executable
    # The final len(sys.argv) entries are the program and its arguments. Keep
    # -m/-c/-- with that program, after interpreter options such as -W and -X.
    position = len(arguments) - len(sys.argv)
    if position > 1 and arguments[position - 1] in {"-m", "-c", "--"}:
        position -= 1
    # UTF-8 mode is chosen during preinitialization: an earlier -X utf8=0
    # must be replaced, not merely followed by another -X utf8.
    options = iter(arguments[1:position])
    flags = []
    for option in options:
        if option == "-X":
            value = next(options)
            flags.extend((option, "utf8" if value.partition("=")[0] == "utf8" else value))
        elif option in {"-W", "--check-hash-based-pycs"}:
            flags.extend((option, next(options)))
        else:
            flags.append("-Xutf8" if option.startswith("-Xutf8=") else option)
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *flags, *arguments[position:]])
