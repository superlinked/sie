"""Terminal color utilities for consistent output formatting."""

from __future__ import annotations

import sys


class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    MAGENTA = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[0;37m"
    BOLD = "\033[1m"
    NC = "\033[0m"  # No Color / Reset

    @classmethod
    def disable(cls) -> None:
        """Disable colors (for non-TTY output)."""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.MAGENTA = ""
        cls.CYAN = ""
        cls.WHITE = ""
        cls.BOLD = ""
        cls.NC = ""


# Auto-disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


def log(msg: str, color: str = "") -> None:
    """Print a message with optional color.

    Args:
        msg: The message to print.
        color: Optional ANSI color code (use Colors.* constants).
    """
    if color:
        print(f"{color}{msg}{Colors.NC}")
    else:
        print(msg)


def log_step(msg: str) -> None:
    """Print a step header with visual separator.

    Args:
        msg: The step description.
    """
    separator = "=" * 60
    log(f"\n{separator}", Colors.CYAN)
    log(f"  {msg}", Colors.CYAN)
    log(separator, Colors.CYAN)


def log_success(msg: str) -> None:
    """Print a success message in green.

    Args:
        msg: The success message.
    """
    log(f"✓ {msg}", Colors.GREEN)


def log_warning(msg: str) -> None:
    """Print a warning message in yellow.

    Args:
        msg: The warning message.
    """
    log(f"⚠ {msg}", Colors.YELLOW)


def log_error(msg: str) -> None:
    """Print an error message in red.

    Args:
        msg: The error message.
    """
    log(f"✗ {msg}", Colors.RED)
