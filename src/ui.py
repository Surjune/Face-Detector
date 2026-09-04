"""Terminal output helpers.

Windows consoles frequently run a legacy code page that cannot encode a check
mark, and printing one there raises or prints a replacement character. The
marker is therefore chosen once from what the output stream actually supports
rather than assumed, so the same code reads correctly on a UTF-8 terminal and
on a cp1252 one.
"""

from __future__ import annotations

import sys

# Wide enough to underline the longest section heading.
RULE_WIDTH = 50

# Section bodies are indented to sit under their heading.
INDENT = "  "


def _encodable(character: str) -> bool:
    """Whether the current stdout encoding can represent a character."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        character.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


TICK = "\u2713" if _encodable("\u2713") else "+"


def section(title: str) -> None:
    """A numbered stage heading with a rule under it."""
    print()
    print(title)
    print("-" * RULE_WIDTH)


def ok(message: str) -> None:
    """A completed step."""
    print(f"{TICK} {message}")


def detail(label: str, value: str) -> None:
    """A labelled sub-item beneath a step."""
    print(f"{INDENT}- {label}: {value}")


def plain(message: str = "") -> None:
    """A line with no marker, indented to match step bodies."""
    print(f"{INDENT}{message}" if message else "")


def block(text: str) -> None:
    """A multi-line block, indented to match step bodies."""
    for line in text.splitlines():
        print(f"{INDENT}{line}")


def count(value: int) -> str:
    """Format a number with thousands separators for readability."""
    return f"{value:,}"


def shorten(value: str, keep: int = 10) -> str:
    """Abbreviate a long hex string as a prefix plus ellipsis."""
    return value if len(value) <= keep * 2 else f"{value[:keep]}...{value[-3:]}"
