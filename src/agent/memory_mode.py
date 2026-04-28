from __future__ import annotations

from enum import StrEnum


class MemoryMode(StrEnum):
    """Modes for cross-conversation memory behavior."""

    NONE = "none"
    RAW = "raw"
    SUMMARY = "summary"


DEFAULT_MEMORY_MODE = MemoryMode.NONE


def parse_memory_mode(value: str | None) -> MemoryMode:
    """Parse a memory mode string with a safe default.

    Args:
        value: User-provided memory mode value.

    Returns:
        A valid ``MemoryMode`` enum value. Unknown values fall back to ``none``.
    """

    if value is None:
        return DEFAULT_MEMORY_MODE

    normalized = value.strip().lower()
    for mode in MemoryMode:
        if normalized == mode.value:
            return mode
    return DEFAULT_MEMORY_MODE
