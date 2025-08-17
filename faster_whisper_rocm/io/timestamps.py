"""Timestamp formatting utilities for subtitle and transcript outputs."""

from __future__ import annotations


def format_timestamp(seconds: float) -> str:
    """Format seconds into SRT/VTT-style timestamp ``HH:MM:SS,mmm``.

    Negative values are clamped to 0.

    Args:
        seconds: The timestamp in seconds.

    Returns:
        A string formatted as ``HH:MM:SS,mmm``.
    """
    m, s = divmod(max(0.0, seconds), 60)
    h, m = divmod(int(m), 60)
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"
