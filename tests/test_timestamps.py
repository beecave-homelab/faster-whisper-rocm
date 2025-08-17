"""Tests for timestamp formatting in faster_whisper_rocm.io.timestamps."""

from __future__ import annotations

from faster_whisper_rocm.io.timestamps import format_timestamp


def test_format_timestamp_zero() -> None:
    """Tests that a timestamp of zero is formatted correctly."""
    assert format_timestamp(0.0) == "00:00:00,000"


def test_format_timestamp_positive_seconds() -> None:
    """Tests that positive timestamps are formatted correctly."""
    assert format_timestamp(1.234) == "00:00:01,234"
    assert format_timestamp(61.5) == "00:01:01,500"


def test_format_timestamp_negative_clamped() -> None:
    """Tests that negative timestamps are clamped to zero."""
    assert format_timestamp(-2.0) == "00:00:00,000"


def test_format_timestamp_truncates_ms() -> None:
    """Tests that millisecond part of the timestamp is truncated, not rounded."""
    # Ensure we don't round up to the next second
    assert format_timestamp(59.999) == "00:00:59,999"
