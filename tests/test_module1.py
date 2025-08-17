"""Additional unit tests for internal helpers in CLI."""

from __future__ import annotations

import pytest
import typer

from faster_whisper_rocm.cli.parsing import parse_key_value_options
from faster_whisper_rocm.io.timestamps import format_timestamp


def test_parse_key_value_options_parses_json_and_strings() -> None:
    """Tests that `parse_key_value_options` correctly parses mixed values."""
    parsed = parse_key_value_options(
        [
            "a=1",
            "b=true",
            'c="hi"',
            "d=foo",
        ]
    )
    assert parsed["a"] == 1
    assert parsed["b"] is True
    assert parsed["c"] == "hi"
    assert parsed["d"] == "foo"


def test_parse_key_value_options_invalid_raises() -> None:
    """Tests that `parse_key_value_options` raises an error for invalid input."""
    with pytest.raises(typer.BadParameter):
        parse_key_value_options(["oops"])  # missing '='


def test_format_timestamp_edge_cases() -> None:
    """Tests `format_timestamp` with edge cases."""
    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(1.234) == "00:00:01,234"
    assert format_timestamp(61.5) == "00:01:01,500"
    # negative clamped to 0
    assert format_timestamp(-5.0) == "00:00:00,000"
