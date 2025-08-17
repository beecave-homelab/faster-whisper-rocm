"""Tests for CLI parsing helpers in faster_whisper_rocm.cli.parsing."""

from __future__ import annotations

import pytest
from typer import BadParameter

from faster_whisper_rocm.cli.parsing import parse_key_value_options


def test_parse_key_value_options_parses_json_and_strings() -> None:
    """Tests that key-value options are parsed as JSON values where possible."""
    opts = [
        "a=1",
        "b=1.5",
        "c=true",
        "d=false",
        "e=null",
        'f="hello"',
        "g=not_json",
        'h={"k": 2}',
        "i=[1,2,3]",
    ]
    parsed = parse_key_value_options(opts)
    assert parsed["a"] == 1
    assert parsed["b"] == 1.5
    assert parsed["c"] is True
    assert parsed["d"] is False
    assert parsed["e"] is None
    assert parsed["f"] == "hello"
    assert parsed["g"] == "not_json"
    assert parsed["h"] == {"k": 2}
    assert parsed["i"] == [1, 2, 3]


def test_parse_key_value_options_raises_on_missing_equals() -> None:
    """Tests that a BadParameter is raised for options missing an '=' separator."""
    with pytest.raises(BadParameter):
        parse_key_value_options(["invalid_item"])  # type: ignore[list-item]
