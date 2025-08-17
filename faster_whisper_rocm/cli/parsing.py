"""CLI parsing helpers for faster_whisper_rocm.

This module contains small, testable utilities used by Typer commands,
kept separate from command implementations for better maintainability.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import typer


def parse_key_value_options(options: List[str]) -> Dict[str, Any]:
    """Parse repeated ``--opt key=value`` pairs into a dictionary.

    Attempts JSON parsing on values; falls back to raw strings when JSON fails.

    Args:
        options: List of strings in the form ``key=value``.

    Returns:
        A dictionary mapping keys to parsed values.

    Raises:
        typer.BadParameter: If an item does not contain an '=' character.
    """
    parsed: Dict[str, Any] = {}
    for item in options:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid opt '{item}', expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        value_str = value.strip()
        try:
            parsed[key] = json.loads(value_str)
        except json.JSONDecodeError:
            parsed[key] = value_str
    return parsed
