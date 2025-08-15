"""Tests for `faster_whisper_rocm.utils`.

Uses pytest to validate core helpers.
"""
from faster_whisper_rocm.utils import greet


def test_greet_basic() -> None:
    assert greet("Alice") == "Hello, Alice!"


def test_greet_strips_and_defaults() -> None:
    assert greet("  Bob  ") == "Hello, Bob!"
    assert greet("") == "Hello, there!"
