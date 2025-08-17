"""Tests for the Whisper model loading utilities."""

from __future__ import annotations

from typing import Any, Dict

import pytest

import faster_whisper_rocm.models.whisper as mw


def test_load_whisper_model_importerror_when_impl_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that load_whisper_model raises ImportError if implementation is missing."""
    # Force WhisperModel global to None to trigger ImportError in helper
    monkeypatch.setattr(mw, "WhisperModel", None)
    with pytest.raises(ImportError):
        mw.load_whisper_model("dummy-model")


def test_load_whisper_model_uses_override_class(
    monkeypatch: pytest.MonkeyPatch,  # noqa: ARG001
) -> None:
    """Tests that load_whisper_model can use a custom WhisperModel class."""

    class Dummy:
        """A dummy class to substitute for WhisperModel."""

        def __init__(self, name: str, **kwargs: Dict[str, Any]) -> None:
            """Initializes the Dummy class.

            Args:
                name: The model name.
                **kwargs: Additional keyword arguments.
            """
            self.name = name
            self.kwargs = kwargs

    obj = mw.load_whisper_model("dummy", _WhisperModel=Dummy, device="cpu")
    assert isinstance(obj, Dummy)
    assert obj.name == "dummy" and obj.kwargs.get("device") == "cpu"
