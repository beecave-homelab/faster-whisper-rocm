"""Whisper model loading utilities.

This module centralizes import and instantiation of faster-whisper's WhisperModel
so the CLI can remain thin and tests can mock in a single place.
"""

from __future__ import annotations

from typing import Any, Optional, Type

# The WhisperModel class is dynamically imported to avoid loading heavy
# dependencies at startup. This global can be used by tests for monkeypatching.
WhisperModel: Optional[Type[Any]] = None


def load_whisper_model(
    model: str,
    *,
    _WhisperModel: Optional[Type[Any]] = None,
    **kwargs: Any,
) -> Any:
    """Instantiate a WhisperModel.

    Args:
        model: Model name or local path.
        _WhisperModel: Optional override/class to instantiate (used by tests).
        **kwargs: Keyword arguments forwarded to the model constructor.

    Returns:
        An instance of the WhisperModel-compatible class.

    Raises:
        ImportError: If the faster-whisper package is not installed.
    """
    # Allow test hook to override the model implementation.
    if _WhisperModel is not None:
        return _WhisperModel(model, **kwargs)

    # Dynamic import to defer loading faster-whisper.
    try:
        from faster_whisper import WhisperModel as _FWWhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper is not installed. Please install it before using this command."
        )

    # If the test hook has patched the global, use it.
    impl = WhisperModel if WhisperModel is not None else _FWWhisperModel
    return impl(model, **kwargs)
