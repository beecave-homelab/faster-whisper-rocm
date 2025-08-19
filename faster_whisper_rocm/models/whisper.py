"""Whisper model loading utilities.

This module centralizes import and instantiation of faster-whisper's WhisperModel
so the CLI can remain thin and tests can mock in a single place.
"""

from __future__ import annotations

# The WhisperModel implementation is resolved at import time if available.
# Tests may monkeypatch this global to simulate availability or to inject fakes.
WhisperModel: type[object] | None = None

try:  # Best-effort import; keep None if the package is not installed
    from faster_whisper import WhisperModel as _FWWhisperModel  # type: ignore

    WhisperModel = _FWWhisperModel
except Exception:  # pragma: no cover - environment dependent
    WhisperModel = None


def load_whisper_model(
    model: str,
    *,
    _whisper_model: type[object] | None = None,
    **kwargs: object,
) -> object:
    """Instantiate a WhisperModel.

    Args:
        model: Model name or local path.
        _whisper_model: Optional override/class to instantiate (used by tests).
        **kwargs: Keyword arguments forwarded to the model constructor.

    Returns:
        An instance of the WhisperModel-compatible class.

    Raises:
        ImportError: If the faster-whisper package is not installed.
    """
    # Allow test hook to override the model implementation.
    if _whisper_model is not None:
        return _whisper_model(model, **kwargs)

    # Use the resolved global implementation when available.
    if WhisperModel is None:
        raise ImportError(
            "WhisperModel implementation is not available. Ensure "
            "'faster-whisper' is installed, or provide a custom implementation "
            "via the _whisper_model parameter."
        )

    return WhisperModel(model, **kwargs)
