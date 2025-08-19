"""Command implementations for the faster_whisper_rocm CLI.

Each module exposes a `run_*` function that is invoked by thin Typer wrappers
in `faster_whisper_rocm/cli/app.py`.
"""

from __future__ import annotations

# Re-export run_* functions for convenience and to satisfy __all__ exports
from .install_ctranslate2 import run_install_ctranslate2
from .model_info import run_model_info
from .transcribe import run_transcribe

__all__ = ["run_install_ctranslate2", "run_model_info", "run_transcribe"]
