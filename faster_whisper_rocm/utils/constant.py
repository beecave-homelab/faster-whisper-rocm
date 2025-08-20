"""Project-wide configuration constants for CLI defaults.

This module is the ONLY place that imports and invokes the environment loader.
All other modules must import constants from here instead of reading os.environ.
"""

from __future__ import annotations

import os
from pathlib import Path

from .env_loader import load_project_env

# Load .env once at import time (idempotent inside env_loader)
load_project_env()


# -----------------
# Helper converters
# -----------------


def _get_str(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else default


def _get_opt_str(key: str) -> str | None:
    val = os.getenv(key)
    return None if val is None or val == "" else val


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_opt_int(key: str) -> int | None:
    val = os.getenv(key)
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_opt_float(key: str) -> float | None:
    val = os.getenv(key)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    s = val.strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def _get_opt_path(key: str) -> Path | None:
    val = os.getenv(key)
    if val is None or val == "":
        return None
    return Path(val)


def _get_path(key: str, default: Path) -> Path:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    return Path(val)


def _get_opt_list_csv(key: str) -> list[str]:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return []
    # Split on commas, strip whitespace
    return [item.strip() for item in val.split(",") if item.strip()]


# Prefix used for all transcribe-related environment variables
ENV_PREFIX = "FWR_TRANSCRIBE_"

# ------------------
# Model init options
# ------------------
DEFAULT_MODEL: str = _get_str(f"{ENV_PREFIX}MODEL", "Systran/faster-whisper-medium")
DEFAULT_DEVICE: str = _get_str(f"{ENV_PREFIX}DEVICE", "cuda")
DEFAULT_COMPUTE_TYPE: str = _get_str(f"{ENV_PREFIX}COMPUTE_TYPE", "float16")
DEFAULT_DEVICE_INDEX: str | None = _get_opt_str(f"{ENV_PREFIX}DEVICE_INDEX")
DEFAULT_CPU_THREADS: int = _get_int(f"{ENV_PREFIX}CPU_THREADS", 0)
DEFAULT_NUM_WORKERS: int = _get_int(f"{ENV_PREFIX}NUM_WORKERS", 1)
DEFAULT_DOWNLOAD_ROOT: Path | None = _get_opt_path(f"{ENV_PREFIX}DOWNLOAD_ROOT")
DEFAULT_LOCAL_FILES_ONLY: bool = _get_bool(f"{ENV_PREFIX}LOCAL_FILES_ONLY", False)

# ----------------
# Transcribe args
# ----------------
DEFAULT_LANGUAGE: str | None = _get_opt_str(f"{ENV_PREFIX}LANGUAGE")
DEFAULT_TASK: str = _get_str(f"{ENV_PREFIX}TASK", "transcribe")
DEFAULT_BEAM_SIZE: int = _get_int(f"{ENV_PREFIX}BEAM_SIZE", 1)
DEFAULT_BEST_OF: int | None = _get_opt_int(f"{ENV_PREFIX}BEST_OF")
DEFAULT_PATIENCE: float | None = _get_opt_float(f"{ENV_PREFIX}PATIENCE")
DEFAULT_LENGTH_PENALTY: float | None = _get_opt_float(f"{ENV_PREFIX}LENGTH_PENALTY")
DEFAULT_TEMPERATURE: float = _get_float(f"{ENV_PREFIX}TEMPERATURE", 0.0)
DEFAULT_TEMPERATURE_INCREMENT_ON_FALLBACK: float = _get_float(
    f"{ENV_PREFIX}TEMPERATURE_INCREMENT_ON_FALLBACK", 0.2
)
DEFAULT_COMPRESSION_RATIO_THRESHOLD: float | None = _get_opt_float(
    f"{ENV_PREFIX}COMPRESSION_RATIO_THRESHOLD"
)
DEFAULT_LOG_PROB_THRESHOLD: float | None = _get_opt_float(
    f"{ENV_PREFIX}LOG_PROB_THRESHOLD"
)
DEFAULT_NO_SPEECH_THRESHOLD: float | None = _get_opt_float(
    f"{ENV_PREFIX}NO_SPEECH_THRESHOLD"
)
DEFAULT_CONDITION_ON_PREVIOUS_TEXT: bool = _get_bool(
    f"{ENV_PREFIX}CONDITION_ON_PREVIOUS_TEXT", True
)
DEFAULT_INITIAL_PROMPT: str | None = _get_opt_str(f"{ENV_PREFIX}INITIAL_PROMPT")
DEFAULT_PREFIX: str | None = _get_opt_str(f"{ENV_PREFIX}PREFIX")
DEFAULT_SUPPRESS_BLANK: bool = _get_bool(f"{ENV_PREFIX}SUPPRESS_BLANK", True)
DEFAULT_SUPPRESS_TOKENS: str | None = _get_opt_str(f"{ENV_PREFIX}SUPPRESS_TOKENS")
DEFAULT_WITHOUT_TIMESTAMPS: bool = _get_bool(f"{ENV_PREFIX}WITHOUT_TIMESTAMPS", False)
DEFAULT_MAX_INITIAL_TIMESTAMP: float = _get_float(
    f"{ENV_PREFIX}MAX_INITIAL_TIMESTAMP", 1.0
)
DEFAULT_WORD_TIMESTAMPS: bool = _get_bool(f"{ENV_PREFIX}WORD_TIMESTAMPS", False)
DEFAULT_PREPEND_PUNCTUATIONS: str = _get_str(
    f"{ENV_PREFIX}PREPEND_PUNCTUATIONS", "“¿([{-"
)
DEFAULT_APPEND_PUNCTUATIONS: str = _get_str(
    f"{ENV_PREFIX}APPEND_PUNCTUATIONS", "”.:;?!)}]"
)
DEFAULT_VAD_FILTER: bool = _get_bool(f"{ENV_PREFIX}VAD_FILTER", True)
DEFAULT_VAD_PARAMETERS: str | None = _get_opt_str(f"{ENV_PREFIX}VAD_PARAMETERS")

# -------------
# Output config
# -------------
DEFAULT_OUTPUT_FORMAT: str = _get_str(f"{ENV_PREFIX}OUTPUT_FORMAT", "txt")
DEFAULT_OUTPUT: Path = _get_path(f"{ENV_PREFIX}OUTPUT", Path("data/transcripts"))
DEFAULT_MAX_SEGMENTS: int = _get_int(f"{ENV_PREFIX}MAX_SEGMENTS", -1)
DEFAULT_PRINT_LANGUAGE: bool = _get_bool(f"{ENV_PREFIX}PRINT_LANGUAGE", True)
DEFAULT_PRINT_PROB: bool = _get_bool(f"{ENV_PREFIX}PRINT_PROB", True)
DEFAULT_SHOW_PROGRESS: bool = _get_bool(f"{ENV_PREFIX}SHOW_PROGRESS", True)
DEFAULT_OPT: list[str] = _get_opt_list_csv(f"{ENV_PREFIX}OPT")

# -----------------------------------------------
# Input validation: accepted audio/video extensions
# -----------------------------------------------
# Defaults use dot-prefixed, lowercase extensions.
_DEFAULT_AUDIO_EXTS: list[str] = [
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
    ".alac",
    ".mka",
]
_DEFAULT_VIDEO_EXTS: list[str] = [
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".mpeg",
    ".mpg",
    ".m4v",
    ".ts",
    ".m2ts",
    ".wmv",
    ".flv",
    ".3gp",
]

# Users may override the full set via CSV env var.
# Items may include or omit the dot; casing is normalized downstream.
DEFAULT_ACCEPTED_INPUT_EXTS: list[str] = _get_opt_list_csv(
    f"{ENV_PREFIX}ACCEPTED_INPUT_EXTS"
) or (_DEFAULT_AUDIO_EXTS + _DEFAULT_VIDEO_EXTS)

# ---------------------
# Subtitle formatting
# ---------------------
# Limits
DEFAULT_SUB_MAX_CPS: float = _get_float(f"{ENV_PREFIX}MAX_CPS", 17.0)
DEFAULT_SUB_MIN_CPS: float = _get_float(f"{ENV_PREFIX}MIN_CPS", 12.0)
DEFAULT_SUB_MAX_LINE_CHARS: int = _get_int(f"{ENV_PREFIX}MAX_LINE_CHARS", 42)
DEFAULT_SUB_MAX_LINES_PER_BLOCK: int = _get_int(f"{ENV_PREFIX}MAX_LINES_PER_BLOCK", 2)
DEFAULT_SUB_MAX_SEGMENT_DURATION_SEC: float = _get_float(
    f"{ENV_PREFIX}MAX_SEGMENT_DURATION_SEC", 5.5
)
DEFAULT_SUB_MIN_SEGMENT_DURATION_SEC: float = _get_float(
    f"{ENV_PREFIX}MIN_SEGMENT_DURATION_SEC", 1.2
)
DEFAULT_SUB_DISPLAY_BUFFER_SEC: float = _get_float(
    f"{ENV_PREFIX}DISPLAY_BUFFER_SEC", 0.2
)
# Repetition clamp: limit consecutive identical words in rendered cues
DEFAULT_SUB_MAX_CONSECUTIVE_REPEATS: int = _get_int(
    f"{ENV_PREFIX}MAX_CONSECUTIVE_REPEATS", 2
)

# Timing / gap
DEFAULT_SUB_FPS: float = _get_float(f"{ENV_PREFIX}FPS", 25.0)
DEFAULT_SUB_GAP_FRAMES: int = _get_int(f"{ENV_PREFIX}GAP_FRAMES", 2)
# Derived minimum gap (seconds)
DEFAULT_SUB_MIN_GAP_SEC: float = (
    DEFAULT_SUB_GAP_FRAMES / DEFAULT_SUB_FPS if DEFAULT_SUB_FPS > 0 else 0.0
)

# Boundary rules and heuristics
DEFAULT_SUB_BOUNDARY_CHARS: str = _get_str(f"{ENV_PREFIX}BOUNDARY_CHARS", ".?!…")
DEFAULT_SUB_CLAUSE_CHARS: str = _get_str(f"{ENV_PREFIX}CLAUSE_CHARS", ",;:")
DEFAULT_SUB_SOFT_BOUNDARY_WORDS: list[str] = _get_opt_list_csv(
    f"{ENV_PREFIX}SOFT_BOUNDARY_WORDS"
) or [
    "and",
    "but",
    "that",
    "which",
    "who",
    "where",
    "when",
    "while",
    "so",
]
DEFAULT_SUB_INTERJECTION_WHITELIST: list[str] = _get_opt_list_csv(
    f"{ENV_PREFIX}INTERJECTION_WHITELIST"
) or [
    "whoa",
    "wow",
    "what",
    "oh",
    "hey",
    "ah",
]

# Block size caps
DEFAULT_SUB_MAX_BLOCK_CHARS: int = _get_int(
    f"{ENV_PREFIX}MAX_BLOCK_CHARS",
    DEFAULT_SUB_MAX_LINE_CHARS * DEFAULT_SUB_MAX_LINES_PER_BLOCK,
)
DEFAULT_SUB_MAX_BLOCK_CHARS_SOFT: int = _get_int(
    f"{ENV_PREFIX}MAX_BLOCK_CHARS_SOFT", 90
)
