"""Project-wide environment loader.

Loads a .env file exactly once at application start. This module must only be
imported and invoked by `faster_whisper_rocm.utils.constant` according to the
project's environment variables rule.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Guard to ensure idempotent loading
_ENV_LOADED: bool = False


def _find_repo_root(start: Path | None = None) -> Path:
    """Best-effort repo root detection based on this file location.

    Defaults to two levels above this file: .../faster_whisper_rocm/ -> repo root.

    Returns:
        The detected repository root path.
    """
    if start is None:
        start = Path(__file__).resolve()
    # env_loader.py -> utils -> faster_whisper_rocm -> repo root
    return start.parents[2]


def load_project_env() -> None:
    """Load environment variables from the project's .env file once.

    This function is safe to call multiple times; it will only perform work
    on the first call. It does not override variables that are already set
    in the process environment.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    repo_root = _find_repo_root()
    dotenv_path = repo_root / ".env"

    # Load without overriding already-set environment variables
    load_dotenv(dotenv_path, override=False)

    _ENV_LOADED = True
