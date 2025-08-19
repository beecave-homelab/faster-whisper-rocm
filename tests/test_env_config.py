"""Tests for environment-driven configuration defaults.

Covers `faster_whisper_rocm.utils.constant` typed parsing and
`faster_whisper_rocm.utils.env_loader` idempotent loading and repo root detection.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _reload_constants(with_env: dict[str, str]) -> object:
    """Reload the constants module after setting env variables.

    Args:
        with_env: Mapping of env var names to values to apply before reload.

    Returns:
        The reloaded constants module.
    """
    # Apply envs via pytest's monkeypatch at call sites for isolation.
    import faster_whisper_rocm.utils.constant as const  # reimport for reload

    return importlib.reload(const)


def test_constants_parse_types(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure DEFAULT_* values reflect env overrides with correct types."""
    outdir = tmp_path / "outdir"
    monkeypatch.setenv("FWR_TRANSCRIBE_BEAM_SIZE", "4")
    monkeypatch.setenv("FWR_TRANSCRIBE_VAD_FILTER", "false")
    monkeypatch.setenv("FWR_TRANSCRIBE_OUTPUT_FORMAT", "jsonl")
    monkeypatch.setenv("FWR_TRANSCRIBE_OUTPUT", str(outdir))
    monkeypatch.setenv("FWR_TRANSCRIBE_OPT", "a=1,b=2")
    monkeypatch.setenv("FWR_TRANSCRIBE_DEVICE_INDEX", "")  # empty -> None

    const = _reload_constants({})  # module alias

    assert const.DEFAULT_BEAM_SIZE == 4
    assert const.DEFAULT_VAD_FILTER is False
    assert const.DEFAULT_OUTPUT_FORMAT == "jsonl"
    assert const.DEFAULT_OUTPUT == outdir
    assert const.DEFAULT_OPT == ["a=1", "b=2"]
    assert const.DEFAULT_DEVICE_INDEX is None


def test_env_loader_root_and_idempotent() -> None:
    """Cover repo root finder and verify idempotent env loading."""
    import faster_whisper_rocm.utils.env_loader as el

    # _find_repo_root should be two levels above this file.
    expected = Path(el.__file__).resolve().parents[2]
    assert el._find_repo_root() == expected

    # load_project_env should be safe to call multiple times
    el.load_project_env()
    el.load_project_env()


def test_constants_invalid_values_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid env values should fall back to defaults/None appropriately."""
    monkeypatch.setenv("FWR_TRANSCRIBE_BEST_OF", "oops")  # opt int -> None
    monkeypatch.setenv("FWR_TRANSCRIBE_PATIENCE", "bad")  # opt float -> None
    monkeypatch.setenv("FWR_TRANSCRIBE_TEMPERATURE", "nope")  # float -> default 0.0
    monkeypatch.setenv("FWR_TRANSCRIBE_CPU_THREADS", "xx")  # int -> default 0
    monkeypatch.setenv("FWR_TRANSCRIBE_SUPPRESS_BLANK", "maybe")  # bool -> default True

    import faster_whisper_rocm.utils.constant as const

    const = importlib.reload(const)

    assert const.DEFAULT_BEST_OF is None
    assert const.DEFAULT_PATIENCE is None
    assert const.DEFAULT_TEMPERATURE == 0.0
    assert const.DEFAULT_CPU_THREADS == 0
    assert const.DEFAULT_SUPPRESS_BLANK is True


def test_env_loader_find_repo_root_with_start() -> None:
    """Cover the branch passing an explicit start path to _find_repo_root."""
    import faster_whisper_rocm.utils.env_loader as el

    start = Path(el.__file__).resolve()
    expected = start.parents[2]
    assert el._find_repo_root(start) == expected
