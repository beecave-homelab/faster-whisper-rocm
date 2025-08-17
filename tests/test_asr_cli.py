"""ASR CLI tests (lightweight, no heavy downloads).

These tests verify that the CLI commands are wired correctly and that
non-installing operations work. They avoid running actual model downloads
or GPU execution.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

from typer.testing import CliRunner

from faster_whisper_rocm.cli import app

runner = CliRunner()


def test_transcribe_help() -> None:
    """Tests that the 'transcribe --help' command runs successfully."""
    res = runner.invoke(app, ["transcribe", "--help"])  # type: ignore[list-item]
    assert res.exit_code == 0
    assert "Output format" in res.stdout


def test_model_info_help() -> None:
    """Tests that the 'model-info --help' command runs successfully."""
    res = runner.invoke(app, ["model-info", "--help"])  # type: ignore[list-item]
    assert res.exit_code == 0
    assert "device" in res.stdout


def test_install_ctranslate2_dry_run() -> None:
    """Tests the 'install-ctranslate2 --dry-run' command."""
    # Use an explicit wheel if present; otherwise skip gracefully.
    matches = sorted(glob(str(Path("out") / "ctranslate2-*.whl")))
    if not matches:
        return  # skip if no wheel available in repo
    wheel = matches[-1]
    res = runner.invoke(
        app,
        ["install-ctranslate2", "--dry-run", "--wheel", wheel],  # type: ignore[list-item]
    )
    assert res.exit_code == 0
    assert "Installing CTranslate2" in res.stdout
