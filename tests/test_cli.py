"""CLI tests for `faster_whisper_rocm` using Typer's CliRunner."""

from typer.testing import CliRunner

from faster_whisper_rocm.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    """Tests the main CLI --help flag."""
    result = runner.invoke(app, ["--help"])  # type: ignore[arg-type]
    assert result.exit_code == 0
    assert "faster_whisper_rocm command-line interface" in result.stdout


def test_cli_version() -> None:
    """Tests the main CLI --version flag."""
    result = runner.invoke(app, ["--version"])  # type: ignore[arg-type]
    assert result.exit_code == 0
    assert "faster_whisper_rocm" in result.stdout
