"""Tests for additional CLI command coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from faster_whisper_rocm.cli import app

runner = CliRunner()


class FakeSeg:
    """A fake segment for testing transcription output."""

    def __init__(self, start: float, end: float, text: str) -> None:
        """Initializes the FakeSeg.

        Args:
            start: The start timestamp.
            end: The end timestamp.
            text: The segment text.
        """
        self.start = start
        self.end = end
        self.text = text


class FakeInfo:
    """A fake transcription info object for testing."""

    def __init__(
        self, language: str = "en", prob: float = 0.95, duration: float | None = 3.0
    ) -> None:
        """Initializes the FakeInfo.

        Args:
            language: The detected language code.
            prob: The language probability.
            duration: The audio duration.
        """
        self.language = language
        self.language_probability = prob
        self.duration = duration


class FakeWhisperNoDuration:
    """A fake WhisperModel that returns info without a duration."""

    def __init__(
        self,
        model: str,
        *,
        device: str,
        compute_type: str,
        cpu_threads: int,
        num_workers: int,
        download_root: str | None = None,
        local_files_only: bool = False,
        device_index: int | list[int] | None = None,
    ) -> None:
        """Initializes the fake Whisper model.

        Args:
            model: The model name.
            device: The device to use.
            compute_type: The computation type.
            cpu_threads: The number of CPU threads.
            num_workers: The number of workers.
            download_root: The root directory for model downloads.
            local_files_only: Whether to only use local files.
            device_index: The device index.
        """
        self.model = model

    def transcribe(
        self,
        audio_path: str,
        **_: dict[str, Any],  # noqa: ARG002
    ) -> tuple[list[FakeSeg], FakeInfo]:
        """Runs a fake transcription, returning segments and info without duration.

        Returns:
            tuple[list[FakeSeg], FakeInfo]: The segments and info tuple.
        """
        segs = [FakeSeg(0.0, 1.0, "A"), FakeSeg(1.0, 2.0, "B")]
        info = FakeInfo("en", 0.9, None)  # duration unknown triggers spinner branch
        return segs, info


def _make_audio(tmp_path: Path) -> Path:
    """Creates a dummy audio file for testing.

    Args:
        tmp_path: The temporary directory path provided by pytest.

    Returns:
        The path to the created dummy audio file.
    """
    p = tmp_path / "audio.wav"
    p.write_bytes(b"")
    return p


def test_install_ctranslate2_with_wheel_dry_run(tmp_path: Path) -> None:
    """Tests the 'install-ctranslate2' command with a local wheel in dry-run mode."""
    wheel = tmp_path / "ctranslate2-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"fake_wheel")
    res = runner.invoke(
        app, ["install-ctranslate2", "--wheel", str(wheel), "--dry-run"]
    )  # type: ignore[list-item]
    assert res.exit_code == 0
    # Rich formatting may alter whitespace; check key markers
    assert "Command:" in res.stdout
    assert "-m pip install" in res.stdout


def test_transcribe_spinner_branch_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests that the spinner is used when duration is unknown and output is stdout."""
    monkeypatch.setattr(
        "faster_whisper_rocm.cli.app.WhisperModel", FakeWhisperNoDuration
    )
    audio = _make_audio(tmp_path)
    # stdout with default max_segments (-1) and duration None -> spinner branch
    res = runner.invoke(
        app,
        [
            "transcribe",
            str(audio),
            "--device",
            "cpu",
            "--compute-type",
            "float32",
            "--output",
            "-",
            "--output-format",
            "plain",
        ],
    )
    assert res.exit_code == 0
    assert "A" in res.stdout and "B" in res.stdout


def test_transcribe_output_path_without_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests that an output path without an extension is treated as a directory hint."""
    monkeypatch.setattr(
        "faster_whisper_rocm.cli.app.WhisperModel", FakeWhisperNoDuration
    )
    audio = _make_audio(tmp_path)
    out_path = tmp_path / "noext"  # no suffix -> treat as directory-like path
    res = runner.invoke(
        app,
        [
            "transcribe",
            str(audio),
            "--device",
            "cpu",
            "--compute-type",
            "float32",
            "--output",
            str(out_path),
            "--output-format",
            "jsonl",
            "--max-segments",
            "2",
        ],
    )
    assert res.exit_code == 0
    # With suffixless path, CLI treats it as a directory-like hint
    # and derives filename from audio basename
    dest = out_path / (audio.stem + ".jsonl")
    assert dest.exists() and dest.read_text(encoding="utf-8").strip()


def test_transcribe_missing_whisper_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests that the CLI exits gracefully if WhisperModel is not available."""
    monkeypatch.setattr("faster_whisper_rocm.cli.app.WhisperModel", None)
    audio = _make_audio(tmp_path)
    res = runner.invoke(
        app,
        [
            "transcribe",
            str(audio),
            "--device",
            "cpu",
            "--compute-type",
            "float32",
            "--output",
            "-",
        ],
    )
    assert res.exit_code != 0
