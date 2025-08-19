"""Tests covering additional CLI branches for model-info and transcribe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from faster_whisper_rocm.cli import app

runner = CliRunner()


class FakeSeg:
    """Represents a fake segment of transcribed audio."""

    def __init__(self, start: float, end: float, text: str) -> None:
        """Initializes the FakeSeg.

        Args:
            start: The start time of the segment.
            end: The end time of the segment.
            text: The transcribed text.
        """
        self.start = start
        self.end = end
        self.text = text


class FakeInfo:
    """Represents fake information about the transcription process."""

    def __init__(
        self, language: str = "en", prob: float = 0.95, duration: float | None = 3.0
    ) -> None:
        """Initializes the FakeInfo.

        Args:
            language: The detected language.
            prob: The language probability.
            duration: The audio duration.
        """
        self.language = language
        self.language_probability = prob
        self.duration = duration


class FakeWhisper:
    """A fake WhisperModel for testing purposes."""

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
        """Initializes the FakeWhisper model.

        Args:
            model: The model name.
            device: The device to use.
            compute_type: The compute type.
            cpu_threads: The number of CPU threads.
            num_workers: The number of workers.
            download_root: The download root directory.
            local_files_only: Whether to use local files only.
            device_index: The device index.
        """
        self.model = model
        # capture to avoid unused warnings
        self.kw = dict(
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            download_root=download_root,
            local_files_only=local_files_only,
            device_index=device_index,
        )

    def transcribe(
        self, audio_path: str, **_: dict[str, object]
    ) -> tuple[list[FakeSeg], FakeInfo]:
        """Performs fake transcription.

        Args:
            audio_path: The path to the audio file.
            **_: Additional keyword arguments.

        Returns:
            A tuple containing a list of segments and an info object.
        """
        segs = [FakeSeg(0.0, 1.0, "A"), FakeSeg(1.0, 2.0, "B")]
        info = FakeInfo("en", 0.91, 3.0)
        return segs, info


def _make_audio(tmp_path: Path) -> Path:
    """Creates a dummy audio file for testing.

    Args:
        tmp_path: The temporary path from pytest.

    Returns:
        The path to the created audio file.
    """
    p = tmp_path / "audio.wav"
    p.write_bytes(b"")
    return p


def test_model_info_single_device_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests the model-info command with a single device index."""
    monkeypatch.setattr("faster_whisper_rocm.cli.app.WhisperModel", FakeWhisper)
    res = runner.invoke(
        app,
        [
            "model-info",
            "--device",
            "cpu",
            "--compute-type",
            "float32",
            "--device-index",
            "0",
            "--local-files-only",
        ],
    )
    assert res.exit_code == 0
    out = res.stdout
    assert "backend" in out and "device_index" in out
    # Output is a Python dict printed by Rich; check presence of key and value
    assert "device_index" in out and "0" in out


def test_install_ctranslate2_run_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests the install-ctranslate2 command's success case."""
    wheel = tmp_path / "ctranslate2-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"fake_wheel")

    called: dict[str, Any] = {"args": None}

    def _fake_check_call(cmd: list[str]) -> int:
        called["args"] = cmd
        return 0

    monkeypatch.setattr("subprocess.check_call", _fake_check_call)

    res = runner.invoke(app, ["install-ctranslate2", "--wheel", str(wheel)])
    assert res.exit_code == 0
    assert called["args"] is not None
    assert "installed successfully" in res.stdout.lower()


def test_transcribe_no_progress_srt_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests transcription to an SRT file with no progress bar."""
    monkeypatch.setattr("faster_whisper_rocm.cli.app.WhisperModel", FakeWhisper)
    audio = _make_audio(tmp_path)
    outfile = tmp_path / "out.srt"
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
            str(outfile),
            "--output-format",
            "srt",
            "--no-progress",
            "--max-segments",
            "2",
        ],
    )
    assert res.exit_code == 0
    assert outfile.exists()


def test_transcribe_plain_header_prob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests plain text output with language and probability headers."""
    monkeypatch.setattr("faster_whisper_rocm.cli.app.WhisperModel", FakeWhisper)
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
            "--output-format",
            "plain",
            "--print-language",
            "--print-prob",
            "--max-segments",
            "1",
        ],
    )
    assert res.exit_code == 0
    assert "Language: en" in res.stdout
    assert "Prob:" in res.stdout
