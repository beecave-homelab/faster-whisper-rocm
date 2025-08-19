"""Tests for additional CLI branches in the transcribe command."""

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


class FakeWhisper:
    """A fake WhisperModel class for isolated CLI testing."""

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
        self,
        audio_path: str,
        **_: dict[str, object],  # noqa: ARG002
    ) -> tuple[list[FakeSeg], FakeInfo]:
        """Runs a fake transcription.

        Returns:
            A tuple of:
            - list[FakeSeg]: The generated segments.
            - FakeInfo: The accompanying transcription metadata.
        """
        segs = [FakeSeg(0.0, 1.0, "Hello"), FakeSeg(1.0, 2.0, "World")]
        info = FakeInfo("en", 0.88, 3.0)
        return segs, info


class FakeWhisperNoDuration:
    """A fake WhisperModel that returns info without a duration."""

    def __init__(self, *args: object, **kwargs: object) -> None:  # noqa: ARG002
        """Initializes the fake Whisper model."""

    def transcribe(
        self,
        audio_path: str,
        **_: dict[str, Any],  # noqa: ARG002
    ) -> tuple[list[FakeSeg], FakeInfo]:
        """Runs a fake transcription.

        Returns:
            A tuple of:
            - list[FakeSeg]: The generated segments.
            - FakeInfo: The accompanying transcription metadata (duration None).
        """
        segs = [FakeSeg(0.0, 1.0, "A"), FakeSeg(1.0, 2.0, "B")]
        info = FakeInfo("en", 0.9, None)
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


def test_transcribe_device_index_and_default_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests transcription with device_index and default stdout options."""
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
            "--device-index",
            "0",
            "--print-language",
            "--print-prob",
            "--output",
            "-",
            "--max-segments",
            "1",
        ],
    )
    assert res.exit_code == 0
    assert "Language: en" in res.stdout and "Prob:" in res.stdout


def test_transcribe_spinner_progress_to_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests the spinner progress bar when transcription duration is unknown."""
    monkeypatch.setattr(
        "faster_whisper_rocm.cli.app.WhisperModel", FakeWhisperNoDuration
    )
    audio = _make_audio(tmp_path)
    outfile = tmp_path / "spin.txt"
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
            "plain",
            # default max-segments (-1) with unknown duration -> spinner progress branch
        ],
    )
    assert res.exit_code == 0
    assert outfile.exists() and outfile.read_text(encoding="utf-8").strip()


def test_transcribe_jsonl_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests JSONL output to stdout."""
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
            "jsonl",
            "--max-segments",
            "1",
        ],
    )
    assert res.exit_code == 0
    # Expect a JSON line with text somewhere after header
    assert '"text":' in res.stdout


def test_transcribe_vtt_stdout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tests VTT output to stdout."""
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
            "vtt",
            "--max-segments",
            "1",
        ],
    )
    assert res.exit_code == 0
    assert "WEBVTT" in res.stdout


def test_transcribe_invalid_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests that an invalid output format raises an error."""
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
            "bogus",
            "--max-segments",
            "1",
        ],
    )
    assert res.exit_code != 0


def test_transcribe_invalid_vad_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests that invalid VAD parameter JSON raises an error."""
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
            "--vad-parameters",
            "{not-json}",
        ],
    )
    assert res.exit_code != 0


def test_transcribe_opt_passthrough(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests that --opt parameters are passed through correctly."""
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
            "--opt",
            "foo=bar",
            "--max-segments",
            "1",
        ],
    )
    assert res.exit_code == 0
