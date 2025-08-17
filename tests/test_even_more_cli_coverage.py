from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from typer.testing import CliRunner

from faster_whisper_rocm.cli import app, app as app_mod

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
        self, language: str = "en", prob: float = 0.95, duration: Optional[float] = 3.0
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
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        device_index: Any = None,
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
        self, audio_path: str, **_: Dict[str, Any]
    ) -> Tuple[List[FakeSeg], FakeInfo]:
        """Performs fake transcription.

        Args:
            audio_path: The path to the audio file.
            **_: Additional keyword arguments.

        Returns:
            A tuple containing a list of segments and an info object.
        """
        segs = [FakeSeg(0.0, 1.0, "X"), FakeSeg(1.0, 2.0, "Y")]
        info = FakeInfo("en", 0.92, 2.0)
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


def test_transcribe_jsonl_file_duration_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests transcription to a JSONL file with duration-based progress."""
    monkeypatch.setattr("faster_whisper_rocm.cli.app.WhisperModel", FakeWhisper)
    audio = _make_audio(tmp_path)
    outfile = tmp_path / "out.jsonl"
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
            "jsonl",
        ],
    )
    assert res.exit_code == 0
    text = outfile.read_text(encoding="utf-8").strip()
    assert "\n" in text and '"text"' in text


def test_transcribe_plain_file_with_segments_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests transcription to a plain file with segment-based progress."""
    monkeypatch.setattr("faster_whisper_rocm.cli.app.WhisperModel", FakeWhisper)
    audio = _make_audio(tmp_path)
    outfile = tmp_path / "out.txt"
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
            "--max-segments",
            "2",
        ],
    )
    assert res.exit_code == 0
    text = outfile.read_text(encoding="utf-8").strip()
    assert "[0.00s -> 1.00s]" in text


def test_install_ctranslate2_dry_run_autofind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests the install-ctranslate2 command with wheel auto-finding."""
    # Create a fake wheel and make app.glob return it
    wheel = tmp_path / "ctranslate2-9.9.9-cp310-cp310-linux_x86_64.whl"
    wheel.write_bytes(b"fake")

    # Set the app.glob hook used by the command module
    setattr(app, "glob", lambda pattern: [str(wheel)])

    res = runner.invoke(app, ["install-ctranslate2", "--dry-run"])  # type: ignore[list-item]
    assert res.exit_code == 0
    assert "Installing CTranslate2 from:" in res.stdout


def test_model_info_multi_device_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests the model-info command with multiple device indices."""
    # Use the FakeWhisper above for instantiation
    setattr(app_mod, "WhisperModel", FakeWhisper)
    res = runner.invoke(
        app,
        [
            "model-info",
            "--device",
            "cpu",
            "--compute-type",
            "float32",
            "--device-index",
            "0,1",
            "--local-files-only",
        ],
    )
    assert res.exit_code == 0
    out = res.stdout
    assert "device_index" in out and "0" in out and "1" in out


def test_transcribe_plain_file_duration_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests transcription to a plain file with duration-based progress."""
    monkeypatch.setattr("faster_whisper_rocm.cli.app.WhisperModel", FakeWhisper)
    audio = _make_audio(tmp_path)
    outfile = tmp_path / "out2.txt"
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
            # no max-segments: should take duration-based progress branch
        ],
    )
    assert res.exit_code == 0
    assert outfile.exists()
