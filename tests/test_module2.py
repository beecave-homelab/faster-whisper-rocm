"""Extended CLI tests to improve coverage for install, model-info, and transcribe."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        self, language: str = "en", prob: float = 0.95, duration: float = 3.0
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
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        device_index: Any = None,
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
        audio_path: str,  # noqa: ARG002
        *,
        language: Optional[str] = None,  # noqa: ARG002
        task: str = "transcribe",  # noqa: ARG002
        beam_size: int = 1,  # noqa: ARG002
        best_of: Optional[int] = None,  # noqa: ARG002
        patience: Optional[float] = None,  # noqa: ARG002
        length_penalty: Optional[float] = None,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
        temperature_increment_on_fallback: float = 0.2,  # noqa: ARG002
        compression_ratio_threshold: Optional[float] = None,  # noqa: ARG002
        log_prob_threshold: Optional[float] = None,  # noqa: ARG002
        no_speech_threshold: Optional[float] = None,  # noqa: ARG002
        condition_on_previous_text: bool = True,  # noqa: ARG002
        initial_prompt: Optional[str] = None,  # noqa: ARG002
        prefix: Optional[str] = None,  # noqa: ARG002
        suppress_blank: bool = True,  # noqa: ARG002
        suppress_tokens: Optional[str] = None,  # noqa: ARG002
        without_timestamps: bool = False,  # noqa: ARG002
        max_initial_timestamp: float = 1.0,  # noqa: ARG002
        word_timestamps: bool = False,  # noqa: ARG002
        prepend_punctuations: str = "“¿([{-",  # noqa: ARG002
        append_punctuations: str = "”.:;?!)}]",  # noqa: ARG002
        vad_filter: bool = True,  # noqa: ARG002
        vad_parameters: Optional[Dict[str, Any]] = None,  # noqa: ARG002
    ) -> Tuple[List[FakeSeg], FakeInfo]:
        """Runs a fake transcription, returning predefined segments and info."""
        segs = [
            FakeSeg(0.0, 1.0, "Hello"),
            FakeSeg(1.0, 2.0, "World"),
            FakeSeg(2.0, 3.0, "Again"),
        ]
        info = FakeInfo("en", 0.9, 3.0)
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


def test_install_ctranslate2_no_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that the install command exits if no ctranslate2 wheel is found."""
    # Force no wheels found
    monkeypatch.setattr("faster_whisper_rocm.cli.app.glob", lambda pattern: [])
    res = runner.invoke(app, ["install-ctranslate2"])  # type: ignore[list-item]
    assert res.exit_code != 0
    assert "No ctranslate2 wheel found" in res.stdout


def test_model_info_missing_whisper_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that `model-info` exits if WhisperModel is not available."""
    monkeypatch.setattr("faster_whisper_rocm.cli.app.WhisperModel", None)
    res = runner.invoke(
        app,
        [
            "model-info",
            "--device",
            "cpu",
            "--compute-type",
            "float32",
            "--local-files-only",
        ],
    )
    assert res.exit_code != 0


def test_model_info_with_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that `model-info` works correctly with a mocked model."""
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
            "0,1",
            "--local-files-only",
        ],
    )
    assert res.exit_code == 0
    assert "backend" in res.stdout
    assert "FakeWhisper" in res.stdout
    assert "device_index" in res.stdout


@pytest.mark.parametrize("fmt", ["plain", "jsonl", "srt", "vtt"])
def test_transcribe_stdout_formats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fmt: str
) -> None:
    """Tests transcription to stdout in various formats."""
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
            fmt,
            "--max-segments",
            "2",
            "--show-progress",
        ],
    )
    assert res.exit_code == 0
    if fmt == "plain":
        assert "Language: en" in res.stdout
        assert "Hello" in res.stdout
    elif fmt == "jsonl":
        # at least one JSON line should be present
        assert any(line.strip().startswith("{") for line in res.stdout.splitlines())
    elif fmt == "srt":
        assert "-->" in res.stdout
    elif fmt == "vtt":
        assert "WEBVTT" in res.stdout


def test_transcribe_output_dir_and_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests transcription output to both a directory and a specific file."""
    monkeypatch.setattr("faster_whisper_rocm.cli.app.WhisperModel", FakeWhisper)
    audio = _make_audio(tmp_path)

    # Directory output (with trailing slash hint)
    outdir = tmp_path / "outd"
    res1 = runner.invoke(
        app,
        [
            "transcribe",
            str(audio),
            "--device",
            "cpu",
            "--compute-type",
            "float32",
            "--output",
            str(outdir) + "/",
            "--output-format",
            "plain",
            "--max-segments",
            "1",
        ],
    )
    assert res1.exit_code == 0
    dest = outdir / (audio.stem + ".txt")
    assert dest.exists() and dest.read_text(encoding="utf-8").strip()

    # File output
    outfile = tmp_path / "x" / "y.vtt"
    res2 = runner.invoke(
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
            "vtt",
            "--max-segments",
            "1",
        ],
    )
    assert res2.exit_code == 0
    assert outfile.exists()
    assert "Saved transcript to" in res2.stdout


def test_transcribe_invalid_vad_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests that invalid VAD parameters raise an error."""
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
            "{bad",
        ],
    )
    assert res.exit_code != 0
    _out = res.stdout
    try:
        _out += res.stderr  # type: ignore[operator]
    except Exception:
        pass
    assert "Invalid JSON for --vad-parameters" in _out


def test_transcribe_unsupported_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests that an unsupported output format raises an error."""
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
        ],
    )
    assert res.exit_code != 0
    _out = res.stdout
    try:
        _out += res.stderr  # type: ignore[operator]
    except Exception:
        pass
    assert "Unsupported format" in _out


def test_transcribe_progress_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests the progress bar display logic for different scenarios."""
    monkeypatch.setattr("faster_whisper_rocm.cli.app.WhisperModel", FakeWhisper)
    audio = _make_audio(tmp_path)

    # segments_total branch (max_segments > 0)
    out1 = tmp_path / "a.txt"
    res1 = runner.invoke(
        app,
        [
            "transcribe",
            str(audio),
            "--device",
            "cpu",
            "--compute-type",
            "float32",
            "--output",
            str(out1),
            "--output-format",
            "plain",
            "--max-segments",
            "2",
            "--show-progress",
        ],
    )
    assert res1.exit_code == 0 and out1.exists()

    # duration branch (default max_segments = -1, info.duration provided)
    out2 = tmp_path / "b.jsonl"
    res2 = runner.invoke(
        app,
        [
            "transcribe",
            str(audio),
            "--device",
            "cpu",
            "--compute-type",
            "float32",
            "--output",
            str(out2),
            "--output-format",
            "jsonl",
            "--show-progress",
        ],
    )
    assert res2.exit_code == 0 and out2.exists()
