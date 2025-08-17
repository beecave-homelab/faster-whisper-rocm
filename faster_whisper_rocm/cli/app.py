"""Typer-based CLI application for `faster_whisper_rocm`.

This module provides the main Typer application instance and a few example
commands to demonstrate structure and separation of concerns. It follows the
project's coding standards and uses Google style docstrings.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from faster_whisper_rocm.__about__ import __version__
from faster_whisper_rocm.cli.commands.install_ctranslate2 import run_install_ctranslate2
from faster_whisper_rocm.cli.commands.model_info import run_model_info
from faster_whisper_rocm.cli.commands.transcribe import run_transcribe
from faster_whisper_rocm.models.whisper import WhisperModel

app = typer.Typer(help="faster_whisper_rocm command-line interface")
console = Console()

# Expose WhisperModel on the Typer app object so tests can monkeypatch
# `faster_whisper_rocm.cli.app.WhisperModel` reliably.
setattr(app, "WhisperModel", WhisperModel)
setattr(app, "glob", glob)


def version_callback(value: bool) -> None:
    """Print the package version and exit if requested.

    Args:
        value: Whether the ``--version`` flag was provided.
    """
    if value:
        console.print(f"faster_whisper_rocm {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    _version: Optional[bool] = typer.Option(  # noqa: UP007 - Optional for clarity in help
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Root command callback.

    This executes before any subcommand and handles global options.

    Args:
        ctx: Typer context object.
        _version: If provided, prints version and exits.
    """
    ctx.ensure_object(dict)


@app.command("install-ctranslate2")
def install_ctranslate2(
    wheel: Optional[Path] = typer.Option(
        None,
        "--wheel",
        help="Path to a CTranslate2 ROCm wheel (.whl). Defaults to newest in out/ctranslate2-*.whl",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show the command without running it."
    ),
) -> None:
    """Install the ROCm CTranslate2 wheel.

    This command delegates to the `run_install_ctranslate2` function to
    handle the actual installation logic.
    """
    run_install_ctranslate2(wheel=wheel, dry_run=dry_run)


@app.command("model-info")
def model_info(
    model: str = typer.Option(
        "Systran/faster-whisper-medium", "--model", help="Model name or local path"
    ),
    device: str = typer.Option("cuda", help="Device to run on (e.g., cuda, cpu)"),
    compute_type: str = typer.Option(
        "float16",
        help="Computation type: int8, int8_float16, float16, float32, etc.",
    ),
    device_index: Optional[str] = typer.Option(
        None,
        help="Device index or comma-separated list (e.g., 0 or 0,1)",
    ),
    cpu_threads: int = typer.Option(0, help="Number of CPU threads to use (0=auto)"),
    num_workers: int = typer.Option(1, help="Number of loader workers"),
    download_root: Optional[Path] = typer.Option(
        None, help="Custom download directory for models"
    ),
    local_files_only: bool = typer.Option(
        False, help="Only use local files, do not attempt downloads"
    ),
) -> None:
    """Print model configuration.

    Delegates to the `run_model_info` function to load a model and display
    its configuration details.
    """
    run_model_info(
        model=model,
        device=device,
        compute_type=compute_type,
        device_index=device_index,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        download_root=download_root,
        local_files_only=local_files_only,
    )


# Timestamp formatting now lives in faster_whisper_rocm/io/timestamps.py


@app.command("transcribe")
def transcribe(
    audio_path: Path = typer.Argument(..., exists=True, help="Path to audio file"),
    # Model init options
    model: str = typer.Option(
        "Systran/faster-whisper-medium", "--model", help="Model name or local path"
    ),
    device: str = typer.Option("cuda", help="Device to run on (e.g., cuda, cpu)"),
    compute_type: str = typer.Option(
        "float16",
        help="Computation type: int8, int8_float16, float16, float32, etc.",
    ),
    device_index: Optional[str] = typer.Option(
        None,
        help="Device index or comma-separated list (e.g., 0 or 0,1)",
    ),
    cpu_threads: int = typer.Option(0, help="Number of CPU threads to use (0=auto)"),
    num_workers: int = typer.Option(1, help="Number of loader workers"),
    download_root: Optional[Path] = typer.Option(
        None, help="Custom download directory for models"
    ),
    local_files_only: bool = typer.Option(
        False, help="Only use local files, do not attempt downloads"
    ),
    # Transcribe options
    language: Optional[str] = typer.Option(None, help="Language code (e.g., en)"),
    task: str = typer.Option("transcribe", help="Task: transcribe or translate"),
    beam_size: int = typer.Option(1, help="Beam size for beam search"),
    best_of: Optional[int] = typer.Option(
        None, help="Number of candidates when sampling"
    ),
    patience: Optional[float] = typer.Option(None, help="Beam search patience"),
    length_penalty: Optional[float] = typer.Option(None, help="Length penalty"),
    temperature: float = typer.Option(0.0, help="Temperature for sampling"),
    temperature_increment_on_fallback: float = typer.Option(
        0.2, help="Temperature increase on fallback"
    ),
    compression_ratio_threshold: Optional[float] = typer.Option(
        None, help="Compression ratio threshold"
    ),
    log_prob_threshold: Optional[float] = typer.Option(
        None, help="Average log prob threshold"
    ),
    no_speech_threshold: Optional[float] = typer.Option(
        None, help="No speech probability threshold"
    ),
    condition_on_previous_text: bool = typer.Option(
        True, help="Condition on previous text"
    ),
    initial_prompt: Optional[str] = typer.Option(
        None, help="Optional initial prompt to condition the model"
    ),
    prefix: Optional[str] = typer.Option(None, help="Optional prefix text"),
    suppress_blank: bool = typer.Option(True, help="Suppress blank outputs"),
    suppress_tokens: Optional[str] = typer.Option(
        None, help="Tokens to suppress (e.g., -1 or comma-separated list)"
    ),
    without_timestamps: bool = typer.Option(False, help="Disable timestamps"),
    max_initial_timestamp: float = typer.Option(1.0, help="Max initial timestamp"),
    word_timestamps: bool = typer.Option(False, help="Enable word timestamps"),
    prepend_punctuations: str = typer.Option(
        "“¿([{-", help="Punctuations to prepend to next word"
    ),
    append_punctuations: str = typer.Option(
        "”.:;?!)}]", help="Punctuations to append to previous word"
    ),
    vad_filter: bool = typer.Option(True, help="Enable VAD filtering"),
    vad_parameters: Optional[str] = typer.Option(
        None, help="JSON dict for VAD parameters"
    ),
    # Output control
    output_format: str = typer.Option(
        "plain", help="Output format: plain|jsonl|srt|vtt"
    ),
    output: Optional[Path] = typer.Option(
        Path("data/transcripts"),
        "--output",
        help="Output file path or directory. Defaults to 'data/transcripts'. If a dir or path without extension is given, the filename is derived from the audio. Use '-' for stdout.",
    ),
    max_segments: int = typer.Option(
        -1, help="Maximum number of segments to print (-1 for all)"
    ),
    print_language: bool = typer.Option(True, help="Print detected language"),
    print_prob: bool = typer.Option(True, help="Print language probability"),
    show_progress: bool = typer.Option(
        True,
        "--show-progress/--no-progress",
        help="Show live progress (number of segments processed)",
    ),
    opt: List[str] = typer.Option(
        [], "--opt", help="Extra transcribe options as key=value (repeatable)"
    ),
) -> None:
    """Transcribe an audio file.

    This command uses a Whisper model to transcribe the given audio file,
    handling everything from model loading to output formatting. It delegates
    the core logic to the `run_transcribe` function.
    """
    run_transcribe(
        audio_path=audio_path,
        model=model,
        device=device,
        compute_type=compute_type,
        device_index=device_index,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        download_root=download_root,
        local_files_only=local_files_only,
        language=language,
        task=task,
        beam_size=beam_size,
        best_of=best_of,
        patience=patience,
        length_penalty=length_penalty,
        temperature=temperature,
        temperature_increment_on_fallback=temperature_increment_on_fallback,
        compression_ratio_threshold=compression_ratio_threshold,
        log_prob_threshold=log_prob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
        prefix=prefix,
        suppress_blank=suppress_blank,
        suppress_tokens=suppress_tokens,
        without_timestamps=without_timestamps,
        max_initial_timestamp=max_initial_timestamp,
        word_timestamps=word_timestamps,
        prepend_punctuations=prepend_punctuations,
        append_punctuations=append_punctuations,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        output_format=output_format,
        output=output,
        max_segments=max_segments,
        print_language=print_language,
        print_prob=print_prob,
        show_progress=show_progress,
        opt=opt,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
