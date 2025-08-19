"""Typer-based CLI application for `faster_whisper_rocm`.

This module provides the main Typer application instance and a few example
commands to demonstrate structure and separation of concerns. It follows the
project's coding standards and uses Google style docstrings.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from faster_whisper_rocm.__about__ import __version__
from faster_whisper_rocm.cli.commands.install_ctranslate2 import run_install_ctranslate2
from faster_whisper_rocm.cli.commands.model_info import run_model_info
from faster_whisper_rocm.cli.commands.transcribe import run_transcribe
from faster_whisper_rocm.models.whisper import WhisperModel
from faster_whisper_rocm.utils import constant as const

app = typer.Typer(help="faster_whisper_rocm command-line interface")
console = Console()

# Expose hooks on the Typer app object so tests can monkeypatch reliably
app.WhisperModel = WhisperModel
app.glob = glob


def version_callback(value: bool) -> None:
    """Print the package version and exit if requested.

    Args:
        value: Whether the ``--version`` flag was provided.

    Raises:
        typer.Exit: If the version was requested and the program should
            terminate after printing it.
    """
    if value:
        console.print(f"faster_whisper_rocm {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    _version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            is_flag=True,
            callback=version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = None,  # noqa: UP007 - Optional for clarity in help
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
    wheel: Annotated[
        Path | None,
        typer.Option(
            "--wheel",
            help=(
                "Path to a CTranslate2 ROCm wheel (.whl). "
                "Defaults to newest in out/ctranslate2-*.whl"
            ),
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            is_flag=True,
            help="Show the command without running it.",
        ),
    ] = False,
) -> None:
    """Install the ROCm CTranslate2 wheel.

    This command delegates to the `run_install_ctranslate2` function to
    handle the actual installation logic.
    """
    run_install_ctranslate2(wheel=wheel, dry_run=dry_run)


@app.command("model-info")
def model_info(
    model: Annotated[
        str,
        typer.Option("--model", help="Model name or local path"),
    ] = "Systran/faster-whisper-medium",
    device: Annotated[
        str,
        typer.Option(help="Device to run on (e.g., cuda, cpu)"),
    ] = "cuda",
    compute_type: Annotated[
        str,
        typer.Option(
            help=("Computation type: int8, int8_float16, float16, float32, etc.")
        ),
    ] = "float16",
    device_index: Annotated[
        str | None,
        typer.Option(help="Device index or comma-separated list (e.g., 0 or 0,1)"),
    ] = None,
    cpu_threads: Annotated[
        int,
        typer.Option(help="Number of CPU threads to use (0=auto)"),
    ] = 0,
    num_workers: Annotated[
        int,
        typer.Option(help="Number of loader workers"),
    ] = 1,
    download_root: Annotated[
        Path | None,
        typer.Option(help="Custom download directory for models"),
    ] = None,
    local_files_only: Annotated[
        bool,
        typer.Option(help="Only use local files, do not attempt downloads"),
    ] = False,
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
    audio_path: Annotated[Path, typer.Argument(exists=True, help="Path to audio file")],
    # Model init options
    model: Annotated[
        str,
        typer.Option("--model", help="Model name or local path"),
    ] = const.DEFAULT_MODEL,
    device: Annotated[
        str,
        typer.Option(help="Device to run on (e.g., cuda, cpu)"),
    ] = const.DEFAULT_DEVICE,
    compute_type: Annotated[
        str,
        typer.Option(
            help="Computation type: int8, int8_float16, float16, float32, etc.",
        ),
    ] = const.DEFAULT_COMPUTE_TYPE,
    device_index: Annotated[
        str | None,
        typer.Option(help="Device index or comma-separated list (e.g., 0 or 0,1)"),
    ] = const.DEFAULT_DEVICE_INDEX,
    cpu_threads: Annotated[
        int,
        typer.Option(help="Number of CPU threads to use (0=auto)"),
    ] = const.DEFAULT_CPU_THREADS,
    num_workers: Annotated[
        int,
        typer.Option(help="Number of loader workers"),
    ] = const.DEFAULT_NUM_WORKERS,
    download_root: Annotated[
        Path | None,
        typer.Option(help="Custom download directory for models"),
    ] = const.DEFAULT_DOWNLOAD_ROOT,
    local_files_only: Annotated[
        bool,
        typer.Option(help="Only use local files, do not attempt downloads"),
    ] = const.DEFAULT_LOCAL_FILES_ONLY,
    # Transcribe options
    language: Annotated[
        str | None,
        typer.Option(help="Language code (e.g., en)"),
    ] = const.DEFAULT_LANGUAGE,
    task: Annotated[
        str,
        typer.Option(help="Task: transcribe or translate"),
    ] = const.DEFAULT_TASK,
    beam_size: Annotated[
        int,
        typer.Option(help="Beam size for beam search"),
    ] = const.DEFAULT_BEAM_SIZE,
    best_of: Annotated[
        int | None,
        typer.Option(help="Number of candidates when sampling"),
    ] = const.DEFAULT_BEST_OF,
    patience: Annotated[
        float | None,
        typer.Option(help="Beam search patience"),
    ] = const.DEFAULT_PATIENCE,
    length_penalty: Annotated[
        float | None,
        typer.Option(help="Length penalty"),
    ] = const.DEFAULT_LENGTH_PENALTY,
    temperature: Annotated[
        float,
        typer.Option(help="Temperature for sampling"),
    ] = const.DEFAULT_TEMPERATURE,
    temperature_increment_on_fallback: Annotated[
        float,
        typer.Option(help="Temperature increase on fallback"),
    ] = const.DEFAULT_TEMPERATURE_INCREMENT_ON_FALLBACK,
    compression_ratio_threshold: Annotated[
        float | None,
        typer.Option(help="Compression ratio threshold"),
    ] = const.DEFAULT_COMPRESSION_RATIO_THRESHOLD,
    log_prob_threshold: Annotated[
        float | None,
        typer.Option(help="Average log prob threshold"),
    ] = const.DEFAULT_LOG_PROB_THRESHOLD,
    no_speech_threshold: Annotated[
        float | None,
        typer.Option(help="No speech probability threshold"),
    ] = const.DEFAULT_NO_SPEECH_THRESHOLD,
    condition_on_previous_text: Annotated[
        bool,
        typer.Option(help="Condition on previous text"),
    ] = const.DEFAULT_CONDITION_ON_PREVIOUS_TEXT,
    initial_prompt: Annotated[
        str | None,
        typer.Option(help="Optional initial prompt to condition the model"),
    ] = const.DEFAULT_INITIAL_PROMPT,
    prefix: Annotated[
        str | None,
        typer.Option(help="Optional prefix text"),
    ] = const.DEFAULT_PREFIX,
    suppress_blank: Annotated[
        bool,
        typer.Option(help="Suppress blank outputs"),
    ] = const.DEFAULT_SUPPRESS_BLANK,
    suppress_tokens: Annotated[
        str | None,
        typer.Option(help="Tokens to suppress (e.g., -1 or comma-separated list)"),
    ] = const.DEFAULT_SUPPRESS_TOKENS,
    without_timestamps: Annotated[
        bool,
        typer.Option(help="Disable timestamps"),
    ] = const.DEFAULT_WITHOUT_TIMESTAMPS,
    max_initial_timestamp: Annotated[
        float,
        typer.Option(help="Max initial timestamp"),
    ] = const.DEFAULT_MAX_INITIAL_TIMESTAMP,
    word_timestamps: Annotated[
        bool,
        typer.Option(help="Enable word timestamps"),
    ] = const.DEFAULT_WORD_TIMESTAMPS,
    prepend_punctuations: Annotated[
        str,
        typer.Option(help="Punctuations to prepend to next word"),
    ] = const.DEFAULT_PREPEND_PUNCTUATIONS,
    append_punctuations: Annotated[
        str,
        typer.Option(help="Punctuations to append to previous word"),
    ] = const.DEFAULT_APPEND_PUNCTUATIONS,
    vad_filter: Annotated[
        bool,
        typer.Option(help="Enable VAD filtering"),
    ] = const.DEFAULT_VAD_FILTER,
    vad_parameters: Annotated[
        str | None,
        typer.Option(
            help=(
                "JSON dict for VAD parameters. "
                "See https://github.com/snakers4/silero-vad#parameters for details."
            ),
        ),
    ] = const.DEFAULT_VAD_PARAMETERS,
    # Output control
    output_format: Annotated[
        str,
        typer.Option(help="Output format: plain|jsonl|srt|vtt"),
    ] = const.DEFAULT_OUTPUT_FORMAT,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help=(
                "Output file path or directory. Defaults to 'data/transcripts'. "
                "If a dir or path without extension is given, the filename is "
                "derived from the audio. Use '-' for stdout."
            ),
        ),
    ] = const.DEFAULT_OUTPUT,
    max_segments: Annotated[
        int,
        typer.Option(help="Maximum number of segments to print (-1 for all)"),
    ] = const.DEFAULT_MAX_SEGMENTS,
    print_language: Annotated[
        bool,
        typer.Option(help="Print detected language"),
    ] = const.DEFAULT_PRINT_LANGUAGE,
    print_prob: Annotated[
        bool,
        typer.Option(help="Print language probability"),
    ] = const.DEFAULT_PRINT_PROB,
    show_progress: Annotated[
        bool,
        typer.Option(
            "--show-progress/--no-progress",
            help=("Show live progress (number of segments processed)"),
        ),
    ] = const.DEFAULT_SHOW_PROGRESS,
    opt: Annotated[
        list[str],
        typer.Option(
            "--opt",
            help=("Extra transcribe options as key=value (repeatable)"),
        ),
    ] = const.DEFAULT_OPT,
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
