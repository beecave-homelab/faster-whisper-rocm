"""Typer-based CLI application for `faster_whisper_rocm`.

This module provides the main Typer application instance and a few example
commands to demonstrate structure and separation of concerns. It follows the
project's coding standards and uses Google style docstrings.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import inspect
import subprocess
import sys
from glob import glob
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from contextlib import nullcontext

from ..__about__ import __version__
from ..utils.helpers import greet

try:
    # Import lazily to allow running non-ASR commands without the dependency.
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover - handled at call time
    WhisperModel = None  # type: ignore[assignment]

app = typer.Typer(help="faster_whisper_rocm command-line interface")
console = Console()


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
        version: If provided, prints version and exits.
    """
    ctx.ensure_object(dict)


@app.command()
def hello(name: str = typer.Argument(..., help="Name to greet")) -> None:
    """Greet a user by name.

    Args:
        name: The name to greet.
    """
    message = greet(name)
    console.print(message)


def _parse_key_value_options(options: List[str]) -> Dict[str, Any]:
    """Parse repeated ``--opt key=value`` pairs into a dictionary.

    Args:
        options: List of strings in the form ``key=value``.

    Returns:
        A dictionary mapping keys to parsed values (JSON if possible, else string).
    """
    parsed: Dict[str, Any] = {}
    for item in options:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid opt '{item}', expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        value_str = value.strip()
        try:
            parsed[key] = json.loads(value_str)
        except json.JSONDecodeError:
            parsed[key] = value_str
    return parsed


@app.command("install-ctranslate2")
def install_ctranslate2(
    wheel: Optional[Path] = typer.Option(
        None,
        "--wheel",
        help="Path to a CTranslate2 ROCm wheel (.whl). Defaults to newest in out/ctranslate2-*.whl",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command without running it."),
) -> None:
    """Install the ROCm CTranslate2 wheel, overriding the preinstalled version.

    This runs: ``python -m pip install --force-reinstall --no-deps <wheel>``.
    """
    if wheel is None:
        matches = sorted(glob(str(Path("out") / "ctranslate2-*.whl")))
        if not matches:
            console.print("[red]No ctranslate2 wheel found in out/ directory.[/red]")
            raise typer.Exit(code=1)
        wheel = Path(matches[-1])

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        str(wheel),
    ]

    console.print(f"Installing CTranslate2 from: [bold]{wheel}[/bold]")
    console.print("Command: " + " ".join(cmd))
    if dry_run:
        return

    try:
        subprocess.check_call(cmd)
        console.print("[green]CTranslate2 installed successfully.[/green]")
    except subprocess.CalledProcessError as e:  # pragma: no cover
        console.print(f"[red]pip install failed with code {e.returncode}[/red]")
        raise typer.Exit(code=e.returncode or 1)


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
    """Load model and print basic configuration info."""
    if WhisperModel is None:
        raise typer.Exit(
            "faster-whisper is not installed. Run 'pdm install' first, and optionally 'faster-whisper-rocm install-ctranslate2'."
        )

    idx: Any
    if device_index is None:
        idx = None
    else:
        idx = [int(x.strip()) for x in device_index.split(",") if x.strip()]
        if len(idx) == 1:
            idx = idx[0]

    _init_kwargs: Dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
        "cpu_threads": cpu_threads,
        "num_workers": num_workers,
        "download_root": str(download_root) if download_root else None,
        "local_files_only": local_files_only,
    }
    if idx is not None:
        _init_kwargs["device_index"] = idx
    model_obj = WhisperModel(
        model,
        **_init_kwargs,
    )
    # Best-effort info display based on provided settings.
    console.print(
        {
            "model": model,
            "device": device,
            "compute_type": compute_type,
            "device_index": idx,
            "cpu_threads": cpu_threads,
            "num_workers": num_workers,
            "download_root": str(download_root) if download_root else None,
            "local_files_only": local_files_only,
            "backend": type(model_obj).__name__,
        }
    )


def _format_timestamp(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    h, m = divmod(int(m), 60)
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"


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
    best_of: Optional[int] = typer.Option(None, help="Number of candidates when sampling"),
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
        help=(
            "Output file path or directory (default dir: data/transcripts). "
            "If a directory or a path without extension is provided, the file name will be derived from the audio name and output format. "
            "Use '-' to write to stdout."
        ),
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
    """Transcribe audio with faster-whisper and print formatted results."""
    if WhisperModel is None:
        raise typer.Exit(
            "faster-whisper is not installed. Run 'pdm install' first, and optionally 'faster-whisper-rocm install-ctranslate2'."
        )

    idx: Any
    if device_index is None:
        idx = None
    else:
        idx = [int(x.strip()) for x in device_index.split(",") if x.strip()]
        if len(idx) == 1:
            idx = idx[0]

    _init_kwargs: Dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
        "cpu_threads": cpu_threads,
        "num_workers": num_workers,
        "download_root": str(download_root) if download_root else None,
        "local_files_only": local_files_only,
    }
    if idx is not None:
        _init_kwargs["device_index"] = idx
    model_obj = WhisperModel(
        model,
        **_init_kwargs,
    )

    # Assemble transcribe kwargs
    kwargs: Dict[str, Any] = {
        "language": language,
        "task": task,
        "beam_size": beam_size,
        "best_of": best_of,
        "patience": patience,
        "length_penalty": length_penalty,
        "temperature": temperature,
        "temperature_increment_on_fallback": temperature_increment_on_fallback,
        "compression_ratio_threshold": compression_ratio_threshold,
        "log_prob_threshold": log_prob_threshold,
        "no_speech_threshold": no_speech_threshold,
        "condition_on_previous_text": condition_on_previous_text,
        "initial_prompt": initial_prompt,
        "prefix": prefix,
        "suppress_blank": suppress_blank,
        "suppress_tokens": suppress_tokens,
        "without_timestamps": without_timestamps,
        "max_initial_timestamp": max_initial_timestamp,
        "word_timestamps": word_timestamps,
        "prepend_punctuations": prepend_punctuations,
        "append_punctuations": append_punctuations,
        "vad_filter": vad_filter,
    }

    if vad_parameters:
        try:
            kwargs["vad_parameters"] = json.loads(vad_parameters)
        except Exception as e:
            raise typer.BadParameter(f"Invalid JSON for --vad-parameters: {e}")

    # Merge pass-through opts
    kwargs.update(_parse_key_value_options(opt))

    # Filter kwargs to match installed faster_whisper API
    sig = inspect.signature(model_obj.transcribe)
    allowed = set(sig.parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed and v is not None}

    segments, info = model_obj.transcribe(str(audio_path), **filtered_kwargs)

    # Resolve destination path if output is set (default is a directory)
    dest_path: Optional[Path]
    if output is None:
        dest_path = None
    else:
        # Decide whether 'output' is a directory-like path (no suffix or ends with slash)
        ext_map = {"plain": ".txt", "jsonl": ".jsonl", "srt": ".srt", "vtt": ".vtt"}
        out_str = str(output)
        if out_str == "-":
            dest_path = None
        else:
            is_dir_hint = out_str.endswith(("/", "\\")) or (output.suffix == "")
            if is_dir_hint:
                out_dir = Path(out_str)
                out_dir.mkdir(parents=True, exist_ok=True)
                dest_path = out_dir / (audio_path.stem + ext_map.get(output_format, ".txt"))
            else:
                dest_path = Path(output)
                dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Header (only to stdout, not into files)
    if dest_path is None and print_language:
        header = f"Language: {getattr(info, 'language', '?')}"
        if print_prob:
            prob = getattr(info, "language_probability", None)
            if isinstance(prob, (float, int)):
                header += f" Prob: {round(float(prob), 3)}"
        console.print(header)

    # If writing to stdout in a non-interactive context, disable progress to avoid corrupting output
    if dest_path is None and show_progress and not sys.stdout.isatty():
        show_progress = False

    # Output with optional progress
    duration = getattr(info, "duration", None)
    has_duration_total = isinstance(duration, (int, float)) and duration and duration > 0
    use_segments_total = max_segments > 0

    if show_progress:
        if use_segments_total:
            progress_cm = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total} segs"),
                TimeElapsedColumn(),
                transient=True,
            )
        elif has_duration_total:
            progress_cm = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
            )
        else:
            progress_cm = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            )
    else:
        progress_cm = nullcontext()

    count = 0
    with progress_cm as prog:
        if show_progress:
            if use_segments_total:
                task = prog.add_task("Transcribing", total=max_segments)

                def _progress_segments(i: int) -> None:
                    prog.update(task, completed=i)

            elif has_duration_total:
                task = prog.add_task("Transcribing", total=float(duration))

                def _progress_duration(sec: float) -> None:
                    # Clamp to total in case of minor rounding
                    prog.update(task, completed=min(float(sec), float(duration)))

            else:

                task = prog.add_task("Transcribing (segments: 0)", total=None)

                def _progress_spinner(i: int) -> None:
                    prog.update(task, description=f"Transcribing (segments: {i})")
        # No-op fallbacks when progress disabled
        def _noop(*_args: object, **_kwargs: object) -> None:
            return None

        if output_format == "plain":
            if dest_path is None:
                for i, seg in enumerate(segments, 1):
                    if show_progress:
                        if use_segments_total:
                            _progress_segments(i)
                        elif has_duration_total:
                            _progress_duration(seg.end)
                        else:
                            _progress_spinner(i)
                    console.print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")
                    count += 1
                    if max_segments >= 0 and count >= max_segments:
                        break
            else:
                with dest_path.open("w", encoding="utf-8") as f:
                    for i, seg in enumerate(segments, 1):
                        if show_progress:
                            if use_segments_total:
                                _progress_segments(i)
                            elif has_duration_total:
                                _progress_duration(seg.end)
                            else:
                                _progress_spinner(i)
                        f.write(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}\n")
                        count += 1
                        if max_segments >= 0 and count >= max_segments:
                            break
                console.print(f"Saved transcript to {dest_path}")
        elif output_format == "jsonl":
            if dest_path is None:
                for i, seg in enumerate(segments, 1):
                    if show_progress:
                        if use_segments_total:
                            _progress_segments(i)
                        elif has_duration_total:
                            _progress_duration(seg.end)
                        else:
                            _progress_spinner(i)
                    record = {"start": seg.start, "end": seg.end, "text": seg.text}
                    console.print(json.dumps(record, ensure_ascii=False))
                    count += 1
                    if max_segments >= 0 and count >= max_segments:
                        break
            else:
                with dest_path.open("w", encoding="utf-8") as f:
                    for i, seg in enumerate(segments, 1):
                        if show_progress:
                            if use_segments_total:
                                _progress_segments(i)
                            elif has_duration_total:
                                _progress_duration(seg.end)
                            else:
                                _progress_spinner(i)
                        record = {"start": seg.start, "end": seg.end, "text": seg.text}
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
                        if max_segments >= 0 and count >= max_segments:
                            break
                console.print(f"Saved transcript to {dest_path}")
        elif output_format == "srt":
            if dest_path is None:
                idx_num = 1
                for i, seg in enumerate(segments, 1):
                    if show_progress:
                        if use_segments_total:
                            _progress_segments(i)
                        elif has_duration_total:
                            _progress_duration(seg.end)
                        else:
                            _progress_spinner(i)
                    start = _format_timestamp(seg.start)
                    end = _format_timestamp(seg.end)
                    console.print(f"{idx_num}\n{start} --> {end}\n{seg.text}\n")
                    idx_num += 1
                    count += 1
                    if max_segments >= 0 and count >= max_segments:
                        break
            else:
                with dest_path.open("w", encoding="utf-8") as f:
                    idx_num = 1
                    for i, seg in enumerate(segments, 1):
                        if show_progress:
                            if use_segments_total:
                                _progress_segments(i)
                            elif has_duration_total:
                                _progress_duration(seg.end)
                            else:
                                _progress_spinner(i)
                        start = _format_timestamp(seg.start)
                        end = _format_timestamp(seg.end)
                        f.write(f"{idx_num}\n{start} --> {end}\n{seg.text}\n\n")
                        idx_num += 1
                        count += 1
                        if max_segments >= 0 and count >= max_segments:
                            break
                console.print(f"Saved transcript to {dest_path}")
        elif output_format == "vtt":
            if dest_path is None:
                console.print("WEBVTT\n")
                for i, seg in enumerate(segments, 1):
                    if show_progress:
                        if use_segments_total:
                            _progress_segments(i)
                        elif has_duration_total:
                            _progress_duration(seg.end)
                        else:
                            _progress_spinner(i)
                    start = _format_timestamp(seg.start).replace(",", ".")
                    end = _format_timestamp(seg.end).replace(",", ".")
                    console.print(f"{start} --> {end}\n{seg.text}\n")
                    count += 1
                    if max_segments >= 0 and count >= max_segments:
                        break
            else:
                with dest_path.open("w", encoding="utf-8") as f:
                    f.write("WEBVTT\n\n")
                    for i, seg in enumerate(segments, 1):
                        if show_progress:
                            if use_segments_total:
                                _progress_segments(i)
                            elif has_duration_total:
                                _progress_duration(seg.end)
                            else:
                                _progress_spinner(i)
                        start = _format_timestamp(seg.start).replace(",", ".")
                        end = _format_timestamp(seg.end).replace(",", ".")
                        f.write(f"{start} --> {end}\n{seg.text}\n\n")
                        count += 1
                        if max_segments >= 0 and count >= max_segments:
                            break
                console.print(f"Saved transcript to {dest_path}")
        else:
            raise typer.BadParameter("Unsupported format. Choose from plain|jsonl|srt|vtt")


if __name__ == "__main__":  # pragma: no cover
    app()
