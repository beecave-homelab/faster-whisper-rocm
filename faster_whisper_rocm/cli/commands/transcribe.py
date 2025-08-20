"""CLI command for transcribing audio files.

This module provides the `run_transcribe` function, which is called by the
`transcribe` command in the main Typer application. It handles model loading,
transcription, and output formatting.
"""

from __future__ import annotations

import inspect
import json
import sys
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from faster_whisper_rocm.cli.parsing import parse_key_value_options
from faster_whisper_rocm.io.subtitle_formatter import (
    build_cues_from_segments,
    refine_cues,
)
from faster_whisper_rocm.io.subtitle_formatter import (
    format_srt as _format_srt,
)
from faster_whisper_rocm.io.subtitle_formatter import (
    format_vtt as _format_vtt,
)
from faster_whisper_rocm.models.whisper import load_whisper_model
from faster_whisper_rocm.utils.constant import DEFAULT_ACCEPTED_INPUT_EXTS

console = Console()


def run_transcribe(
    *,
    audio_path: Path,
    # Model init options
    model: str,
    device: str,
    compute_type: str,
    device_index: str | None,
    cpu_threads: int,
    num_workers: int,
    download_root: Path | None,
    local_files_only: bool,
    # Transcribe options
    language: str | None,
    task: str,
    beam_size: int,
    best_of: int | None,
    patience: float | None,
    length_penalty: float | None,
    temperature: float,
    temperature_increment_on_fallback: float,
    compression_ratio_threshold: float | None,
    log_prob_threshold: float | None,
    no_speech_threshold: float | None,
    condition_on_previous_text: bool,
    initial_prompt: str | None,
    prefix: str | None,
    suppress_blank: bool,
    suppress_tokens: str | None,
    without_timestamps: bool,
    max_initial_timestamp: float,
    word_timestamps: bool,
    prepend_punctuations: str,
    append_punctuations: str,
    vad_filter: bool,
    vad_parameters: str | None,
    # Output
    output_format: str,
    output: Path | None,
    max_segments: int,
    print_language: bool,
    print_prob: bool,
    show_progress: bool,
    opt: list[str],
) -> None:
    """Transcribes an audio file using a Whisper model.

    This function loads a Whisper model, transcribes the given audio file, and
    outputs the result in the specified format. It supports a wide range of
    options for model initialization, transcription, and output formatting.

    Args:
        audio_path: Path to the audio file to transcribe.
        model: Name or path of the Whisper model to use.
        device: Device to use for computation (e.g., "cuda", "cpu").
        compute_type: The computation type to use for the model.
        device_index: Comma-separated list of device IDs to use.
        cpu_threads: Number of CPU threads to use per worker.
        num_workers: Number of workers to use for transcription.
        download_root: Path to a directory to download models to.
        local_files_only: If True, only use local model files.
        language: Language of the audio. If None, it will be auto-detected.
        task: Task to perform: "transcribe" or "translate".
        beam_size: Beam size for decoding.
        best_of: Number of candidates to consider from the beam.
        patience: Beam search patience factor.
        length_penalty: Length penalty for beam search.
        temperature: Temperature for sampling.
        temperature_increment_on_fallback: Temperature to increase on fallback.
        compression_ratio_threshold: Threshold for compression ratio.
        log_prob_threshold: Threshold for log probability.
        no_speech_threshold: Threshold for no speech detection.
        condition_on_previous_text: If True, condition on previous text.
        initial_prompt: Optional initial prompt for the model.
        prefix: Optional prefix for the transcription.
        suppress_blank: If True, suppress blank segments.
        suppress_tokens: Comma-separated list of tokens to suppress.
        without_timestamps: If True, do not include timestamps in the output.
        max_initial_timestamp: Maximum initial timestamp for transcription.
        word_timestamps: If True, include word-level timestamps.
        prepend_punctuations: Punctuations to prepend to the text.
        append_punctuations: Punctuations to append to the text.
        vad_filter: If True, apply VAD filter.
        vad_parameters: JSON string of VAD parameters.
        output_format: Format for the output (txt, jsonl, srt, vtt).
        output: Path to the output file. If None, prints to stdout.
        max_segments: Maximum number of segments to transcribe.
        print_language: If True, print the detected language.
        print_prob: If True, print the language probability.
        show_progress: If True, display a progress bar.
        opt: List of key-value pairs for other model options.

    Raises:
        typer.BadParameter: If ``--vad-parameters`` contains invalid JSON, or
            an unsupported ``output_format`` is provided.
    """
    # Resolve WhisperModel test hook
    try:
        app_mod = import_module("faster_whisper_rocm.cli.app")
        app_obj = getattr(app_mod, "app", None)
        whisper_model_hook = getattr(app_obj, "WhisperModel", None) if app_obj else None
    except (ImportError, AttributeError):  # pragma: no cover
        whisper_model_hook = None

    # Early input validation
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise typer.BadParameter(f"No such file: '{audio_path}'")
    if not audio_path.is_file():
        raise typer.BadParameter(f"Input path is not a file: '{audio_path}'")
    # Normalize configured extensions to dot-prefixed lowercase
    # (e.g., 'MP3' -> '.mp3', 'wav' -> '.wav')
    allowed_exts = {
        (ext if ext.startswith(".") else f".{ext}").lower()
        for ext in DEFAULT_ACCEPTED_INPUT_EXTS
        if ext
    }
    suffix = audio_path.suffix.lower()
    if suffix not in allowed_exts:
        allowed_str = ", ".join(sorted(allowed_exts))
        raise typer.BadParameter(
            "Unsupported input file type. Accepted audio/video extensions: "
            f"{allowed_str}"
        )

    # Handle device_index
    idx: object
    if device_index is None:
        idx = None
    else:
        idx = [int(x.strip()) for x in device_index.split(",") if x.strip()]
        if len(idx) == 1:
            idx = idx[0]

    # Init model
    init_kwargs: dict[str, object] = {
        "device": device,
        "compute_type": compute_type,
        "cpu_threads": cpu_threads,
        "num_workers": num_workers,
        "download_root": str(download_root) if download_root else None,
        "local_files_only": local_files_only,
    }
    if idx is not None:
        init_kwargs["device_index"] = idx

    model_obj = load_whisper_model(
        model,
        _whisper_model=whisper_model_hook,
        **init_kwargs,
    )

    # Assemble transcribe kwargs
    kwargs: dict[str, object] = {
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
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"Invalid JSON for --vad-parameters: {e}") from e

    # Merge pass-through opts
    kwargs.update(parse_key_value_options(opt))

    # Filter kwargs to match installed faster_whisper API
    sig = inspect.signature(model_obj.transcribe)
    allowed = set(sig.parameters.keys())
    filtered_kwargs = {
        k: v for k, v in kwargs.items() if k in allowed and v is not None
    }

    segments, info = model_obj.transcribe(str(audio_path), **filtered_kwargs)

    # Normalize/alias output format (backward-compatibility for legacy 'plain')
    if output_format == "plain":
        output_format = "txt"

    # Resolve destination path if output is set (default is stdout)
    dest_path: Path | None
    if output is None:
        dest_path = None
    else:
        # Decide whether 'output' is directory-like
        # (no suffix or ends with slash)
        ext_map = {"txt": ".txt", "jsonl": ".jsonl", "srt": ".srt", "vtt": ".vtt"}
        out_str = str(output)
        if out_str == "-":
            dest_path = None
        else:
            is_dir_hint = out_str.endswith(("/", "\\")) or (output.suffix == "")
            if is_dir_hint:
                out_dir = Path(out_str)
                out_dir.mkdir(parents=True, exist_ok=True)
                dest_path = out_dir / (
                    audio_path.stem + ext_map.get(output_format, ".txt")
                )
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

    # If writing to stdout in a non-interactive context, disable progress to
    # avoid corrupting output
    if dest_path is None and show_progress and not sys.stdout.isatty():
        show_progress = False

    # Output with optional progress
    duration = getattr(info, "duration", None)
    has_duration_total = (
        isinstance(duration, (int, float)) and duration and duration > 0
    )
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

        if not show_progress:
            _progress_segments = _noop  # type: ignore[assignment]
            _progress_duration = _noop  # type: ignore[assignment]
            _progress_spinner = _noop  # type: ignore[assignment]

        if output_format == "txt":
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
                console.print(f"Saved transcript to '{dest_path}'")
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
                console.print(f"Saved transcript to '{dest_path}'")
        elif output_format == "srt":
            # Collect segments first (for formatting & refinement)
            seg_list = []
            for i, seg in enumerate(segments, 1):
                if show_progress:
                    if use_segments_total:
                        _progress_segments(i)
                    elif has_duration_total:
                        _progress_duration(seg.end)
                    else:
                        _progress_spinner(i)
                seg_list.append(seg)
                count += 1
                if max_segments >= 0 and count >= max_segments:
                    break

            cues = build_cues_from_segments(seg_list)
            cues = refine_cues(cues)
            content = _format_srt(cues)

            if dest_path is None:
                console.print(content.rstrip("\n"))
            else:
                with dest_path.open("w", encoding="utf-8") as f:
                    f.write(content)
                console.print(f"Saved transcript to '{dest_path}'")
        elif output_format == "vtt":
            # Collect segments first
            seg_list = []
            for i, seg in enumerate(segments, 1):
                if show_progress:
                    if use_segments_total:
                        _progress_segments(i)
                    elif has_duration_total:
                        _progress_duration(seg.end)
                    else:
                        _progress_spinner(i)
                seg_list.append(seg)
                count += 1
                if max_segments >= 0 and count >= max_segments:
                    break

            cues = build_cues_from_segments(seg_list)
            cues = refine_cues(cues)
            content = _format_vtt(cues)

            if dest_path is None:
                console.print(content.rstrip("\n"))
            else:
                with dest_path.open("w", encoding="utf-8") as f:
                    f.write(content)
                console.print(f"Saved transcript to '{dest_path}'")
        else:
            raise typer.BadParameter(
                "Unsupported format. Choose from txt|jsonl|srt|vtt"
            )
