"""CLI command for inspecting Whisper model configurations.

This module provides the `run_model_info` function, which is called by the
`model-info` command in the main Typer application. It loads a Whisper model
and prints its configuration details.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from rich.console import Console

from faster_whisper_rocm.models.whisper import load_whisper_model

console = Console()


def run_model_info(
    *,
    model: str,
    device: str,
    compute_type: str,
    device_index: str | None,
    cpu_threads: int,
    num_workers: int,
    download_root: Path | None,
    local_files_only: bool,
) -> None:
    """Load a model and print its configuration.

    This function initializes a Whisper model based on the provided parameters
    and prints its configuration details to the console. It respects the test
    hook `app.WhisperModel` if set on the Typer app.

    Args:
        model: The name or local path of the Whisper model to inspect.
        device: The device to use for model loading (e.g., 'cuda', 'cpu').
        compute_type: The computation type for the model (e.g., 'float16').
        device_index: The index or a comma-separated list of device indices.
        cpu_threads: The number of CPU threads to use.
        num_workers: The number of workers for data loading.
        download_root: A custom directory to download models to.
        local_files_only: If True, only use local files and do not download.

    Raises:
        SystemExit: If the WhisperModel backend is unavailable (exit code 1).
    """
    # Resolve WhisperModel hook from the main app (for tests)
    try:
        app_mod = import_module("faster_whisper_rocm.cli.app")
        app_obj = getattr(app_mod, "app", None)
        whisper_model_hook = getattr(app_obj, "WhisperModel", None) if app_obj else None
    except (ImportError, AttributeError):  # pragma: no cover - fallback
        whisper_model_hook = None

    # If the test hook indicates WhisperModel is unavailable, abort early.
    if whisper_model_hook is None:
        console.print("WhisperModel backend is not available.", style="bold red")
        raise SystemExit(1)

    idx: object
    if device_index is None:
        idx = None
    else:
        idx = [int(x.strip()) for x in device_index.split(",") if x.strip()]
        if len(idx) == 1:
            idx = idx[0]

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
        _WhisperModel=whisper_model_hook,
        **init_kwargs,
    )

    console.print({
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "device_index": idx,
        "cpu_threads": cpu_threads,
        "num_workers": num_workers,
        "download_root": str(download_root) if download_root else None,
        "local_files_only": local_files_only,
        "backend": type(model_obj).__name__,
    })
