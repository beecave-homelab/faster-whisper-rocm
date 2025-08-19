"""Install the ROCm CTranslate2 wheel, overriding the preinstalled version."""

from __future__ import annotations

import subprocess
import sys
from glob import glob
from importlib import import_module
from pathlib import Path

import typer
from rich.console import Console

console = Console()


def run_install_ctranslate2(*, wheel: Path | None, dry_run: bool) -> None:
    """Install the ROCm CTranslate2 wheel, overriding the preinstalled version.

    This function constructs and executes a `pip install` command to force the
    reinstallation of a CTranslate2 wheel. It is designed to be called from
    the CLI.

    Args:
        wheel: The file path to the CTranslate2 wheel to install. If None,
            it automatically finds the latest wheel in the `out/` directory.
        dry_run: If True, the installation command is printed to the console
            but not executed.

    Raises:
        typer.Exit: If no wheel is found in the `out/` directory when
            ``wheel`` is not provided, or if the pip installation command
            fails.
    """
    # Resolve hooks from the main app (for test monkeypatching support)
    try:
        app_mod = import_module("faster_whisper_rocm.cli.app")
        app_obj = getattr(app_mod, "app", None)
        _glob = getattr(app_obj, "glob", glob) if app_obj is not None else glob
    except (ImportError, AttributeError):  # pragma: no cover - fallback
        _glob = glob

    if wheel is None:
        matches = sorted(_glob(str(Path("out") / "ctranslate2-*.whl")))
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
    # Print on a single line to avoid Rich soft-wrapping in narrow terminals
    console.print("Command: " + " ".join(cmd), soft_wrap=True)
    if dry_run:
        return

    try:
        subprocess.check_call(cmd)
        console.print("[green]CTranslate2 installed successfully.[/green]")
    except subprocess.CalledProcessError as e:  # pragma: no cover
        console.print(f"[red]pip install failed with code {e.returncode}[/red]")
        raise typer.Exit(code=e.returncode or 1) from e
