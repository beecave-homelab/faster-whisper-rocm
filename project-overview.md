---
description: Repository overview and CLI architecture
---

# Project Overview

This repository provides a Typer-based Python CLI with ROCm-accelerated speech transcription via faster-whisper using a custom ROCm build of CTranslate2.

## Structure

- `pyproject.toml`
  - PDM-based project config
  - Dependencies: `typer`, `rich`, `faster-whisper`
  - Console script: `faster-whisper-rocm` → `faster_whisper_rocm/cli/app.py:app`
- `README.md`
  - Getting started, CLI usage, ROCm override instructions
- `project-overview.md` (this file)
  - Architecture and interdependencies
- `faster_whisper_rocm/`
  - `__about__.py`: version string
  - `__init__.py`: exports
  - `cli/app.py`: Typer CLI implementation
    - Commands:
      - `hello`: demo greeting
      - `install-ctranslate2`: installs ROCm CTranslate2 wheel from `out/`
      - `model-info`: prints model/device/compute configuration
      - `transcribe`: fully-featured wrapper over `faster_whisper.WhisperModel.transcribe`
  - `utils/helpers.py`: small helpers
- `tests/`
  - `test_cli.py`: basic CLI tests for help/version/hello
  - `test_asr_cli.py`: smoke tests for new commands (no heavy downloads)
- `data/samples/`
  - Example audio files (e.g., `test_long.wav`)
- `CTranslate2/`
  - Upstream CTranslate2 source and Dockerfiles (`docker_rocm/Dockerfile.rocm`)
- `out/`
  - Prebuilt wheels, including ROCm CTranslate2 wheel (e.g., `ctranslate2-3.23.0-...whl`)

## Interdependencies

- `faster-whisper` depends on `ctranslate2` for backend compute.
- To leverage ROCm, install the custom ROCm `ctranslate2` wheel from `out/` so it overrides the default version installed transitively via faster-whisper.

## CLI Details

- Entry point: `faster-whisper-rocm`
- Commands:
  - `install-ctranslate2 [--wheel PATH] [--dry-run]`
    - Finds latest `out/ctranslate2-*.whl` when `--wheel` is not specified
    - Runs: `python -m pip install --force-reinstall --no-deps <wheel>`
  - `model-info` options:
    - `--model`, `--device`, `--compute-type`, `--device-index`, `--cpu-threads`, `--num-workers`, `--download-root`, `--local-files-only`
  - `transcribe` options:
    - Model init options mirror `WhisperModel(...)`
    - Transcription options mirror `.transcribe(...)` including VAD and decoding controls
    - Output control: `--output-format plain|jsonl|srt|vtt`, `--output PATH` (default dir: `data/transcripts/`; accepts directory or file path), `--max-segments` (default: -1 = unlimited), `--print-language`, `--print-prob`, `--show-progress/--no-progress` (default: show; auto-disables on non-TTY stdout)
    - Progress behavior:
      - When `--max-segments > 0`, the progress bar is segment-based with a known total; transcription stops exactly after `max_segments` segments.
      - Otherwise, if the input duration is known, a duration-based progress bar is shown.
      - If duration is unknown, a spinner is shown as an indeterminate indicator.
    - Pass-through: `--opt key=value` (repeatable) forwards any additional supported parameter without code changes

## ROCm CTranslate2 Override Workflow

1. `pdm install`
2. `pdm run faster-whisper-rocm install-ctranslate2`
   - or specify wheel: `--wheel out/ctranslate2-<ver>.whl`
3. Verify:
   - `pdm run python -c "import ctranslate2, sys; print(ctranslate2.__version__)"`

### Optional: Vendor shared library into site-packages

For systems where the dynamic loader does not find `libctranslate2.so.3` automatically, you can vendor the ROCm library into the installed `ctranslate2` package and patch the Python extension RPATH:

- `pdm run prepare_ctranslate2_rocm [<path/to/libctranslate2.so.3>]`
  - Copies the shared lib to `site-packages/ctranslate2/.rocm_libs/`
  - Creates a SONAME symlink (e.g., `libctranslate2.so.3 -> libctranslate2.so.3.<ver>`) when needed
  - Patches `ctranslate2` extension(s) RPATH to `$ORIGIN/.rocm_libs` via `patchelf`
  - Requires `patchelf` installed on the system

Quick check after preparing:

```bash
pdm run python -c "import ctranslate2, sys; print('OK', ctranslate2.__version__, 'from', ctranslate2.__file__)"
```

## Example

```bash
pdm run faster-whisper-rocm transcribe data/samples/test_long.wav \
  --model Systran/faster-whisper-medium \
  --device cuda \
  --compute-type float16 \
  --beam-size 1 \
  --vad-filter \
  --output-format plain \
  --show-progress \
  --max-segments 10
```

Note: The default for `--max-segments` is -1 (unlimited). The example above uses 10 only for quick testing/truncation.

When `--output` is omitted, output is written to `data/transcripts/<audio_basename>.<ext>` based on `--output-format`. If `--output` points to a directory, that directory is used. If it points to a filename, the exact path is written. Progress display auto-disables when writing to stdout in non-interactive contexts to avoid corrupting output.

## Notes

- Tests avoid model downloads; heavy tests should be opt-in via env flags if added later.
- No changes are made to `CTranslate2/` source for CLI operation; only the wheel installation is required to activate ROCm.
