---
description: Repository overview and CLI architecture
---

# Project Overview

[![Version](https://img.shields.io/badge/Version-v0.2.0-informational)]

> Current Version: v0.2.0 (21-08-2025)

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
  - `cli/app.py`: Typer CLI entrypoint (thin). Commands are registered here and
    delegated to modules in `cli/commands/`.
  - `cli/commands/`: modular command implementations
    - `install_ctranslate2.py`
    - `model_info.py`
    - `transcribe.py`
  - `cli/parsing.py`: CLI parsing helpers (e.g., `parse_key_value_options`).
  - `models/whisper.py`: centralized model loader. Re-exports `WhisperModel` (best-effort import) and provides `load_whisper_model()` used by the CLI.
  - Commands:
    - `install-ctranslate2`: installs ROCm CTranslate2 wheel from `out/`
    - `model-info`: prints model/device/compute configuration
    - `transcribe`: fully-featured wrapper over `faster_whisper.WhisperModel.transcribe`
  - `io/timestamps.py`: formatting utilities for SRT/VTT timestamps.
  - `utils/helpers.py`: small helpers (no greeting helpers; `hello` removed)
- `tests/`
  - `test_cli.py`: basic CLI tests for help/version
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
    - Output control: `--output-format txt|jsonl|srt|vtt`, `--output PATH` (default dir: `data/transcripts/`; accepts directory or file path), `--max-segments` (default: -1 = unlimited), `--print-language`, `--print-prob`, `--show-progress/--no-progress` (default: show; auto-disables on non-TTY stdout)
    - Progress behavior:
      - When `--max-segments > 0`, the progress bar is segment-based with a known total; transcription stops exactly after `max_segments` segments.
      - Otherwise, if the input duration is known, a duration-based progress bar is shown.
      - If duration is unknown, a spinner is shown as an indeterminate indicator.
    - Pass-through: `--opt key=value` (repeatable) forwards any additional supported parameter without code changes. Parsing implemented in `faster_whisper_rocm/cli/parsing.py`.
    - Input types: Only audio/video files are accepted by extension allowlist. Supported extensions:
      - Audio: `.wav, .mp3, .m4a, .flac, .ogg, .opus, .aac, .wma, .aiff, .aif, .alac, .mka`
      - Video: `.mp4, .mkv, .mov, .avi, .webm, .mpeg, .mpg, .m4v, .ts, .m2ts, .wmv, .flv, .3gp`
      - Any other type (e.g., `.srt`, `.vtt`, `.txt`, `.pdf`) is rejected with a clear `BadParameter` error.

## Hugging Face Cache Manager (scripts/hf_models.py) — Developer Notes

This repository includes a standalone utility script to manage the local Hugging
Face cache with a polished Rich-based CLI.

- Location: `scripts/hf_models.py`
- Dependencies: `typer`, `huggingface_hub`, `rich`
- Run:

  ```bash
  # Direct
  python scripts/hf_models.py --help

  # Via PDM
  pdm run python scripts/hf_models.py list
  ```

### What it does

- `list`: Scans the effective HF cache and prints repositories.
  - Default: pretty Rich table with visible borders, auto-expanding to terminal
    width.
  - JSON mode: `--json` prints machine-readable JSON (stable field names).
  - Fields: `repo_id`, `type`, `framework` (best-effort detection), `size`,
    `nb_files`, `last_accessed`, `last_modified`, `path`, `revisions`.
  - Filters: `--repo-type model|dataset|space|all`, `--contains <substring>`.
  - Display modes:
    - Default (compact): `repo_id`, `type`, `framework`, `size`, `files`,
      `last_accessed`.
    - `--more` (extended): adds `last_modified`, `path`, and `revisions`.
    - `--less` (minimal): `repo_id`, `type`, `size`.

- `cleanup <days>`: Marks repos not accessed in the last N days.
  - Default: `--dry-run` shows a Rich table with expected freed size; does not
    delete.
  - Sorting: oldest access first.

- `remove <repo_id>`: Deletes all cached revisions for a repo (with type).
  - `--dry-run` prints what would be freed; `-y` skips confirmation.

- `cache-dir` group: Show/set/unset `HF_HUB_CACHE` in the project `.env`.
- `project` group: Show/set/unset project download root in `.env` via
  `FWR_TRANSCRIBE_DOWNLOAD_ROOT`.

### Rich styling

- Uses `box.SQUARE`, `expand=True`, and `header_style="bold"` for clear tables.
- Colors:
  - Repo type: model=cyan, dataset=magenta, space=blue (`_style_repo_type`).
  - Framework: pytorch=red, tensorflow=yellow, flax=green, onnx=bright_cyan,
    ctranslate2=bright_magenta, unknown=dim (`_style_framework`).
- Less important columns (`path`, `revisions`) are dimmed; long values fold
  (`overflow="fold"`) on `repo_id`, `path`, and `revisions`.

### Framework detection

- `_detect_model_framework(snapshot_dir)` inspects the latest snapshot directory
  for heuristics: PyTorch safetensors/bin, TensorFlow `.h5`, Flax `.msgpack`,
  ONNX `.onnx`, and CTranslate2 patterns (`model.bin` + index/vocabulary files).
  Falls back to `unknown`.

### Exit codes and scripting

- JSON mode keeps output stable for automation.
- Exit code `0` with a message when there are no items/candidates (safe for
  CI). Non-zero codes are used for validation/HTTP/IO errors in `pull`.

### Extending the table

- Add a new column in `list_cached()` and update each `table.add_row(...)` call.
- Keep lines ≤ 88 chars; prefer breaking args across lines.
- Place helpers above usage (e.g., `_style_*` before `list_cached`).

## Model Loading

- Model instantiation is delegated to `faster_whisper_rocm/models/whisper.py` via `load_whisper_model()`.
- The CLI exposes a test hook `app.WhisperModel` which defaults to the imported `WhisperModel` (or `None` if not installed). Tests can monkeypatch `faster_whisper_rocm.cli.app.WhisperModel` to inject fakes. The CLI passes this hook into the loader.

## Environment-driven configuration

- Centralized env loading happens in `faster_whisper_rocm/utils/env_loader.py` via `load_project_env()`. It is called exactly once from `faster_whisper_rocm/utils/constant.py` at import time and is idempotent.
- All transcribe CLI defaults are exposed as typed `DEFAULT_*` constants in `faster_whisper_rocm/utils/constant.py` and read from environment variables prefixed with `FWR_TRANSCRIBE_`.
- The CLI (`faster_whisper_rocm/cli/app.py`) imports these constants and wires them directly into Typer option defaults for the `transcribe` command.
- Configure defaults by creating a `.env` file at the repository root. See `.env.example` for a complete list of supported variables and their default values.

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
  --output-format txt \
  --show-progress \
  --max-segments 10
```

Note: The default for `--max-segments` is -1 (unlimited). The example above uses 10 only for quick testing/truncation.

When `--output` is omitted, output is written to `data/transcripts/<audio_basename>.<ext>` based on `--output-format`. If `--output` points to a directory, that directory is used. If it points to a filename, the exact path is written. Progress display auto-disables when writing to stdout in non-interactive contexts to avoid corrupting output.

## Notes

- Tests avoid model downloads; heavy tests should be opt-in via env flags if added later.
- No changes are made to `CTranslate2/` source for CLI operation; only the wheel installation is required to activate ROCm.
- Coverage is enabled by default with pytest-cov. After running `pdm run pytest`, open `htmlcov/index.html` for the HTML report (XML at `coverage.xml`).
