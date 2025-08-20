# Python CLI Package Boilerplate (PDM + Typer)

A modern Python CLI package boilerplate using PDM for package management and Typer for CLI functionality. The package is organized into clearly separated submodules (e.g., `faster_whisper_rocm/cli/`, `faster_whisper_rocm/utils/`) and follows the coding rules in `docs/python-coding-standards.md` with Google style docstrings.

## Versions

**Current version**: 0.1.0

## Table of Contents

- [Versions](#versions)
- [Badges](#badges)
- [Repository Contents](#repository-contents)
- [Getting Started (PDM)](#getting-started-pdm)
- [Faster-Whisper ROCm CLI](#faster-whisper-rocm-cli)
- [License](#license)
- [Contributing](#contributing)

## Badges

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Repository Contents

- **Python Coding Standards**: see `docs/python-coding-standards.md`.
- **Package**: `faster_whisper_rocm/` with submodules:
  - `faster_whisper_rocm/cli/` Typer app and commands
  - `faster_whisper_rocm/utils/` reusable helpers
  - `faster_whisper_rocm/__about__.py` version metadata
  - `faster_whisper_rocm/__init__.py` top-level exports
- **Tests**: `tests/`
- **Build/Deps**: `pyproject.toml` managed by PDM
- **CLI Entry Point**: `faster-whisper-rocm` console script → `faster_whisper_rocm/cli/app.py:app`
- **Project Overview**: see `project-overview.md`

## Getting Started (PDM)

Prerequisite: install [PDM](https://pdm.fming.dev)

```bash
python3 -m pip install -U pdm
```

Install dependencies and set up a local venv:

```bash
pdm install
```

Run the CLI (installed console script):

```bash
pdm run faster-whisper-rocm --help
pdm run faster-whisper-rocm --version
```

Run tests:

```bash
pdm run pytest
```

Coverage:

- Enabled by default via pytest-cov (branch + line).
- HTML report: `htmlcov/index.html`, XML: `coverage.xml`.
- Threshold: fail under 85% (see `pyproject.toml` `[tool.pytest.ini_options].addopts`).
- Exclusions: system-specific script `faster_whisper_rocm/cli/prepare_ctranslate2_rocm.py` is omitted from coverage.

Code quality:

```bash
pdm run ruff check .
pdm run black .
```

Notes:

- Legacy files like `setup.py` and `requirements.txt` are no longer used with PDM.
- Entry point is defined in `pyproject.toml` under `[project.scripts]` as `faster-whisper-rocm`.

## Environment configuration (.env)

All optional flags of the `transcribe` command can be configured via environment variables. A centralized loader reads a `.env` file at startup and typed defaults are exposed from `faster_whisper_rocm/utils/constant.py`.

- Prefix: `FWR_TRANSCRIBE_`
- Example file: `.env.example` (copy to `.env` and edit)
- CLI flags always override env defaults

Quick start:

```bash
cp .env.example .env
# edit .env, e.g.:
echo 'FWR_TRANSCRIBE_DEVICE=cpu' >> .env
echo 'FWR_TRANSCRIBE_OUTPUT_FORMAT=jsonl' >> .env
```

These map directly to Typer defaults in `faster_whisper_rocm/cli/app.py`. See `.env.example` for the full list and descriptions.

## Faster-Whisper ROCm CLI

The CLI wraps `faster-whisper` with ROCm-enabled CTranslate2. It exposes a fully featured `transcribe` command and helpers.

1. Ensure dependencies are installed:

  ```bash
  pdm install
  ```

1. Override CTranslate2 with the ROCm wheel provided in `out`:

  ```bash
  # Installs the newest out/ctranslate2-*.whl into the active environment
  pdm run faster-whisper-rocm install-ctranslate2

  # Or specify a wheel explicitly
  pdm run faster-whisper-rocm install-ctranslate2 --wheel out/ctranslate2-3.23.0-cp310-cp310-linux_x86_64.whl
  ```

1. Transcribe audio (txt, jsonl, srt, or vtt output):

  ```bash
  pdm run faster-whisper-rocm transcribe data/samples/test_long.wav \
    --model Systran/faster-whisper-medium \
    --device cuda \
    --compute-type float16 \
    --beam-size 1 \
    --vad-filter \
    --output-format txt \
    --max-segments 10
  ```

Note:

- The default for `--max-segments` is -1 (unlimited). The example above uses 10 only for quick testing/truncation.
- The new `--output` option defaults to the directory `data/transcripts/`. If you omit `--output`, the transcript is saved as `data/transcripts/<audio_basename>.<ext>` based on `--output-format`. Pass a directory to choose a different folder, or pass a full file path to write to that exact path. Use `--output -` to write to stdout.
- Progress display is enabled by default via `--show-progress/--no-progress`. It auto-disables when writing to stdout in non-interactive contexts (e.g., piping) to avoid corrupting output.
- Progress behavior:
  - When `--max-segments > 0`, the progress bar is segment-based with a known total. Transcription stops exactly after `max_segments` segments.
  - Otherwise, if the input duration is known, a duration-based progress bar is shown.
  - If duration is unknown, a spinner is shown as an indeterminate indicator.

- Accepted input types: Only audio/video files by extension. Audio: `.wav, .mp3, .m4a, .flac, .ogg, .opus, .aac, .wma, .aiff, .aif, .alac, .mka`. Video: `.mp4, .mkv, .mov, .avi, .webm, .mpeg, .mpg, .m4v, .ts, .m2ts, .wmv, .flv, .3gp`. Other types (e.g., `.srt`, `.vtt`, `.txt`, `.pdf`) are rejected with a clear error.

Common options (subset; use `--help` for full list):

```bash
  --language en                       # language code
  --task transcribe|translate         # task type
  --best-of 5                         # sampling candidates
  --temperature 0.0                   # sampling temperature
  --condition-on-previous-text / --no-condition-on-previous-text
  --initial-prompt "..."
  --word-timestamps                   # word-level timestamps
  --vad-filter                        # enable VAD
  --vad-parameters '{"min_silence_duration_ms": 500}'
  --output-format txt|jsonl|srt|vtt # output format
  --output PATH                       # file or directory (default: data/transcripts/); use '-' for stdout
  --max-segments -1                   # number of segments to print; -1 = unlimited (default)
  --show-progress / --no-progress     # live segment count while transcribing (default: show)
  --opt key=value                     # pass-through extra options (repeatable)
```

Inspect model/device settings:

```bash
pdm run faster-whisper-rocm model-info --model Systran/faster-whisper-medium --device cuda --compute-type float16
```

## Hugging Face Cache Manager (Rich output)

A lightweight CLI at `scripts/hf_models.py` helps manage your local Hugging Face
cache with pretty, colorized tables using Rich. It auto-expands to your terminal
width, shows visible borders, and color-codes repo type and detected framework.

Quick examples:

```bash
# List cached repos (compact default view)
python scripts/hf_models.py list

# Minimal columns (repo_id, type, size)
python scripts/hf_models.py list --less

# Full details (all columns)
python scripts/hf_models.py list --more

# JSON output for scripting
python scripts/hf_models.py list --json

# Preview cleanup candidates older than 30 days (no deletion)
python scripts/hf_models.py cleanup 30 --dry-run
```

Notes:

- Filter by type: `--repo-type model|dataset|space|all`
- Default view is compact; use `--more` for full details, or `--less` for the
  smallest table.
- Env-based cache resolution precedence: `HF_HUB_CACHE` > `HF_HOME/hub` >
  `XDG_CACHE_HOME/huggingface/hub` > `~/.cache/huggingface/hub`

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE) for more information.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
