# Versions Changelog

- [v0.2.0 (Current) - 21-08-2025](#v020-current---21-08-2025)
- [v0.1.0 - Initial Release](#v010---initial-release)

---

## **v0.2.0** (Current) - *21-08-2025*

### ✨ Brief Description

Minor release with GPU architecture/Docker Compose support and CLI enhancements.

### ✨ New Features in v0.2.0

- Added: GPU architecture support and Docker Compose configuration

### 🔧 Improvements in v0.2.0

- Refactored: Updated transcribe command output format to `txt`
- Updated: Transcribe model options and introduced Hugging Face model manager utility
- Maintenance: Removed deprecated `CTranslate2` subproject from the repository

### 🐛 Bug Fixes in v0.2.0

- None

### 📝 Key Commits in v0.2.0

`131af5f` chore 📦: Remove CTranslate2 subproject due to deprecation

`d83a2c7` chore 📦: Update transcribe model options and add HF model manager

`34a3f97` docs 📝: Update transcribe command output format to txt

`3375acd` refactor ♻️: Update output format to txt in transcribe command

`d4acbf4` feat ✨: Add GPU architecture support and Docker Compose configuration

---

## **v0.1.0** - *Initial Release*

### 🎉 Brief Description

Initial release of `faster-whisper-rocm` CLI with Typer-based commands and
project structure, packaging, tests, and documentation.

### ✨ Features in v0.1.0

- Typer CLI entrypoint `faster-whisper-rocm`
- Commands: `transcribe`, `model-info`, `install-ctranslate2`
- Environment-driven defaults via `.env` with typed constants
- Basic tests and project scaffolding

### 📝 Notes

- Version set in `pyproject.toml` and `faster_whisper_rocm/__about__.py` as `0.1.0`.
- Release date not recorded in git history window; can be added later if needed.

---
