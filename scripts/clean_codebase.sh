#!/usr/bin/env bash
# clean_codebase.sh
# Run Ruff to auto-fix, format, and clean up the codebase.
# Usage: bash scripts/clean_codebase.sh

set -e

# Format code with Ruff
pdm run ruff format .

# Run Ruff to auto-fix lint issues
pdm run ruff check . --fix

echo "Codebase cleaned: `ruff format` and `ruff check --fix` completed."
