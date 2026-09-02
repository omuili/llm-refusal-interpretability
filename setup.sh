#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PYTHON_BIN" .venv
  source .venv/bin/activate
  uv pip install -e ".[dev]"
else
  "$PYTHON_BIN" -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -e ".[dev]"
fi

python -m src.preflight --config configs/default.yaml
