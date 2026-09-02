#!/usr/bin/env bash
set -euo pipefail

echo "=== Pre-publication check ==="

# Always operate from the repository root.
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo
echo "[1/5] Checking staged filenames..."

forbidden='(^|/)(data/|outputs/|\.venv/|\.env$)|generations\.jsonl$|responses\.jsonl$|\.safetensors$|\.bin$|\.pt$|\.pth$|\.ckpt$|\.npz$'

bad="$(
  git diff --cached --name-only |
  grep -E "$forbidden" || true
)"

if [[ -n "$bad" ]]; then
  echo "ERROR: private or heavyweight files are staged:"
  echo "$bad"
  exit 1
fi

echo "PASS: no private or heavyweight files staged."


echo
echo "[2/5] Checking tracked repository files for Hugging Face tokens..."

token_hits="$(
  git grep -nEI 'hf_[A-Za-z0-9]{20,}' -- . || true
)"

if [[ -n "$token_hits" ]]; then
  echo "ERROR: possible Hugging Face token found in tracked repository files:"
  echo "$token_hits"
  exit 1
fi

echo "PASS: no Hugging Face token found in tracked repository files."


echo
echo "[3/5] Checking staged additions for Hugging Face tokens..."

staged_token_hits="$(
  git diff --cached --no-ext-diff --unified=0 |
  grep -E '^\+.*hf_[A-Za-z0-9]{20,}' || true
)"

if [[ -n "$staged_token_hits" ]]; then
  echo "ERROR: possible Hugging Face token found in staged changes:"
  echo "$staged_token_hits"
  exit 1
fi

echo "PASS: no Hugging Face token found in staged changes."


echo
echo "[4/5] Compiling Python source..."

python -m compileall -q src tests

echo "PASS: Python compilation succeeded."


echo
echo "[5/5] Running tests..."

pytest -q

echo
echo "========================================"
echo "PASS: pre-publication checks completed."
echo "========================================"
