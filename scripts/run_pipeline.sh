#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

python -m src.preflight --config "$CONFIG"
python -m src.prepare_data --config "$CONFIG"
python -m src.extract_activations --config "$CONFIG" --split discovery
python -m src.extract_activations --config "$CONFIG" --split validation
python -m src.extract_activations --config "$CONFIG" --split test
python -m src.discover_direction --config "$CONFIG"
python -m src.train_probe --config "$CONFIG"
python -m src.causal_eval --config "$CONFIG"
python -m src.report --config "$CONFIG"
