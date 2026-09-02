from __future__ import annotations

import argparse
import os
import platform
import sys

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    except Exception as exc:
        print(f"PyTorch check failed: {exc}")
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except Exception as exc:
        print(f"Transformers check failed: {exc}")
    print(f"Model: {cfg['model']['id']}")
    print(f"Dataset: {cfg['dataset']['id']} (gated access may be required)")
    print(f"HF token present in environment: {bool(os.environ.get('HF_TOKEN'))}")
    print("Privacy defaults: raw prompts and raw generations are not written to results.")


if __name__ == "__main__":
    main()
