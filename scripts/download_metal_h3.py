#!/usr/bin/env python3
"""Download the Comfy-Org MiniMax-H3 files this Mac fork will actually run.

Not the CUDA FastH3 student, not NVFP4, not fp8. Default is pruned_bf16 DiTs
plus the bf16 text encoder and the fp16/fp32 VAEs — the ComfyUI-on-MPS quality
path on M3/M4. Pass --int8 for the pruned INT8 ConvRot DiTs (smaller; ComfyUI
emulates them on M3).

Writes into ComfyUI's model folders. Override with --models-dir, or it looks
at COMFYUI_MODELS_PATH, then ~/ComfyUI-Shared/models, then
~/Documents/ComfyUI/models.

    python3 scripts/download_metal_h3.py
    python3 scripts/download_metal_h3.py --int8
    python3 scripts/download_metal_h3.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = "Comfy-Org/MiniMax-H3"

BF16 = {
    "diffusion_models": [
        "minimax_h3_fl2va_pruned_bf16.safetensors",
        "minimax_h3_ref2va_pruned_bf16.safetensors",
    ],
    "text_encoders": ["qwen3vl_32b_minimax_h3_bf16.safetensors"],
    "vae": [
        "minimax_h3_video_vae_fp16.safetensors",
        "minimax_h3_audio_vae_fp32.safetensors",
    ],
}

INT8 = {
    "diffusion_models": [
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    ],
    "text_encoders": ["qwen3vl_32b_minimax_h3_int8_convrot.safetensors"],
    "vae": BF16["vae"],
}


def models_root(explicit):
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("COMFYUI_MODELS_PATH")
    if env:
        return Path(env).expanduser().resolve()
    shared = Path.home() / "ComfyUI-Shared" / "models"
    if shared.is_dir():
        return shared
    return Path.home() / "Documents" / "ComfyUI" / "models"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", help="ComfyUI models root")
    parser.add_argument("--int8", action="store_true",
                        help="pruned INT8 ConvRot DiTs + matching encoder")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    table = INT8 if args.int8 else BF16
    root = models_root(args.models_dir)
    print(f"repo   {REPO}")
    print(f"target {root}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None
        if not args.dry_run:
            sys.exit("huggingface_hub is not installed. "
                     "pip install huggingface_hub  (or pass --dry-run)")

    for folder, names in table.items():
        dest = root / folder
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            out = dest / name
            hf_name = f"{folder}/{name}"
            if out.exists() and out.stat().st_size > 0:
                print(f"skip   {hf_name} (already at {out})")
                continue
            print(f"{'would' if args.dry_run else 'fetch'} {hf_name} -> {out}")
            if args.dry_run:
                continue
            hf_hub_download(REPO, hf_name, local_dir=str(root))
            if not out.exists():
                sys.exit(f"download finished but {out} is missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
