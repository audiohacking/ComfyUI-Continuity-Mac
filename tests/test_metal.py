"""On-disk H3 locations and the TaoMate turbo default.

    python3 tests/test_metal.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import layout  # noqa: E402
from harness import check, passed  # noqa: E402

metal = layout.load("metal").metal


check("turbo default is the TaoMate file already in models/loras",
      metal.TURBO_LORA, "taomate_h3_3step_comfy.safetensors")
check("TaoMate is the first turbo preset",
      metal.LORA_PRESETS[0]["match"], r"taomate")
check("TaoMate is 3 Euler steps at 0.8",
      (metal.LORA_PRESETS[0]["strength"], metal.LORA_PRESETS[0]["steps"],
       metal.LORA_PRESETS[0]["row"]),
      (0.8, {"draft": 3, "medium": 3, "good": 3},
       {"sampler_name": "euler", "scheduler": "simple"}))

hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
snap = metal.hf_snapshot(metal.HF_COMFY_ORG_H3, hub=hub)
# The cache tree may exist without a completed snapshot; either is fine.
check("hf_snapshot returns a directory or nothing",
      snap is None or Path(snap).is_dir(), True)

ws = metal.h3ws_models()
check("h3-ws models are visible next to this pack",
      ws is None or (ws / "loras").is_dir(), True)

roots = metal.search_roots()
check("ComfyUI models tree is a search root",
      any(str(r).endswith("ComfyUI/models") for r in roots), True)
check("Hugging Face hub snapshot is a search root when present",
      snap is None or any(Path(r) == Path(snap) for r in roots), True)

passed("all metal tests passed")
