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
check("TaoMate owns strength and row, not the quality step table",
      (metal.LORA_PRESETS[0]["strength"], metal.LORA_PRESETS[0].get("steps"),
       metal.LORA_PRESETS[0]["row"]),
      (0.8, None, {"sampler_name": "euler", "scheduler": "simple"}))
check("Tutu also leaves draft/med/good to the family",
      metal.LORA_PRESETS[2].get("steps"), None)

check("FastH3 ConvRot is a CUDA stack name",
      metal.cuda_stack_name(
          "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"),
      "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors")
check("NVFP4 text encoder is a CUDA stack name",
      bool(metal.cuda_stack_name("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")),
      True)
check("Comfy-Org pruned bf16 is not a CUDA stack name",
      metal.cuda_stack_name("minimax_h3_fl2va_pruned_bf16.safetensors"), None)


class _Weights:
    def __init__(self, picked):
        self._picked = picked

    def get(self, name):
        return self._picked.get(name)


try:
    metal.refuse_cuda_stack(_Weights({
        "fl2va": "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors",
    }))
except ValueError as err:
    refused = str(err)
else:
    refused = ""
check("queue refuses a FastH3 ConvRot pick before the sampler",
      "CUDA packed stack" in refused and "pruned_bf16" in refused, True)
metal.refuse_cuda_stack(_Weights({
    "fl2va": "minimax_h3_fl2va_pruned_bf16.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_bf16.safetensors",
}))

# MiniMax H3 Attention.forward views Q as [1, S, heads, dim]. The
# AppleSilicon-FP8 fused kernel takes L from shape[-2], i.e. heads.
h3_qk = (1, 800, 40, 96)
check("H3 Q/K sequence is dim 1, not dim -2",
      metal.h3_rope_length(h3_qk), 800)
check("AppleSilicon-FP8 fused RoPE would use heads as L",
      metal.asfp8_rope_length(h3_qk), 40)
check("that mismatch is what made the VAE decode blocky noise",
      metal.asfp8_rope_length(h3_qk) != metal.h3_rope_length(h3_qk), True)

# 56 heads, 20k packed tokens, query chunk 4096: the 512 GB Mac path.
q_chunk, kv_chunk = metal.mps_attn_chunks(56, 20000, 4096, 20000)
check("uncapped H3 attention on a big Mac crosses MPS's 32-bit index wall",
      56 * 4096 * 20000 > 2 ** 31, True)
check("the pack's chunk cap stays under 2^30 elements",
      56 * q_chunk * kv_chunk <= metal.MPS_ATTN_ELEM_CAP, True)

was_rope = os.environ.get(metal.ASFP8_ROPE_ENV)
was_norm = os.environ.get(metal.ASFP8_NORM_ENV)
try:
    os.environ[metal.ASFP8_ROPE_ENV] = "on"
    os.environ[metal.ASFP8_NORM_ENV] = "on"
    check("protect_h3_mps turns fused RoPE off before that pack installs",
          metal.protect_h3_mps(), "off")
    check("...and fused RMSNorm", os.environ.get(metal.ASFP8_NORM_ENV), "off")
finally:
    for key, was in ((metal.ASFP8_ROPE_ENV, was_rope),
                     (metal.ASFP8_NORM_ENV, was_norm)):
        if was is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = was

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
