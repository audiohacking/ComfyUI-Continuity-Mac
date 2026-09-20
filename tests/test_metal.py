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
check("TaoMate owns the 3-step table and hides the quality stops",
      (metal.LORA_PRESETS[0]["strength"], metal.LORA_PRESETS[0].get("steps"),
       metal.LORA_PRESETS[0].get("fixed"), metal.LORA_PRESETS[0]["row"]),
      (0.8, metal.TAOMATE_STEPS, True,
       {"sampler_name": "euler", "scheduler": "simple"}))
check("Tutu still leaves draft/med/good to the family",
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

class _Accel:
    def __init__(self, **kwargs):
        self.attention = kwargs.get("attention", "default")
        self.spectrum = kwargs.get("spectrum", False)
        self.fp16_accumulation = kwargs.get("fp16_accumulation", False)

metal.refuse_mps_accel(_Accel(attention="sage"), device_type="cpu")
try:
    metal.refuse_mps_accel(_Accel(attention="sage"), device_type="mps")
except ValueError as err:
    sage_refused = str(err)
else:
    sage_refused = ""
check("MPS refuses sage attention", "NVIDIA path" in sage_refused, True)
try:
    metal.refuse_mps_accel(_Accel(attention="kitchen"), device_type="mps")
except ValueError as err:
    kitchen_refused = "NVIDIA path" in str(err)
else:
    kitchen_refused = False
check("MPS refuses kitchen int8", kitchen_refused, True)
try:
    metal.refuse_mps_accel(_Accel(spectrum=True), device_type="mps")
except ValueError as err:
    spectrum_refused = "Spectrum" in str(err)
else:
    spectrum_refused = False
check("MPS refuses Spectrum", spectrum_refused, True)
try:
    metal.refuse_mps_accel(_Accel(fp16_accumulation=True), device_type="mps")
except ValueError as err:
    accum_refused = "cuBLAS" in str(err)
else:
    accum_refused = False
check("MPS refuses fp16 accumulation", accum_refused, True)
metal.refuse_mps_accel(_Accel(attention="default"), device_type="mps")

# MiniMax H3 Attention.forward views Q as [1, S, heads, dim]. The
# AppleSilicon-FP8 fused kernel takes L from shape[-2], i.e. heads.
h3_qk = (1, 800, 40, 96)
check("H3 Q/K sequence is dim 1, not dim -2",
      metal.h3_rope_length(h3_qk), 800)
check("AppleSilicon-FP8 fused RoPE would use heads as L",
      metal.asfp8_rope_length(h3_qk), 40)
check("that mismatch is what made the VAE decode blocky noise",
      metal.asfp8_rope_length(h3_qk) != metal.h3_rope_length(h3_qk), True)
check("fused RoPE is not safe for H3's Q/K layout as-is",
      metal.fused_rope_matches_h3(h3_qk), False)
check("after RoPE, H3 attention is already [B, heads, S, dim]",
      metal.h3_attn_layout_after_rope(h3_qk), (1, 40, 800, 96))

# 56 heads, 20k packed tokens, asked for a 4096 query chunk: that would
# cross MPS's 32-bit index wall if left uncapped.
q_chunk, kv_chunk = metal.mps_attn_chunks(56, 20000, 4096, None)
check("uncapped H3 attention on a big Mac crosses MPS's 32-bit index wall",
      56 * 4096 * 20000 > 2 ** 31, True)
check("the pack's chunk cap stays under 2^30 elements",
      56 * q_chunk * kv_chunk <= metal.MPS_ATTN_ELEM_CAP, True)
check("long sequences keep full KV (no multi-KV crawl)",
      kv_chunk, 20000)
# Explicit kv request still respected, and still under the cap.
q_forced, kv_forced = metal.mps_attn_chunks(56, 20000, 4096, 20000)
check("an explicit kv chunk is honored under the cap",
      56 * q_forced * kv_forced <= metal.MPS_ATTN_ELEM_CAP, True)
# Ref2VA-scale: must not floor q at 256 (that used to shrink kv and flip
# into the checkpointed multi-KV path).
q_long, kv_long = metal.mps_attn_chunks(56, 120000, 1024, None)
check("Ref2VA-length keeps full KV", kv_long, 120000)
check("...by shrinking q below 256 rather than chopping kv",
      q_long < 256 and 56 * q_long * kv_long <= metal.MPS_ATTN_ELEM_CAP, True)

was_rope = os.environ.get(metal.ASFP8_ROPE_ENV)
was_norm = os.environ.get(metal.ASFP8_NORM_ENV)
was_low = os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")
was_high = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
try:
    os.environ[metal.ASFP8_ROPE_ENV] = "on"
    os.environ[metal.ASFP8_NORM_ENV] = "on"
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.8"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1.0"
    check("protect_h3_mps turns fused RoPE off before that pack installs",
          metal.protect_h3_mps(), "off")
    check("...and fused RMSNorm", os.environ.get(metal.ASFP8_NORM_ENV), "off")
    low, high = metal.shield_mps_watermark()
    check("MPS low watermark reclaims cache instead of holding 80%",
          low, "0.2")
    check("...and high is 0.0 on a ≥256 GB Mac or 1.0 otherwise",
          high in ("0.0", "1.0"), True)
    if metal.physical_ram_bytes() >= 256 * 1024 ** 3:
        check("a 512 GB Mac disables the false 464 GB cap", high, "0.0")
finally:
    for key, was in ((metal.ASFP8_ROPE_ENV, was_rope),
                     (metal.ASFP8_NORM_ENV, was_norm),
                     ("PYTORCH_MPS_LOW_WATERMARK_RATIO", was_low),
                     ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", was_high)):
        if was is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = was

# DeepStack guard: empty mask + matching sizes rebuilds; mismatch skips.
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    embeds = torch.zeros(1, 20, 4)
    ds = [torch.zeros(5, 4), torch.zeros(5, 4)]
    info = [{"type": "image", "index": 2, "size": 5,
             "extra": {"deepstack": ds}}]
    empty = torch.zeros(1, 20, dtype=torch.bool)
    mask, out_ds = metal._reconcile_deepstack(embeds, info, empty, ds)
    check("empty DeepStack mask rebuilds when sizes agree",
          mask is not None and int(mask.sum()) == 5 and out_ds is ds, True)
    bad_ds = [torch.zeros(7, 4)]
    mask2, out2 = metal._reconcile_deepstack(embeds, info, empty, bad_ds)
    check("size mismatch skips DeepStack instead of raising",
          mask2 is None and out2 is None, True)
    ok = torch.zeros(1, 20, dtype=torch.bool)
    ok[0, 2:7] = True
    mask3, out3 = metal._reconcile_deepstack(embeds, info, ok, ds)
    check("a matching mask is left alone",
          mask3 is ok and out3 is ds, True)

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
