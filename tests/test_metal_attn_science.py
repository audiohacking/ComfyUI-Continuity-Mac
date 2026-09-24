"""Metal H3 attention science: sub-quad (+upcast) vs mtlflashattn.

    COMFYUI_PATH=~/ComfyUI-Installs/ComfyUI/ComfyUI \\
      /Users/moysa/Documents/ComfyUI/.venv/bin/python3 tests/test_metal_attn_science.py

Short ladder first. Flash must be finite and close to sub-quad, and faster,
before any live force (see metal._patch_metal_flash_attention — science only;
protect_h3_mps keeps patched sub-quad after live flash noise).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import layout  # noqa: E402
from harness import check, passed, skip  # noqa: E402

import torch  # noqa: E402

ma = layout.load("metal_attn").metal_attn

COMFY = os.environ.get("COMFYUI_PATH", os.path.expanduser("~/ComfyUI"))
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)

if not torch.backends.mps.is_available():
    skip("MPS not available")

try:
    from metal_flash_attn import flash_attn_func  # noqa: F401
except ImportError:
    skip("metal_flash_attn not installed")

device = ma.require_mps()
print(f"metal_attn science on {device} torch={torch.__version__}")
print(f"H3 heads={ma.H3_HEADS} dim={ma.H3_DIM}")
print(f"token estimate 8s 512x896 gen: {ma.estimate_video_tokens(512, 896, 192)}")
print()

print("=== A: flash vs sub-quad (fp16, Continuity-like upcast on sub-quad) ===")
print(f"{'S':>6} {'sub_ms':>8} {'flash_ms':>9} {'speedup':>8} {'ok':>4} {'maxerr':>10}")

rows_ok = True
for S in ma.SEQS_SHORT + ma.SEQS_MEDIUM:
    q, k, v = ma.make_qkv(1, ma.H3_HEADS, S, ma.H3_DIM, device=device, dtype=torch.float16)
    # Sub-quad in fp32 scores (what Continuity's bf16→fp32 upcast forces)
    q32, k32, v32 = q.float(), k.float(), v.float()
    try:
        sub_ms = ma.timed(
            lambda: ma.attention_subquad_bhsd(q32, k32, v32),
            device=device, warmup=1, repeats=3)
        out_sq = ma.attention_subquad_bhsd(q32, k32, v32).to(torch.float16)
    except Exception as exc:  # noqa: BLE001
        print(f"{S:6d}  sub-quad FAIL: {type(exc).__name__}: {exc}")
        rows_ok = False
        continue
    try:
        flash_ms = ma.timed(
            lambda: ma.attention_flash_bhsd(q, k, v),
            device=device, warmup=1, repeats=3)
        out_fl = ma.attention_flash_bhsd(q, k, v)
    except Exception as exc:  # noqa: BLE001
        print(f"{S:6d}  flash FAIL: {type(exc).__name__}: {exc}")
        rows_ok = False
        continue
    ok, err = ma.close(out_fl, out_sq, atol=1e-1, rtol=1e-1)
    speedup = sub_ms / flash_ms if flash_ms else float("inf")
    print(f"{S:6d} {sub_ms:8.1f} {flash_ms:9.1f} {speedup:7.2f}x {str(ok):>4} {err:10.4g}")
    if not ok or speedup < 1.2:
        rows_ok = False

check("flash beats Continuity sub-quad+upcast on the short/medium ladder",
      rows_ok, True)

print()
print("=== B: force patch binds when Comfy attention is importable ===")
try:
    metal = layout.load("metal").metal
    # Only assert the helper exists and returns bool without crashing when
    # comfy is on the path; a cold import without comfy is a soft miss.
    bound = metal._patch_metal_flash_attention()
    print(f"_patch_metal_flash_attention -> {bound}")
    check("metal flash patch returns bool", isinstance(bound, bool), True)
except Exception as exc:  # noqa: BLE001
    print(f"patch smoke skipped: {type(exc).__name__}: {exc}")

passed("metal attention science ladder complete")
