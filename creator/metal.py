"""Apple Metal extras for this Mac fork: where H3 files live, and turbo.

Continuity still drives ComfyUI core. On this machine the packed files sit in
ComfyUI's own `models/` tree, may also exist under the Hugging Face hub cache,
and LoRAs may live in the sibling h3-ws checkout. `register_search_paths`
adds every one of those that exists so a picker sees the file once, wherever
it landed.

Turbo on this fork still offers TaoMate by filename. That file is a 3-step
distill, so its preset owns the count (one number, no draft/med/good
spread). LightX2V, Tutu and PDD keep their own tables.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
COMFY_MODELS = Path.home() / "Documents" / "ComfyUI" / "models"
SHARED_MODELS = Path.home() / "ComfyUI-Shared" / "models"

# Packed Comfy-Org MiniMax-H3 as huggingface_hub stores it.
HF_COMFY_ORG_H3 = "models--Comfy-Org--MiniMax-H3"

TURBO_LORA = "taomate_h3_3step_comfy.safetensors"
# TaoMate was distilled at 3 NFE. One count, so the quality stops hide —
# three buttons writing the same 3 would be VDN's "fixed" case again.
TAOMATE_STEPS = {"draft": 3, "medium": 3, "good": 3}

# Guess-time exclusions. VAEs share a folder. DiT / CLIP needles keep the
# CUDA packed stacks out of an empty-node guess: h3-ws runs native
# MiniMax-H3 FL2VA, and ComfyUI's equivalent is Comfy-Org pruned_bf16 plus
# the bf16 Qwen encoder. FastH3 INT8 ConvRot + NVFP4 load, then the first
# MPS sampler step never returns.
SLOT_AVOID = {
    "vae": [r"t1[_-]?image", r"image[_-]vae", r"audio"],
    "dit": [r"fasth3", r"fastvideo", r"nvfp4", r"convrot", r"fp8"],
    "clip": [r"nvfp4", r"convrot", r"fp8", r"awq"],
}

# Same files, refused at queue so a remembered CUDA pick cannot hang Metal.
_CUDA_STACK = re.compile(
    r"fasth3|fastvideo|nvfp4|convrot|fp8[_-]?scaled", re.IGNORECASE)

# Filename presets the turbo switch matches. Same shape as the original
# lightx2v card: strength, shifts, and a row when the file was distilled
# against one. `steps` only when that file was distilled against a count —
# LightX2V and Tutu leave draft / med / good to the family; TaoMate is a
# 3-step distill so it owns the table and `fixed` hides the three stops.
LORA_PRESETS = (
    {"match": r"taomate", "strength": 0.8,
     "shift_video": 12, "shift_audio": 3,
     "row": {"sampler_name": "euler", "scheduler": "simple"},
     "steps": TAOMATE_STEPS, "fixed": True,
     "note": "TaoMate is a 3-step distill; the quality stops would all write 3."},
    {"match": "lightx2v", "strength": 0.6, "shift_video": 6, "shift_audio": 3},
    {"match": r"tutu|20to8-nfe|20to8_nfe", "strength": 0.8,
     "shift_video": 12, "shift_audio": 3,
     "row": {"sampler_name": "euler", "scheduler": "beta"}},
)


def hf_snapshot(repo_dirname, hub=None):
    """Current snapshot dir for a `models--org--name` cache tree, or None."""
    root = Path(hub or HF_HUB) / repo_dirname
    ref = root / "refs" / "main"
    if ref.is_file():
        snap = root / "snapshots" / ref.read_text().strip()
        if snap.is_dir():
            return snap
    snaps = root / "snapshots"
    if not snaps.is_dir():
        return None
    kids = [p for p in snaps.iterdir() if p.is_dir()]
    if not kids:
        return None
    return max(kids, key=lambda p: p.stat().st_mtime)


def h3ws_models():
    """Sibling h3-ws/models, when this pack lives next to that checkout."""
    here = Path(__file__).resolve()
    # creator/metal.py -> repo root -> ../h3-ws/models
    sibling = here.parents[1].parent / "h3-ws" / "models"
    return sibling if sibling.is_dir() else None


def search_roots():
    """Every directory this fork will ask ComfyUI to search, if it exists.

    Order is preference for *new* files, not exclusivity: ComfyUI lists the
    union. Shared is last so a leftover there still shows after a move into
    the ComfyUI models tree.
    """
    roots = []
    if COMFY_MODELS.is_dir():
        roots.append(COMFY_MODELS)
    snap = hf_snapshot(HF_COMFY_ORG_H3)
    if snap is not None:
        roots.append(snap)
    ws = h3ws_models()
    if ws is not None:
        roots.append(ws)
    if SHARED_MODELS.is_dir():
        roots.append(SHARED_MODELS)
    return roots


def _add(kind, path):
    path = os.path.expanduser(str(path))
    if not os.path.isdir(path):
        return False
    import folder_paths
    folder_paths.add_model_folder_path(kind, path)
    return True


def register_search_paths():
    """Point ComfyUI at every H3 location this machine actually has.

    Safe to call more than once: `add_model_folder_path` appends a root it
    does not already have. Missing folders are skipped.
    """
    added = []
    for root in search_roots():
        for kind in ("diffusion_models", "text_encoders", "vae", "vae_approx",
                     "loras", "checkpoints"):
            if _add(kind, Path(root) / kind):
                added.append(f"{kind}:{Path(root) / kind}")
    return added


def cuda_stack_name(filename):
    """The CUDA packed filename, or None if this is a Metal-safe pick."""
    name = os.path.basename(str(filename or ""))
    return name if name and _CUDA_STACK.search(name) else None


def refuse_cuda_stack(weights):
    """Refuse FastH3 / NVFP4 / ConvRot / fp8 before the sampler starts.

    h3-ws's working path is native MiniMax-H3 FL2VA (not the FastH3 student)
    with TaoMate on top. Continuity still drives ComfyUI core, so the
    matching files are Comfy-Org `*_pruned_bf16` and `qwen3vl_*_bf16`.
    """
    for name in ("fl2va", "ref2va", "clip"):
        filename = weights.get(name) if hasattr(weights, "get") else None
        hit = cuda_stack_name(filename)
        if not hit:
            continue
        raise ValueError(
            f"{hit} is a CUDA packed stack (FastH3 student, NVFP4, ConvRot or "
            f"fp8). h3-ws runs native MiniMax-H3 FL2VA with TaoMate, not that "
            f"file. In the weights pill pick "
            f"minimax_h3_fl2va_pruned_bf16.safetensors / "
            f"minimax_h3_ref2va_pruned_bf16.safetensors and "
            f"qwen3vl_32b_minimax_h3_bf16.safetensors."
        )


# ComfyUI-AppleSilicon-FP8's fused RoPE kernel takes the sequence length from
# x.shape[-2]. That is correct for Flux/WAN [B, heads, L, dim]. MiniMax H3
# Q/K is [B, S, heads, dim], so -2 is heads. The kernel then indexes the
# frequency table by head, every token gets the same few rotations, and the
# VAE decodes blocky noise. The env is what stops that pack installing;
# disarm undoes it if the pack already loaded.
#
# After RoPE, H3 transposes Q/K to [B, heads, S, dim] before
# optimized_attention(skip_reshape=True). Dense pytorch SDPA would see the
# right layout and still be wrong: it materializes QK and aborted at ~383 GB
# on a long packed sequence. Sub-quadratic with the patches below is the
# working path. mtlflashattn is the next speed bet — it never forms QK and
# already expects BHSD — but only once a fallback cannot reach stock SDPA.
ASFP8_ROPE_ENV = "ASFP8_ROPE_FAST"
ASFP8_ROPE_TAG = "[AppleSilicon-FP8/rope-fast]"
ASFP8_NORM_ENV = "ASFP8_FUSED_NORM"
ASFP8_NORM_TAG = "[AppleSilicon-FP8/fused-norm]"

# MPS uses 32-bit indexing. Above this many QK elements the score matrix
# silently corrupts (Comfy-Org/ComfyUI#14837). 512 GB unified memory makes
# sub-quadratic skip chunking because "there is plenty of RAM".
MPS_ATTN_ELEM_CAP = 2 ** 30

# Continuity accelerators that are CUDA/Triton. On MPS they are noise or a
# hang; `default` attention, chunked FFN, and the step caches still run.
NVIDIA_ATTENTION = ("sage", "kitchen", "sla")


def asfp8_rope_length(q_shape):
    """Sequence length the AppleSilicon-FP8 fused kernel would use for Q/K."""
    return int(q_shape[-2])


def h3_rope_length(q_shape):
    """H3 packed-token sequence length (the S in [B, S, heads, dim])."""
    return int(q_shape[1])


def h3_attn_layout_after_rope(q_shape):
    """Q/K as skip_reshape SDPA sees it: [B, heads, S, dim] after the transpose."""
    batch, seq, heads, dim = q_shape
    return (int(batch), int(heads), int(seq), int(dim))


def fused_rope_matches_h3(q_shape):
    """True only if the fused kernel's L is already the packed sequence."""
    return asfp8_rope_length(q_shape) == h3_rope_length(q_shape)


def torch_device_type():
    """ComfyUI's compute device, or '' when core is not loaded (tests)."""
    try:
        import comfy.model_management as mm
        return mm.get_torch_device().type
    except Exception:
        return ""


def refuse_mps_accel(settings, device_type=None):
    """Refuse NVIDIA-only Continuity accelerators when the device is MPS.

    Sage, kitchen int8, SLA and Spectrum are CUDA/Triton. fp16 accumulation
    is a cuBLAS flag. Chunked FFN and the step caches stay — they do not
    swap the attention kernel.
    """
    kind = torch_device_type() if device_type is None else device_type
    if kind != "mps":
        return
    attention = getattr(settings, "attention", "default")
    if attention in NVIDIA_ATTENTION:
        raise ValueError(
            f"attention {attention!r} is an NVIDIA path (sage / kitchen int8 / "
            f"SLA). On Apple GPU leave attention on 'default' — that is the "
            f"Metal-safe H3 path this pack forces."
        )
    if getattr(settings, "spectrum", False):
        raise ValueError(
            "Spectrum is an NVIDIA accelerator. Leave it off on Apple GPU."
        )
    if getattr(settings, "fp16_accumulation", False):
        raise ValueError(
            "fp16 accumulation is a CUDA cuBLAS flag. Leave it off on Apple GPU."
        )


def _disarm_asfp8(tag):
    import sys
    for mod in list(sys.modules.values()):
        if getattr(mod, "TAG", None) != tag:
            continue
        undo = getattr(mod, "uninstall_for_test", None)
        if undo is None:
            return False
        undo()
        return True
    return False


def disarm_asfp8_rope():
    """Undo a fused-RoPE install that already happened this process."""
    return _disarm_asfp8(ASFP8_ROPE_TAG)


def disarm_asfp8_fused_norm():
    """Undo a fused-RMSNorm install. H3's rms_rope goes through F.rms_norm."""
    return _disarm_asfp8(ASFP8_NORM_TAG)


def _patch_sub_quad_empty_to_zeros():
    """ComfyUI#15804: MPS baddbmm(beta=0) still reads torch.empty NaNs."""
    try:
        from comfy.ldm.modules import sub_quadratic_attention as sq
        import torch
    except ImportError:
        return False
    if getattr(sq, "_continuity_mps_zeros", False):
        return True

    def _zeros_empty(fn):
        def wrapped(*args, **kwargs):
            real = torch.empty
            torch.empty = torch.zeros
            try:
                return fn(*args, **kwargs)
            finally:
                torch.empty = real
        wrapped.__name__ = getattr(fn, "__name__", "wrapped")
        return wrapped

    sq._summarize_chunk = _zeros_empty(sq._summarize_chunk)
    sq._get_attention_scores_no_kv_chunking = _zeros_empty(
        sq._get_attention_scores_no_kv_chunking)
    sq._continuity_mps_zeros = True
    return True


def _patch_bf16_attention_upcast():
    """macOS black-image workaround only mapped fp16; H3 is bf16 (#15804)."""
    try:
        import torch
        import comfy.model_management as mm
    except ImportError:
        return False
    if getattr(mm, "_continuity_bf16_upcast", False):
        return True
    orig = mm.force_upcast_attention_dtype

    def force_upcast_attention_dtype():
        table = orig()
        if not table:
            return table
        out = dict(table)
        out[torch.bfloat16] = torch.float32
        return out

    mm.force_upcast_attention_dtype = force_upcast_attention_dtype
    mm._continuity_bf16_upcast = True
    try:
        import comfy.ldm.modules.attention as att
        att.FORCE_UPCAST_ATTENTION_DTYPE = force_upcast_attention_dtype()
    except ImportError:
        pass
    return True


def mps_attn_chunks(batch_x_heads, k_tokens, query_chunk_size=1024,
                    kv_chunk_size=None):
    """Query/KV chunk sizes that keep `heads * q * k` under MPS_ATTN_ELEM_CAP."""
    bxh = max(1, int(batch_x_heads))
    ktok = max(1, int(k_tokens))
    max_q = max(256, MPS_ATTN_ELEM_CAP // (bxh * ktok))
    q = min(int(query_chunk_size or 1024), max_q)
    kv = ktok if kv_chunk_size is None else min(int(kv_chunk_size), ktok)
    max_kv = max(256, MPS_ATTN_ELEM_CAP // (bxh * q))
    return q, min(kv, max_kv)


def _patch_mps_attn_elem_cap():
    """ComfyUI#14837: cap sub-quadratic chunks under MPS's 2^31 index wall."""
    try:
        from comfy.ldm.modules import sub_quadratic_attention as sq
        import comfy.ldm.modules.attention as att
    except ImportError:
        return False
    if getattr(sq, "_continuity_mps_elem_cap", False):
        return True
    orig = sq.efficient_dot_product_attention

    def efficient_dot_product_attention(
            query, key_t, value, query_chunk_size=1024, kv_chunk_size=None,
            **kwargs):
        if getattr(query, "device", None) is not None and query.device.type == "mps":
            query_chunk_size, kv_chunk_size = mps_attn_chunks(
                query.shape[0], key_t.shape[-1], query_chunk_size, kv_chunk_size)
        return orig(
            query, key_t, value, query_chunk_size=query_chunk_size,
            kv_chunk_size=kv_chunk_size, **kwargs)

    sq.efficient_dot_product_attention = efficient_dot_product_attention
    att.efficient_dot_product_attention = efficient_dot_product_attention
    sq._continuity_mps_elem_cap = True
    return True


def shield_h3_mps():
    """Env flags only — safe in prestartup, before ComfyUI imports torch.

    AppleSilicon-FP8 installs fused RoPE / fused RMSNorm from its `__init__`
    after every prestartup. Those flags must already be off.
    """
    os.environ[ASFP8_ROPE_ENV] = "off"
    os.environ[ASFP8_NORM_ENV] = "off"
    disarm_asfp8_rope()
    disarm_asfp8_fused_norm()
    return os.environ.get(ASFP8_ROPE_ENV)


def protect_h3_mps():
    """Force the Metal-safe H3 path from this custom node.

    Called again when the pack imports, after ComfyUI core is loaded, so
    attention patches actually bind. Prestartup only runs `shield_h3_mps`
    so this file does not import torch before ComfyUI wants it.
    """
    env = shield_h3_mps()
    patched = [
        _patch_sub_quad_empty_to_zeros(),
        _patch_bf16_attention_upcast(),
        _patch_mps_attn_elem_cap(),
    ]
    if any(patched):
        print("[Continuity Mac] forced MPS H3 attention: zeros not empty "
              "(ComfyUI#15804), bf16→fp32 upcast, 2^30 chunk cap "
              "(ComfyUI#14837). AppleSilicon-FP8 fused RoPE/RMSNorm off; "
              "pytorch SDPA left off (dense QK). Sage/kitchen/SLA refused.",
              flush=True)
    return env
