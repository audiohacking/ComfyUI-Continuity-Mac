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
    """Query/KV chunk sizes that keep `heads * q * k` under MPS_ATTN_ELEM_CAP.

    Prefer a full-KV pass (ComfyUI's `_get_attention_scores_no_kv_chunking`)
    by shrinking the query chunk. Flooring q at 256 used to push kv under
    `k_tokens`, which flips into the checkpointed multi-KV path — that is
    what made long Ref2VA sequences crawl on Continuity versus native
    h3.c (MPSGraph / fused Metal) on the same Mac.
    """
    bxh = max(1, int(batch_x_heads))
    ktok = max(1, int(k_tokens))
    want_q = max(1, int(query_chunk_size or 1024))

    if kv_chunk_size is not None:
        kv = min(max(1, int(kv_chunk_size)), ktok)
        q = min(want_q, max(1, MPS_ATTN_ELEM_CAP // (bxh * kv)))
        return max(1, q), kv

    # Full KV fits with a (possibly small) query chunk — keep kv == ktok so
    # sub-quadratic takes the no-kv-chunking branch.
    max_q_full_kv = max(1, MPS_ATTN_ELEM_CAP // (bxh * ktok))
    if max_q_full_kv >= 1:
        return min(want_q, max_q_full_kv), ktok

    # Even q=1 overflows — last resort, chunk KV too.
    q = min(want_q, 1024)
    kv = max(1, MPS_ATTN_ELEM_CAP // (bxh * q))
    return q, min(kv, ktok)


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


def physical_ram_bytes():
    """Physical RAM, or 0 when the OS will not say."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 0


def shield_mps_watermark():
    """Stop AppleSilicon-FP8 holding most of a big Mac's unified pool.

    That pack `setdefault`s low=0.8 / high=1.0 so the MPS cache is not
    reclaimed until 80% of Apple's recommended_max and then hard-caps
    there. On a 16–128 GB machine that prevents swap. On a 512 GB Mac
    recommended_max is ~464 GB: the reserved pool sits at ~407 GB
    ("other allocations") while live H3 tensors are ~56 GB (TE + VAE),
    and a 544 MB tile is refused. H3 on other boxes runs in well under
    128 GB — the 400 GB is the watermark, not the model.

    Continuity prestartup runs after that pack, before torch imports,
    so these assignments win. low=0.2 starts reclaiming cache early.
    high=0.0 on ≥256 GB RAM disables the false cap (PyTorch's own
    suggestion on this error). Smaller Macs keep high=1.0.
    """
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.2"
    if physical_ram_bytes() >= 256 * 1024 ** 3:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    else:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1.0"
    return (os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"],
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"])


def shield_h3_mps():
    """Env flags only — safe in prestartup, before ComfyUI imports torch.

    AppleSilicon-FP8 installs fused RoPE / fused RMSNorm from its `__init__`
    after every prestartup. The env flags must already be off. Its MPS
    watermark is overwritten here for the same reason: torch reads those
    ratios once, at MPS init.
    """
    os.environ[ASFP8_ROPE_ENV] = "off"
    os.environ[ASFP8_NORM_ENV] = "off"
    shield_mps_watermark()
    disarm_asfp8_rope()
    disarm_asfp8_fused_norm()
    return os.environ.get(ASFP8_ROPE_ENV)


def _patch_h3_video_vae_mps_encode():
    """Keep H3 video VAE encode on MPS without cloning the encoder.

    fp16 5D GroupNorm + CausalConv3d emit NaN on a clip (Ref2VA went
    black). Converting the whole encoder to fp32 doubled it on top of
    the 49 GB text encoder and OOMed. GroupNorm is 4D per-frame; each
    conv/norm computes in fp32 on the GPU and the module stays fp16.

    Activations stay float32 through the encoder chain — casting each
    layer back to fp16 re-introduced Inf on longer/portrait clips
    (Abatantuono @vid-1). quant_conv is a plain 1x1 Conv3d, so
    `_encode_moments` also runs it in fp32 for the same reason.
    """
    try:
        from comfy.ldm.minimax.vae import (
            CausalConv3d, MiniMaxH3VideoVAE, TemporalIsolatedGroupNorm)
        import torch
        import torch.nn.functional as F
    except ImportError:
        return False
    if getattr(MiniMaxH3VideoVAE, "_continuity_mps_fp32_video_encode", False):
        return True

    orig_norm = TemporalIsolatedGroupNorm.forward

    def group_norm_forward(self, x):
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
            if x.device.type == "mps":
                weight = None if self.weight is None else self.weight.float()
                bias = None if self.bias is None else self.bias.float()
                x = F.group_norm(x.float(), self.num_groups, weight, bias, self.eps)
            else:
                x = torch.nn.GroupNorm.forward(self, x)
            return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return orig_norm(self, x)

    TemporalIsolatedGroupNorm.forward = group_norm_forward

    orig_conv = CausalConv3d.forward

    def conv_forward(self, x, pre_norm=None, spatial_pad=None, residual=None):
        if getattr(x, "device", None) is not None and x.device.type == "mps":
            saved_w = self.weight.data
            saved_b = None if self.bias is None else self.bias.data
            self.weight.data = saved_w.float()
            if saved_b is not None:
                self.bias.data = saved_b.float()
            try:
                return orig_conv(
                    self, x.float(), pre_norm, spatial_pad,
                    None if residual is None else residual.float())
            finally:
                self.weight.data = saved_w
                if saved_b is not None:
                    self.bias.data = saved_b
        return orig_conv(self, x, pre_norm, spatial_pad, residual)

    CausalConv3d.forward = conv_forward

    orig_moments = MiniMaxH3VideoVAE._encode_moments

    def encode_moments(self, x):
        if getattr(x, "device", None) is not None and x.device.type == "mps":
            qc = self.quant_conv
            saved_w = qc.weight.data
            saved_b = None if qc.bias is None else qc.bias.data
            qc.weight.data = saved_w.float()
            if saved_b is not None:
                qc.bias.data = saved_b.float()
            try:
                return orig_moments(self, x.float())
            finally:
                qc.weight.data = saved_w
                if saved_b is not None:
                    qc.bias.data = saved_b
        return orig_moments(self, x)

    MiniMaxH3VideoVAE._encode_moments = encode_moments
    MiniMaxH3VideoVAE._continuity_mps_fp32_video_encode = True
    return True


def _deepstack_mask_report(embeds, embeds_info, visual_pos_masks, deepstack):
    """-> (seq, n_mask, n_ds, rows) for the Qwen3-VL DeepStack inject guard.

    `rows` is a list of (index, size, deepstack0, end) per image embed.
    Never raises: a broken embeds_info entry becomes a None-filled row.
    """
    try:
        seq = int(embeds.shape[1]) if embeds is not None else -1
    except Exception:  # noqa: BLE001
        seq = -1
    try:
        n_mask = (0 if visual_pos_masks is None
                  else int(visual_pos_masks.sum().item()))
    except Exception:  # noqa: BLE001
        n_mask = -1
    try:
        n_ds = 0 if not deepstack else int(deepstack[0].shape[0])
    except Exception:  # noqa: BLE001
        n_ds = -1
    rows = []
    for e in embeds_info or []:
        try:
            if e.get("type") != "image":
                continue
            start = e.get("index")
            size = e.get("size")
            extra = e.get("extra") if isinstance(e.get("extra"), dict) else None
            ds = extra.get("deepstack") if extra else None
            ds0 = int(ds[0].shape[0]) if ds else None
            end = (start + size if isinstance(start, int)
                   and isinstance(size, int) else None)
            rows.append((start, size, ds0, end))
        except Exception:  # noqa: BLE001
            rows.append((None, None, None, None))
    return seq, n_mask, n_ds, rows


def _rebuild_visual_pos_masks(embeds, embeds_info, deepstack):
    """Rebuild the DeepStack mask when Comfy left it empty but sizes agree.

    Returns (mask, deepstack) or (None, None) to skip inject safely. Merged
    vision tokens stay in `embeds` either way; DeepStack is the residual add.
    Never raises — any failure means skip.
    """
    try:
        import torch

        seq, _n_mask, n_ds, rows = _deepstack_mask_report(
            embeds, embeds_info, None, deepstack)
        if n_ds <= 0 or seq <= 0:
            return None, None
        sized = sum(r[1] or 0 for r in rows)
        if sized != n_ds:
            return None, None
        mask = torch.zeros((1, seq), dtype=torch.bool, device=embeds.device)
        for start, size, _ds0, end in rows:
            if not isinstance(start, int) or not isinstance(size, int) or size <= 0:
                continue
            if start < 0 or start >= seq or end is None:
                continue
            mask[0, start:min(end, seq)] = True
        if int(mask.sum().item()) != n_ds:
            return None, None
        return mask, deepstack
    except Exception:  # noqa: BLE001
        return None, None


def _reconcile_deepstack(embeds, embeds_info, visual_pos_masks, deepstack):
    """Make DeepStack safe to inject, or drop it.

    -> (mask, deepstack). `(None, None)` means skip the residual add; the
    merged vision tokens already sit in `embeds`. Never raises.
    """
    import logging

    try:
        if deepstack is None:
            return visual_pos_masks, None
        seq, n_mask, n_ds, rows = _deepstack_mask_report(
            embeds, embeds_info, visual_pos_masks, deepstack)
        if n_ds <= 0:
            return visual_pos_masks, None
        if n_mask == n_ds:
            return visual_pos_masks, deepstack
        logging.warning(
            "[Continuity Mac] Qwen3-VL DeepStack mask/size mismatch: "
            "mask=%d deepstack=%d seq=%d embeds_info(index,size,ds)=%s",
            n_mask, n_ds, seq, rows)
        rebuilt, deepstack2 = _rebuild_visual_pos_masks(
            embeds, embeds_info, deepstack)
        if rebuilt is not None:
            logging.warning(
                "[Continuity Mac] DeepStack mask rebuilt (%d positions)",
                n_ds)
            return rebuilt, deepstack2
        logging.warning(
            "[Continuity Mac] skipping DeepStack inject (merged vision "
            "tokens remain in the prompt)")
        return None, None
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "[Continuity Mac] DeepStack guard failed (%s); skipping inject",
            exc)
        return None, None


def _patch_qwen3vl_deepstack_mask():
    """Keep Ref2VA CLIP encode from dying on an empty DeepStack mask.

    Seen on MPS after long video VAE encodes: `build_image_inputs` returns
    deepstack features (e.g. 3576 visual tokens for image + 2 fps video
    blocks) but `visual_pos_masks` has zero Trues, so Llama2_'s
    `x[mask] += deepstack` raises. Merged vision embeddings are already
    spliced into the sequence — dropping DeepStack is safe enough to
    finish the encode; a matching mask is rebuilt when sizes allow.

    Also wraps Llama2_ so a mismatch that slips past still cannot crash
    the encode mid-layer.
    """
    try:
        from comfy.text_encoders import qwen3vl
        from comfy.text_encoders import llama as llama_te
    except ImportError:
        return False
    if getattr(qwen3vl.Qwen3VL, "_continuity_deepstack_guard", False):
        return True

    import logging

    orig_build = qwen3vl.Qwen3VL.build_image_inputs

    def build_image_inputs(self, embeds, embeds_info):
        try:
            position_ids, visual_pos_masks, deepstack = orig_build(
                self, embeds, embeds_info)
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "[Continuity Mac] Qwen3-VL build_image_inputs failed (%s); "
                "continuing without DeepStack / MRoPE extras",
                exc)
            return None, None, None
        visual_pos_masks, deepstack = _reconcile_deepstack(
            embeds, embeds_info, visual_pos_masks, deepstack)
        return position_ids, visual_pos_masks, deepstack

    qwen3vl.Qwen3VL.build_image_inputs = build_image_inputs
    qwen3vl.Qwen3VL._continuity_deepstack_guard = True

    # Belt: if anything still hands Llama a mismatched pair, skip the add
    # rather than raising mid-layer and killing the whole Ref2VA encode.
    if not getattr(llama_te.Llama2_, "_continuity_deepstack_inject_guard", False):
        _orig_llama_forward = llama_te.Llama2_.forward

        def llama_forward(self, *args, **kwargs):
            deepstack = kwargs.get("deepstack_embeds")
            mask = kwargs.get("visual_pos_masks")
            if deepstack is not None:
                try:
                    n_ds = int(deepstack[0].shape[0])
                    n_mask = (0 if mask is None else int(mask.sum().item()))
                except Exception:  # noqa: BLE001
                    n_ds, n_mask = -1, -1
                if n_ds > 0 and n_mask != n_ds:
                    logging.warning(
                        "[Continuity Mac] Llama DeepStack inject skipped "
                        "(mask=%s deepstack=%s)", n_mask, n_ds)
                    kwargs = dict(kwargs)
                    kwargs["deepstack_embeds"] = None
                    kwargs["visual_pos_masks"] = None
            return _orig_llama_forward(self, *args, **kwargs)

        llama_te.Llama2_.forward = llama_forward
        llama_te.Llama2_._continuity_deepstack_inject_guard = True

    return True


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
        _patch_h3_video_vae_mps_encode(),
        _patch_qwen3vl_deepstack_mask(),
    ]
    if any(patched):
        print("[Continuity Mac] forced MPS H3 attention: zeros not empty "
              "(ComfyUI#15804), bf16→fp32 upcast, 2^30 chunk cap "
              "(ComfyUI#14837). AppleSilicon-FP8 fused RoPE/RMSNorm off; "
              "pytorch SDPA left off (dense QK). Sage/kitchen/SLA refused. "
              "Ref2VA video encode on GPU (fp32 activations end-to-end, "
              "no encoder clone). DeepStack mask guard on. "
              "MPS watermark low=%s high=%s (ASFP8 0.8/1.0 was holding "
              "the 512 GB pool)."
              % (os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
                 os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")),
              flush=True)
    return env
