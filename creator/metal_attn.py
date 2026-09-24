"""Apple Metal research for MiniMax H3 attention (sampler bottleneck).

Self-contained enough to run without a live prompt: builds Q/K/V at
H3-like layouts and times Continuity's current path (sub-quadratic)
against mtlflashattn. Numerical checks gate any live override.

H3 after RoPE feeds optimized_attention as [B, heads, S, dim]
(skip_reshape=True). metal_flash_attn.flash_attn_func wants
[B, S, heads, dim] — we transpose at the boundary.

See docs/metal_sampler_opportunities.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# H3 DiT defaults from comfy.ldm.minimax.model.MiniMaxH3
H3_HEADS = 56
H3_DIM = 128

# Short ladder — raise S only after smaller rows pass.
SEQS_SHORT = (256, 512, 1024)
SEQS_MEDIUM = (2048, 4096)


def require_mps():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    return torch.device("mps")


def sync(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()


def timed(fn, *, device, warmup=2, repeats=5) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync(device)
        samples = []
        for _ in range(repeats):
            sync(device)
            t0 = time.perf_counter()
            fn()
            sync(device)
            samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def close(a: torch.Tensor, b: torch.Tensor, *, atol, rtol):
    if a.shape != b.shape:
        return False, float("inf")
    err = (a.float() - b.float()).abs().max().item()
    ok = bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
              and torch.isfinite(a).all() and torch.isfinite(b).all())
    return ok, float(err)


def make_qkv(batch, heads, seq, dim, *, device, dtype):
    """BHSD layout (Comfy skip_reshape)."""
    q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    return q, k, v


def attention_sdpa_bhsd(q, k, v):
    """Stock PyTorch SDPA on BHSD — dense scores (reference / danger at long S)."""
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def attention_flash_bhsd(q, k, v):
    """mtlflashattn on BHSD via transpose to BSHD."""
    from metal_flash_attn import flash_attn_func
    # flash: [B, S, H, D]
    qq = q.transpose(1, 2).contiguous()
    kk = k.transpose(1, 2).contiguous()
    vv = v.transpose(1, 2).contiguous()
    out = flash_attn_func(qq, kk, vv, causal=False)
    return out.transpose(1, 2).contiguous()


def attention_subquad_bhsd(q, k, v, *, query_chunk_size=1024):
    """Comfy sub-quadratic path if importable; else chunked SDPA stand-in."""
    try:
        from comfy.ldm.modules.sub_quadratic_attention import (
            efficient_dot_product_attention)
    except ImportError:
        # Stand-in: query-chunked SDPA (similar memory idea, not identical math)
        return _chunked_sdpa(q, k, v, query_chunk_size=query_chunk_size)

    # efficient_dot_product_attention expects query [B*H, q, dim],
    # key_t [B*H, dim, k], value [B*H, k, dim]
    b, h, s, d = q.shape
    qflat = q.reshape(b * h, s, d)
    kflat = k.reshape(b * h, s, d)
    vflat = v.reshape(b * h, s, d)
    key_t = kflat.transpose(1, 2)
    out = efficient_dot_product_attention(
        qflat, key_t, vflat,
        query_chunk_size=query_chunk_size,
        kv_chunk_size=None,
        use_checkpoint=False,
    )
    return out.reshape(b, h, s, d)


def _chunked_sdpa(q, k, v, *, query_chunk_size=256):
    b, h, s, d = q.shape
    outs = []
    for i in range(0, s, query_chunk_size):
        qi = q[:, :, i:i + query_chunk_size]
        outs.append(F.scaled_dot_product_attention(qi, k, v, dropout_p=0.0))
    return torch.cat(outs, dim=2)


@dataclass
class AttnRow:
    label: str
    seq: int
    heads: int
    dim: int
    dtype: str
    sdpa_ms: float | None
    flash_ms: float | None
    subquad_ms: float | None
    flash_vs_subquad: float | None
    flash_ok: bool | None
    flash_err: float | None


def estimate_video_tokens(width, height, frames, *, spatial_stride=32,
                          temporal_stride=4):
    """Rough packed video-token count (VAE 16× + patch 2 → 32 spatial)."""
    th = max(1, height // spatial_stride)
    tw = max(1, width // spatial_stride)
    tt = max(1, (frames + temporal_stride - 1) // temporal_stride)
    return th * tw * tt
