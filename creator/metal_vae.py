"""Apple Metal research for MiniMax H3 video-VAE encode.

Self-contained: no ComfyUI import. Reimplements the ResNet-cell ops the
H3 encoder runs between convolutions (per-frame GroupNorm, SiLU,
reflect spatial pad, causal temporal pad, Conv3d, residual add) so we
can measure and A/B candidates on MPS without booting core.

Design rules for this fork
-------------------------
* Activations stay float32 on MPS until opportunity **E8** in
  `docs/metal_vae_opportunities.md` is cleared by a *full* MiniMax
  encoder A/B — not by toy cells.
* Prefer NCDHW until **E7** says otherwise (measured reject so far).
* Nothing ships into `metal.py`'s live patches until a science suite
  row says so: wall-time win + numerical match at the stated tolerance.
* Do not discard small wins. Log every candidate in
  `docs/metal_vae_opportunities.md` (`keep` / `parked` / `open`).

Public surface is intentionally small: reference eager ops, candidate
ops, a sync'd MPS timer, and shapes used by the science suite.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shapes. Short ladder first; Ref2VA-ish last.
# ---------------------------------------------------------------------------

# (label, channels, frames, height, width) — NCDHW after ImageNet-ish stem.
# Keep short shapes tiny so a suite finishes in seconds; raise precision
# only after a hypothesis survives.
SHAPES_SHORT = (
    ("c64_t3_32", 64, 3, 32, 32),
    ("c128_t3_64", 128, 3, 64, 64),
    ("c256_t3_32", 256, 3, 32, 32),
)

SHAPES_MEDIUM = (
    ("c128_t5_128", 128, 5, 128, 128),
    ("c256_t5_64", 256, 5, 64, 64),
)

# One slice of a 768-short-edge ref encode (rough; not a full VAE).
SHAPES_REF2VAISH = (
    ("c128_t9_192", 128, 9, 192, 192),
)


def require_mps():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    return torch.device("mps")


def sync(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@dataclass(frozen=True)
class Timing:
    label: str
    ms: float
    bytes_touched: int = 0


def timed(fn, *, device, warmup=3, repeats=10) -> float:
    """Median wall ms of `fn()`, MPS-synchronised. No grad."""
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


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


# ---------------------------------------------------------------------------
# Reference eager path (matches published H3 CausalConv3d fold points).
# ---------------------------------------------------------------------------

def per_frame_group_norm(x, weight, bias, num_groups, eps):
    """x [B,C,T,H,W] -> same. Stats per frame (T folded into batch)."""
    b, c, t, h, w = x.shape
    y = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
    y = F.group_norm(y, num_groups, weight, bias, eps)
    return y.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def eager_norm_silu_pad(x, weight, bias, num_groups, eps, spatial_pad, front,
                        *, do_silu=True):
    """Separate passes: GN -> SiLU -> reflect HW pad -> causal T pad.

    `spatial_pad` is (left, right, top, bottom) for F.pad on NCDHW.
    `front` is zeros prepended on T (causal).
    """
    y = per_frame_group_norm(x, weight, bias, num_groups, eps)
    if do_silu:
        y = F.silu(y)
    if any(spatial_pad):
        # F.pad on 5D: (W_left, W_right, H_top, H_bottom, T_front, T_back)
        y = F.pad(y, (*spatial_pad, 0, 0), mode="reflect")
    if front:
        y = F.pad(y, (0, 0, 0, 0, front, 0), mode="constant")
    return y


def eager_norm_silu_pad_one_pad(x, weight, bias, num_groups, eps, spatial_pad,
                               front, *, do_silu=True):
    """Same math, one pad call (candidate: fewer kernel launches)."""
    y = per_frame_group_norm(x, weight, bias, num_groups, eps)
    if do_silu:
        y = F.silu(y)
    left, right, top, bottom = spatial_pad
    if left or right or top or bottom or front:
        # Reflect on HW, constant 0 on T — cannot mix modes in one F.pad.
        # So this path only wins when front==0; otherwise falls back.
        if front:
            if left or right or top or bottom:
                y = F.pad(y, (left, right, top, bottom, 0, 0), mode="reflect")
            y = F.pad(y, (0, 0, 0, 0, front, 0), mode="constant")
        else:
            y = F.pad(y, (left, right, top, bottom, 0, 0), mode="reflect")
    return y


def fused_norm_silu_pad_inplace(x, weight, bias, num_groups, eps, spatial_pad,
                                front, *, do_silu=True):
    """Candidate: GN then *inplace* SiLU, then the same pads as eager.

    Not a Metal kernel — measures whether the SiLU alloc matters before we
    spend time on a real fused shader. Must match `eager_norm_silu_pad`
    bit-for-bit at fp32 (SiLU inplace on a fresh GN output is exact).
    """
    y = per_frame_group_norm(x, weight, bias, num_groups, eps)
    if do_silu:
        y = F.silu(y, inplace=True)
    if any(spatial_pad):
        y = F.pad(y, (*spatial_pad, 0, 0), mode="reflect")
    if front:
        y = F.pad(y, (0, 0, 0, 0, front, 0), mode="constant")
    return y


# ---------------------------------------------------------------------------
# Conv + residual; weight-cache candidate.
# ---------------------------------------------------------------------------

class Conv3dCell(nn.Module):
    """3x3x3 conv matching CausalConv3d after padding is already applied."""

    def __init__(self, channels, device, dtype=torch.float32):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=0,
                              bias=True, device=device, dtype=dtype)

    def forward(self, x, residual=None):
        y = self.conv(x)
        if residual is not None:
            y = y + residual
        return y


class Fp32WeightCache:
    """Keep float32 weight/bias copies; avoid per-forward `.float()` clones.

    Continuity's live MPS patch currently reassigns `weight.data = saved.float()`
    every call. This object measures whether that tax is real.
    """

    def __init__(self, module: nn.Conv3d):
        self.module = module
        self._w = module.weight.detach().float().contiguous()
        self._b = None if module.bias is None else module.bias.detach().float().contiguous()
        self._installed = False

    def install(self):
        if self._installed:
            return
        self._saved_w = self.module.weight.data
        self._saved_b = None if self.module.bias is None else self.module.bias.data
        self.module.weight.data = self._w
        if self._b is not None:
            self.module.bias.data = self._b
        self._installed = True

    def restore(self):
        if not self._installed:
            return
        self.module.weight.data = self._saved_w
        if self._saved_b is not None:
            self.module.bias.data = self._saved_b
        self._installed = False


def float_weights_every_call(module: nn.Conv3d, x, residual=None):
    """Mirror of today's Continuity MPS tax: cast weights, run, restore."""
    saved_w = module.weight.data
    saved_b = None if module.bias is None else module.bias.data
    module.weight.data = saved_w.float()
    if saved_b is not None:
        module.bias.data = saved_b.float()
    try:
        y = module(x.float())
        if residual is not None:
            y = y + residual.float()
        return y
    finally:
        module.weight.data = saved_w
        if saved_b is not None:
            module.bias.data = saved_b


def cached_fp32_conv(module: nn.Conv3d, cache: Fp32WeightCache, x,
                     residual=None):
    """Conv with weights permanently held as fp32 (no per-call cast)."""
    cache.install()
    y = module(x)
    if residual is not None:
        y = y + residual
    return y


# ---------------------------------------------------------------------------
# ResNet-cell breakdown (the unit NVIDIA fused).
# ---------------------------------------------------------------------------

@dataclass
class Breakdown:
    shape: str
    gn_ms: float
    silu_ms: float
    pad_ms: float
    conv_ms: float
    residual_ms: float
    total_eager_ms: float
    weight_cast_ms: float
    channels_last_conv_ms: float | None


def breakdown_cell(channels, frames, height, width, *, device, dtype=torch.float32,
                   num_groups=32, spatial_pad=(1, 1, 1, 1), front=2,
                   warmup=2, repeats=8) -> Breakdown:
    """Time each stage of one padded conv cell on `device`."""
    x = torch.randn(1, channels, frames, height, width, device=device, dtype=dtype)
    w = torch.randn(channels, device=device, dtype=dtype)
    b = torch.randn(channels, device=device, dtype=dtype)
    # Affine init like GroupNorm.
    nn.init.ones_(w)
    nn.init.zeros_(b)

    def gn():
        return per_frame_group_norm(x, w, b, num_groups, 1e-6)

    y0 = gn()
    def silu():
        return F.silu(y0)

    y1 = silu()
    def pad():
        z = F.pad(y1, (*spatial_pad, 0, 0), mode="reflect")
        return F.pad(z, (0, 0, 0, 0, front, 0), mode="constant")

    padded = pad()
    cell = Conv3dCell(channels, device, dtype)
    # Residual is pre-pad spatial size; H3 folds residual at conv out.
    # For timing we add a same-shaped tensor after conv (approx epilogue).
    residual = torch.randn_like(cell.conv(padded))

    def conv():
        return cell.conv(padded)

    def resid():
        return residual + residual  # pure bandwidth add of conv-out size

    def eager_all():
        z = eager_norm_silu_pad(x, w, b, num_groups, 1e-6, spatial_pad, front)
        return cell.conv(z) + residual

    # Weight-cast tax on an fp16 module body (simulates Continuity patch).
    cell_fp16 = Conv3dCell(channels, device, torch.float16)
    x16 = padded.half()
    def cast_tax():
        return float_weights_every_call(cell_fp16.conv, x16)

    cl_ms = None
    try:
        padded_cl = padded.contiguous(memory_format=torch.channels_last_3d)
        w_cl = cell.conv.weight.contiguous(memory_format=torch.channels_last_3d)
        def conv_cl():
            return F.conv3d(padded_cl, w_cl, cell.conv.bias, cell.conv.stride)
        cl_ms = timed(conv_cl, device=device, warmup=warmup, repeats=repeats)
    except Exception:
        cl_ms = None

    label = f"c{channels}_t{frames}_{height}"
    return Breakdown(
        shape=label,
        gn_ms=timed(gn, device=device, warmup=warmup, repeats=repeats),
        silu_ms=timed(silu, device=device, warmup=warmup, repeats=repeats),
        pad_ms=timed(pad, device=device, warmup=warmup, repeats=repeats),
        conv_ms=timed(conv, device=device, warmup=warmup, repeats=repeats),
        residual_ms=timed(resid, device=device, warmup=warmup, repeats=repeats),
        total_eager_ms=timed(eager_all, device=device, warmup=warmup, repeats=repeats),
        weight_cast_ms=timed(cast_tax, device=device, warmup=warmup, repeats=repeats),
        channels_last_conv_ms=cl_ms,
    )


def close(a: torch.Tensor, b: torch.Tensor, *, atol, rtol) -> tuple[bool, float]:
    """-> (ok, max_abs_err)."""
    if a.shape != b.shape:
        return False, float("inf")
    err = (a - b).abs().max().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol) and torch.isfinite(a).all() \
        and torch.isfinite(b).all()
    return bool(ok), float(err)
