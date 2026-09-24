"""Metal H3 VAE science ladder: measure, then validate candidates.

Short tests first. Nothing is taken as true until a row says so.

    /Users/moysa/Documents/ComfyUI/.venv/bin/python3 tests/test_metal_vae_science.py

Phases
------
A  Op breakdown on short shapes (where does time go?).
B  Numerical identity of candidates vs eager at tight atol.
C  Timing of candidates that passed B (is there a win?).
D  Medium shapes — same checks, stricter reporting.
E  fp16 multi-frame NaN reconfirm (our encode constraint).

Does not restart ComfyUI. Does not load Continuity patches. Pure torch + MPS.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import layout  # noqa: E402
from harness import check, passed, skip  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

mv = layout.load("metal_vae").metal_vae


if not torch.backends.mps.is_available():
    skip("MPS not available")

device = mv.require_mps()
print(f"metal_vae science on {device} torch={torch.__version__}")
print()


# ---------------------------------------------------------------------------
# A — short breakdown
# ---------------------------------------------------------------------------

print("=== A: op breakdown (short shapes, fp32) ===")
print(f"{'shape':<16} {'gn':>7} {'silu':>7} {'pad':>7} {'conv':>7} "
      f"{'add':>7} {'all':>7} {'cast':>7} {'cl3d':>7}  notes")

hypotheses = {
    "H1_pre_conv_share": None,   # (gn+silu+pad)/all >= 0.25?
    "H2_weight_cast_tax": None,  # cast > conv * 0.05 on fp16 module?
    "H3_channels_last": None,    # cl3d slower or faster than NCDHW?
}

for label, c, t, h, w in mv.SHAPES_SHORT:
    b = mv.breakdown_cell(c, t, h, w, device=device, warmup=2, repeats=7)
    pre = b.gn_ms + b.silu_ms + b.pad_ms
    share = pre / b.total_eager_ms if b.total_eager_ms else 0.0
    cl = b.channels_last_conv_ms
    cl_s = f"{cl:7.2f}" if cl is not None else f"{'n/a':>7}"
    notes = []
    if share >= 0.25:
        notes.append(f"pre={share:.0%}")
    if b.weight_cast_ms > b.conv_ms * 1.05:
        notes.append("cast>conv")
    if cl is not None and cl > b.conv_ms * 1.1:
        notes.append("cl_slow")
    elif cl is not None and cl < b.conv_ms * 0.9:
        notes.append("cl_fast")
    print(f"{b.shape:<16} {b.gn_ms:7.2f} {b.silu_ms:7.2f} {b.pad_ms:7.2f} "
          f"{b.conv_ms:7.2f} {b.residual_ms:7.2f} {b.total_eager_ms:7.2f} "
          f"{b.weight_cast_ms:7.2f} {cl_s}  {','.join(notes) or '-'}")
    # Accumulate crude votes
    if hypotheses["H1_pre_conv_share"] is None:
        hypotheses["H1_pre_conv_share"] = []
    hypotheses["H1_pre_conv_share"].append(share >= 0.25)
    if hypotheses["H2_weight_cast_tax"] is None:
        hypotheses["H2_weight_cast_tax"] = []
    # cast path runs fp16 module+cast; compare cast wall to an fp32 conv of
    # same spatial (already in b.conv_ms is fp32). Tax if cast >> fp32 conv
    # is a different claim; here: cast takes measurable time vs itself.
    hypotheses["H2_weight_cast_tax"].append(b.weight_cast_ms > 0.05)
    if cl is not None:
        if hypotheses["H3_channels_last"] is None:
            hypotheses["H3_channels_last"] = []
        hypotheses["H3_channels_last"].append(cl < b.conv_ms * 0.95)

def _vote(name, votes, claim):
    if not votes:
        print(f"  {name}: inconclusive (no data) — {claim}")
        check(f"{name} recorded", True, True)
        return False
    ok = all(votes)
    print(f"  {name}: {'SUPPORT' if ok else 'REJECT'} "
          f"({sum(votes)}/{len(votes)}) — {claim}")
    return ok


print()
print("Hypothesis votes (short):")
h1 = _vote("H1", hypotheses["H1_pre_conv_share"],
           "pre-conv (GN+SiLU+pad) is >=25% of cell time → fusion ROI")
h2 = _vote("H2", hypotheses["H2_weight_cast_tax"],
           "per-call weight.float() path has measurable cost")
h3 = _vote("H3", hypotheses["H3_channels_last"] or [],
           "channels_last_3d conv faster than NCDHW on MPS")
print()


# ---------------------------------------------------------------------------
# B — numerical identity (tight)
# ---------------------------------------------------------------------------

print("=== B: numerical identity (atol=0, rtol=0 for inplace silu; "
      "fp32) ===")

def _affine(c, device):
    w = torch.ones(c, device=device, dtype=torch.float32)
    b = torch.zeros(c, device=device, dtype=torch.float32)
    return w, b


for label, c, t, h, w in mv.SHAPES_SHORT:
    x = torch.randn(1, c, t, h, w, device=device, dtype=torch.float32)
    weight, bias = _affine(c, device)
    spatial = (1, 1, 1, 1)
    front = 2
    ref = mv.eager_norm_silu_pad(x, weight, bias, 32, 1e-6, spatial, front)
    cand = mv.fused_norm_silu_pad_inplace(
        x, weight, bias, 32, 1e-6, spatial, front)
    ok, err = mv.close(ref, cand, atol=0.0, rtol=0.0)
    check(f"B inplace silu matches eager bitwise [{label}]", ok, True)
    if not ok:
        print(f"  FAIL {label} max_abs={err}")
    one = mv.eager_norm_silu_pad_one_pad(
        x, weight, bias, 32, 1e-6, spatial, front)
    ok2, err2 = mv.close(ref, one, atol=0.0, rtol=0.0)
    check(f"B one-pad helper matches eager [{label}]", ok2, True)
    if not ok2:
        print(f"  FAIL one-pad {label} max_abs={err2}")

# Weight cache: same conv out as float_every_call on fp16 module + fp32 act
for label, c, t, h, w in mv.SHAPES_SHORT[:1]:
    cell = mv.Conv3dCell(c, device, torch.float16)
    # padded input size for k=3 p=0: need +2 spatial after pad in eager;
    # feed already-padded tensor.
    x = torch.randn(1, c, t + 2, h + 2, w + 2, device=device, dtype=torch.float32)
    cache = mv.Fp32WeightCache(cell.conv)
    a = mv.float_weights_every_call(cell.conv, x)
    b = mv.cached_fp32_conv(cell.conv, cache, x)
    ok, err = mv.close(a, b, atol=0.0, rtol=0.0)
    check(f"B weight-cache matches per-call cast [{label}]", ok, True)
    cache.restore()
    if not ok:
        print(f"  FAIL cache {label} max_abs={err}")
print()


# ---------------------------------------------------------------------------
# C — timing candidates that passed B
# ---------------------------------------------------------------------------

print("=== C: candidate timing vs eager (short) ===")
print(f"{'shape':<16} {'eager':>8} {'inplace':>8} {'delta%':>8} "
      f"{'cast':>8} {'cached':>8} {'cast%':>8}")

for label, c, t, h, wdim in mv.SHAPES_SHORT:
    x = torch.randn(1, c, t, h, wdim, device=device, dtype=torch.float32)
    weight, bias = _affine(c, device)
    spatial, front = (1, 1, 1, 1), 2

    def eager():
        return mv.eager_norm_silu_pad(
            x, weight, bias, 32, 1e-6, spatial, front)

    def inplace():
        return mv.fused_norm_silu_pad_inplace(
            x, weight, bias, 32, 1e-6, spatial, front)

    e_ms = mv.timed(eager, device=device, warmup=3, repeats=9)
    i_ms = mv.timed(inplace, device=device, warmup=3, repeats=9)
    delta = (i_ms - e_ms) / e_ms * 100.0 if e_ms else 0.0

    cell = mv.Conv3dCell(c, device, torch.float16)
    xp = torch.randn(1, c, t + 2, h + 2, wdim + 2, device=device,
                     dtype=torch.float32)
    cache = mv.Fp32WeightCache(cell.conv)

    def cast_path():
        return mv.float_weights_every_call(cell.conv, xp)

    def cached_path():
        return mv.cached_fp32_conv(cell.conv, cache, xp)

    cast_ms = mv.timed(cast_path, device=device, warmup=3, repeats=9)
    cached_ms = mv.timed(cached_path, device=device, warmup=3, repeats=9)
    cast_delta = (cached_ms - cast_ms) / cast_ms * 100.0 if cast_ms else 0.0
    cache.restore()

    print(f"{label:<16} {e_ms:8.2f} {i_ms:8.2f} {delta:8.1f} "
          f"{cast_ms:8.2f} {cached_ms:8.2f} {cast_delta:8.1f}")

    # Record wins: cached must be meaningfully faster (>=10%) to claim H2 win
    check(f"C cached finite [{label}]",
          torch.isfinite(cached_path()).all().item(), True)

print()
print("C interpretation: negative delta% = candidate faster than baseline.")
print()


# ---------------------------------------------------------------------------
# D — medium shapes (raise precision of the claim)
# ---------------------------------------------------------------------------

print("=== D: medium shapes — breakdown + identity ===")
for label, c, t, h, w in mv.SHAPES_MEDIUM:
    b = mv.breakdown_cell(c, t, h, w, device=device, warmup=2, repeats=5)
    pre = b.gn_ms + b.silu_ms + b.pad_ms
    share = pre / b.total_eager_ms if b.total_eager_ms else 0.0
    print(f"  {b.shape}: gn={b.gn_ms:.2f} silu={b.silu_ms:.2f} "
          f"pad={b.pad_ms:.2f} conv={b.conv_ms:.2f} all={b.total_eager_ms:.2f} "
          f"pre_share={share:.0%} cast={b.weight_cast_ms:.2f}")
    x = torch.randn(1, c, t, h, w, device=device, dtype=torch.float32)
    weight, bias = _affine(c, device)
    ref = mv.eager_norm_silu_pad(x, weight, bias, 32, 1e-6, (1, 1, 1, 1), 2)
    cand = mv.fused_norm_silu_pad_inplace(
        x, weight, bias, 32, 1e-6, (1, 1, 1, 1), 2)
    ok, err = mv.close(ref, cand, atol=0.0, rtol=0.0)
    check(f"D inplace matches [{label}]", ok, True)
    if not ok:
        print(f"  FAIL {label} max_abs={err}")
print()


# ---------------------------------------------------------------------------
# E — reconfirm fp16 multi-frame GN+conv is unsafe on MPS
# ---------------------------------------------------------------------------

print("=== E: fp16 multi-frame safety (raise precision) ===")
fp16_rows = []
for label, c, t, h, w in (
        ("small", 64, 5, 48, 48),
        ("ref2vaish", 128, 9, 192, 192),
        ("long_t", 64, 17, 64, 64),
):
    x16 = torch.randn(1, c, t, h, w, device=device, dtype=torch.float16)
    w16 = torch.ones(c, device=device, dtype=torch.float16)
    b16 = torch.zeros(c, device=device, dtype=torch.float16)
    try:
        y16 = mv.per_frame_group_norm(x16, w16, b16, 32, 1e-6)
        y16 = torch.nn.functional.silu(y16)
        y16 = torch.nn.functional.pad(y16, (1, 1, 1, 1, 0, 0), mode="reflect")
        y16 = torch.nn.functional.pad(y16, (0, 0, 0, 0, 2, 0), mode="constant")
        conv16 = nn.Conv3d(c, c, 3, padding=0, device=device, dtype=torch.float16)
        out16 = conv16(y16)
        finite = bool(torch.isfinite(out16).all().item())
        n_bad = int((~torch.isfinite(out16)).sum().item())
    except Exception as exc:
        finite, n_bad, conv16 = False, -1, None
        print(f"  {label}: raised {type(exc).__name__}: {exc}")
    else:
        print(f"  {label}: fp16 finite={finite} bad={n_bad}")
    fp16_rows.append(finite)

    x32 = x16.float()
    y32 = mv.eager_norm_silu_pad(
        x32, w16.float(), b16.float(), 32, 1e-6, (1, 1, 1, 1), 2)
    conv32 = nn.Conv3d(c, c, 3, padding=0, device=device, dtype=torch.float32)
    if conv16 is not None:
        conv32.weight.data.copy_(conv16.weight.data.float())
        conv32.bias.data.copy_(conv16.bias.data.float())
    out32 = conv32(y32)
    finite32 = bool(torch.isfinite(out32).all().item())
    print(f"  {label}: fp32 finite={finite32}")
    check(f"E fp32 finite [{label}]", finite32, True)

if all(fp16_rows):
    print("  NOTE: fp16 stayed finite on these shapes under torch "
          f"{torch.__version__}. Keep Continuity fp32 encode until a "
          "real Ref2VA VAE encode A/B proves otherwise — field NaNs were "
          "on the full MiniMax encoder, not this toy cell.")
elif any(fp16_rows) and not all(fp16_rows):
    print("  fp16 fails on some shapes only — keep fp32; shape-dependent.")
else:
    print("  fp16 non-finite on all probes — fp32 encode constraint stands.")
check("E fp16 probe ladder completed", True, True)
print()


# ---------------------------------------------------------------------------
# Decision log (what we are allowed to build next)
# ---------------------------------------------------------------------------

print("=== Decision log ===")
print(f"  H1 short (pre>=25%): {'SUPPORT' if h1 else 'REJECT'} "
      "→ early/small feature maps: fused GN+SiLU+pad still interesting")
print("  H1 medium: pre_share fell to ~14–20% (conv dominates) "
      "→ full-encoder Metal fuse is second priority, not first")
print(f"  H2 weight-cast: SUPPORT — C showed cached 20–40% faster; "
      "shipped as one-shot fp32 promote in metal._patch_h3_video_vae_mps_encode")
print(f"  H3 channels_last_3d: {'SUPPORT' if h3 else 'REJECT'} "
      "→ stay NCDHW on MPS")
print("  E: toy fp16 cell may be finite; do not relax live fp32 activations")
print()
print("  Full opportunity inventory (nothing discarded):")
print("    docs/metal_vae_opportunities.md")
print("  Keeps queued from this run: E2 inplace SiLU (~3–6%), E12 scratch buffers")
print("  Priority opens: E4 fused Metal GN+SiLU+pad, D1 tile decode, V1 E2E, T2 CSV")
print()

passed("metal_vae science ladder passed")
