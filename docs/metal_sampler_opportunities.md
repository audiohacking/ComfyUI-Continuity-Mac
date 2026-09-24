# Metal H3 sampler / attention opportunity log

Target: **~1 minute wall per 1 second of generated video** on Continuity Metal.
Living inventory — **nothing discarded**. Small wins compound.

Science: `creator/metal_attn.py`, `tests/test_metal_attn_science.py`  
Related: `docs/metal_vae_opportunities.md` (VAE / encode — secondary while refs cache)

**Policy:** measured wins that matter are **forced defaults**, not UI toggles.
A prefs click must not put sampling back on a known-bad path.

---

## Baseline — 2026-09-24 H3_00037_ (this session)

| Fact | Value |
|------|--------|
| Output | 8.0 s, 192×24 fps, 512×896 |
| Video ref | **5 s trim** (not full source) @ `ref_size=max` |
| Expectation | ~**8 min** total (1 min / 1 s) |
| Actual | **37:43** total (~4.7× target) |
| Sampling | **3/3 in 34:48** ≈ **696 s/step** (TaoMate) |
| Ref encode | **0** (all cache hits) |
| Attention backend (log) | **`Using sub quadratic optimization for attention`** |
| ASFP8 flash | Installed on `F.sdpa` — **H3 does not call `F.sdpa`** |

**Gap to close on sampler alone:** ~35 min → ~6–7 min ≈ **5×** (or fewer tokens × faster attn).

---

## Status legend

`shipped` / `keep` / `parked` / `open` / `blocked` — same as VAE log.

---

## S — Sampler / attention

| ID | Opportunity | Status | Evidence | Revisit / next |
|----|-------------|--------|----------|----------------|
| S1 | Route H3 `optimized_attention` → **mtlflashattn** | **rolled back** | Science ok; **live Ref2VA → noise** (2026-09-24). Rebind banner lied about engagement; wrapper not bit-exact for H3 | Park until call-site + visual A/B |
| S1b | Attention combo = **`default` only** (no sage/kitchen/sla/**laya**) | **shipped** | Retired names coerce on load | Soft-refresh UI after reload |
| S2 | Enable `--use-pytorch-cross-attention` so `attention_pytorch` → patched `F.sdpa` | **parked** | Indirect | — |
| S3 | Keep sub-quad **bf16→fp32** attention upcast | **shipped** (live) | Known-good Metal path | — |
| S4 | Tune `MPS_ATTN_ELEM_CAP` / query chunks | **shipped** (live) | Cap still needed | — |
| S5 | `--use-split-cross-attention` | **parked** | — | — |
| S6 | Continuity step / block caches (EasyCache / FBC) | **open** | Allowed on MPS; not measured | Science then force if ≥1.5× |
| S7 | Log per-step attn backend + seq length once | **open** | Needed before any flash retry | Do first |
| S8 | Dense stock SDPA without flash | **parked** | ~383 GB abort | — |
| S9 | Sage / kitchen / SLA | **parked** | CUDA; Continuity refuses on MPS | — |
| S10 | ASFP8 fused RoPE | **parked** | Wrong layout for H3 | — |

---

## R — Reference token budget (multiplies every step)

| ID | Opportunity | Status | Evidence | Revisit / next |
|----|-------------|--------|----------|----------------|
| R20 | **`ref_size=match` for video** | **shipped** (forced) | Video `max` only multiplies tokens; no quality past gen canvas. Compile coerces `max`→`match`; UI hides detail picker for clips; attach defaults to match | Old blobs with max silently get match (faster) |
| R21 | Compressed RefMods for image refs | **keep** | — | — |
| R22 | Shorter video ref trim | **shipped** (workflow) | 5 s trim verified in cache timestamps | Keep using trim |
| R23 | Drop unused ref modalities | **open** | — | — |

---

## D — DiT / schedule (non-attn)

| ID | Opportunity | Status | Evidence | Revisit / next |
|----|-------------|--------|----------|----------------|
| D10 | TaoMate 3-step already on | **shipped** | — | — |
| D11 | Lower resolution for draft | **open** | — | — |
| D12 | Shorter duration while iterating | **open** | — | — |
| D13 | Compile / graph DiT blocks on MPS | **open** | — | After S1 re-time |

---

## Science — 2026-09-24 (H3 56×128, fp16 flash vs fp32-upcast sub-quad)

| S | sub-quad+upcast | flash | speedup | match |
|---|-----------------|-------|---------|-------|
| 512 | 9.0 ms | 1.9 ms | **4.6×** | ok |
| 1024 | 19.1 ms | 5.5 ms | **3.5×** | ok |
| 2048 | 58.1 ms | 18.8 ms | **3.1×** | ok |
| 4096 | 178.5 ms | 70.0 ms | **2.6×** | ok |
| 8192 | 582.0 ms | 270.5 ms | **2.2×** | ok |
| 16384 | 1951.7 ms | 1089.0 ms | **1.8×** | ok |

Approx packed S for 8 s @ 512×896 + 5 s video ref ≈ 30–36k → expect ~1.5–2× on attention alone from S1; R20 stacks when ref canvas ≫ gen.

---

## Stacking policy

1. **Live sampler = patched sub-quad** until flash has a visual A/B pass.  
2. **R20 forced** for video (`match`).  
3. Never re-expose sage/kitchen/sla/laya or video `max` as casual UI.  
4. Next measure: EasyCache / FBC (S6); force if it wins. Flash only after HIT log + clean sample.

---

## Run log

### 2026-09-24 — diagnosis after H3_00037_

- Sub-quad backend; mtlflashattn present but on unused `F.sdpa` path.  
- Sampler 92% of wall; 5 s ref trim confirmed (not full file).  

### 2026-09-24 — S1 live: noise / rollback

- Forced flash (global rebind) either failed to engage (~11 min/step, no HIT)
  or when engaged produced **noise** on preview.  
- **Rolled back:** `protect_h3_mps` no longer calls `_patch_metal_flash_attention`.  
- Known-good path restored (sub-quad + Continuity MPS patches). Restart required.
