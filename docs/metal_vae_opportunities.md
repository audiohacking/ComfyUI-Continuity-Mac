# Metal H3 VAE / Ref2VA opportunity log

Living inventory for Continuity Metal. **Nothing is discarded.**
Small wins compound. A row may be `shipped`, `keep` (measured positive,
not yet in live path), `parked` (measured weak / negative — revisit when
trigger fires), or `open` (unmeasured).

Science harness: `creator/metal_vae.py`  
Ladder: `tests/test_metal_vae_science.py`  
Live patches: `creator/metal.py`

Re-run baseline anytime:

```bash
/Users/moysa/Documents/ComfyUI/.venv/bin/python3 tests/test_metal_vae_science.py
```

Update this file when a suite row changes a status. Date entries
`YYYY-MM-DD`. Machine notes: M-series MPS, torch recorded in suite banner.

---

## Status legend

| Status | Meaning |
|--------|---------|
| `shipped` | In live Continuity Metal path |
| `keep` | Measured win (any size); queue for land or stack with others |
| `parked` | Measured flat/negative *for now*; revisit when trigger fires |
| `open` | Not measured yet — still an opportunity |
| `blocked` | Needs infra / restart / full VAE weights / permission |

---

## 1. Encoder cell (ResNet / CausalConv3d fold points)

NVIDIA’s win was fusing this cell. We own the Metal version.

| ID | Opportunity | Status | Evidence | Revisit / next |
|----|-------------|--------|----------|----------------|
| E1 | One-shot fp32 weight/bias promote on MPS (no per-forward `.float()`) | **shipped** | 2026-09-24 science C: **−20% to −42%** vs per-call cast | Confirm wall-clock on real Ref2VA encode after restart |
| E2 | Inplace SiLU after per-frame GN | **keep** | 2026-09-24 C: **−1% to −6%** vs out-of-place | Land in live patch or fold into E4; small but free if exact |
| E3 | Single-pad helper when `front==0` (merge reflect pads) | **open** | Identity exists (`eager_norm_silu_pad_one_pad`); timing not isolated when front≠0 (H3 causal usually front>0) | Microbench front=0 downsample paths; early stem |
| E4 | Metal fused `group_norm + SiLU + pad` (fp32, NCDHW out) | **open** | H1 short SUPPORT (pre 39–46% of cell); H1 medium weak (14–20%) | Write Metal/MPS custom op or MLX; A/B vs eager on short *and* medium |
| E5 | Fuse residual add into conv epilogue | **open** | NVIDIA kitchen folds residual; we time residual alone (~0.2 ms short) | Measure `conv+add` vs fused; land with E4 or torch addcmul |
| E6 | Permanent fp32 Module dtype (not only weights) | **parked** | Whole `encoder.float()` / clone OOMed historically | Only if param-only promote (E1) insufficient and RAM headroom measured |
| E7 | `channels_last_3d` for MPS conv3d | **parked** | 2026-09-24 H3 REJECT (cl equal or slower) | Re-test each major torch MPS release; ROCm also lost on NDHWC |
| E8 | fp16 / bf16 activations on encode | **parked** | Field: fp16 full VAE NaN → black Ref2VA. Toy cell finite on torch 2.10 (E ladder) | Full MiniMax encoder A/B only; never relax on toy-cell evidence |
| E9 | fp16 accumulate conv (NVIDIA `--fast`) | **parked** | CUDA/CUTLASS; conflicts with E8 policy | Only if E8 cleared *and* MPS grows accumulate API |
| E10 | Eliminate permute/view copies in per-frame GN | **open** | GN path: permute→view→GN→view→permute | Custom kernel or channels-first GN; measure copy bytes |
| E11 | Prefetch / overlap GN of next block while conv runs | **open** | MPS command buffer overlap unknown | Instruments GPU trace on one encode |
| E12 | Shared scratch buffer across ResNet blocks (no alloc per SiLU/pad) | **keep** | Related to E2; buffer-reuse candidate sketched then simplified | Implement + measure alloc rate via `torch.mps` capture |
| E13 | Depthwise / grouped early stem specialisation | **open** | Unmeasured | Profile full encoder layer list for 1×1 vs 3×3 share |
| E14 | torch.compile / aot_eager on encoder subgraph | **open** | aot_eager works on MPS smoke; inductor Metal immature | Compile one ResnetBlock3D; check graph breaks |
| E15 | Quant_conv (1×1) same promote + fuse with last norm | **shipped** (promote) / **open** (fuse) | E1 covers promote for `quant_conv` | Fuse last GN+quant_conv if present |

---

## 2. Full video VAE encode (end-to-end)

| ID | Opportunity | Status | Evidence | Revisit / next |
|----|-------------|--------|----------|----------------|
| V1 | Wall-clock A/B: Continuity patch vs stock eager on real clip | **blocked** | Needs Comfy with `comfy.ldm.minimax.vae` + restart permission | Desktop was 0.18.5 without H3 VAE module; use newer core or vendored vae |
| V2 | Encode at `ref_size=match` vs `max` (already a Continuity knob) | **open** | encode.py documents match as speed | Document default guidance for Metal Ref2VA; measure token×time |
| V3 | Stride / frame subsample before VAE for long refs | **open** | media already caps length to card | Optional 2× temporal subsample experiment with quality A/B |
| V4 | Chunked temporal encode (stream T in windows) | **open** | Might reduce peak memory / improve cache | Correctness vs causal pad across chunk boundaries |
| V5 | Skip re-encode when latent cache hits (already) | **shipped** | Continuity `latents` / `_cached` | Ensure NaN entries never cached (already) |
| V6 | Parallel encode independent refs (multi-image) | **open** | Ref2VA often many stills | MPS concurrency / memory; measure 4× still encode |
| V7 | Pin encoder weights in GPU, avoid reload between refs | **open** | Model management thrash | Trace load/unload during multi-ref |
| V8 | MLX rewrite of EncoderFCN3D | **open** | Graph fusion MLX may beat MPS eager | Prototype one level; parity tests vs metal_vae eager |

---

## 3. Video VAE decode

| ID | Opportunity | Status | Evidence | Revisit / next |
|----|-------------|--------|----------|----------------|
| D1 | Batched tile decode (Comfy #16187 portable idea) | **open** | NVIDIA PR batches up to 4 tiles by free VRAM | Port Python batching to Metal path; no CUDA needed |
| D2 | int8 VAE weights + MPS/MLX matmul | **open** | NVIDIA int8 decode ~1.4×; kitchen CUDA kernels | Try weight-only quant file on MPS; measure + PSNR vs fp16 |
| D3 | Fuse RMSNorm + linear / SwiGLU in ViT decoder | **open** | kitchen `linear_input_act` | Metal or MLX fused linear; start eager fuse |
| D4 | Decoder attention: keep default Metal SDPA | **shipped** (policy) | Sage/kitchen refused on MPS | Revisit when MPS flash-attn quality proven |
| D5 | bf16 decode activations (encode stays fp32) | **open** | Decode may tolerate lower prec | Finite + PSNR ladder before land |
| D6 | Write preview once, not every decode | **shipped**-ish | encode `_mod_tensors` writes preview if missing | Audit double preview work on RefMod path |

---

## 4. Ref2VA pipeline (encode.py / media / cache)

| ID | Opportunity | Status | Evidence | Revisit / next |
|----|-------------|--------|----------|----------------|
| R1 | Compressed RefMods (fewer DiT tokens) | **shipped** | Continuity RefMod compress | Encourage in Metal docs; measure step time vs full |
| R2 | Card-length trim for video + audio refs | **shipped** | 3.0.4 media clamp | — |
| R3 | Latent reference cache | **shipped** | `_cached` + fingerprints | Stats: hit rate logging optional |
| R4 | Drop NaN cache entries | **shipped** | 3.0.2 | — |
| R5 | Presentation subsample (2 fps) already | **shipped** | encode path | Measure tokenizer cost vs latent cost |
| R6 | Avoid decoding mod→pixels when tokenizer can take latent-only | **open** | `_mod_tensors` always decodes for `<Picture N>` | Model API constraint; watch Comfy |
| R7 | Batch still-ref VAE encodes | **open** | Same as V6 | — |
| R8 | Audio VAE path microbench | **open** | Unmeasured vs video VAE | Science suite sibling for audio |
| R9 | DeepStack mask guard | **shipped** | 3.0.4 | — |
| R10 | MPS attn elem-cap / no slow multi-KV | **shipped** | 3.0.4 | — |
| R11 | ASFP8 rope/norm disarm | **shipped** | metal.py | — |
| R12 | Watermark shield on large RAM | **shipped** | metal.py | — |

---

## 5. Kitchen / NVIDIA ideas — Metal adaptations (not ports)

| ID | Opportunity | Status | Evidence | Revisit / next |
|----|-------------|--------|----------|----------------|
| K1 | `group_norm_silu_pad3d` semantics in Metal | **open** | = E4 | Contribute upstream kitchen Metal backend long-term |
| K2 | `fp16_conv3d` accumulate | **parked** | = E9 | — |
| K3 | `int8_attention` decoder | **parked** | CUDA; refused on MPS | MLX int8 attn experiment under D2/D4 |
| K4 | HIP lesson: do not force NDHWC | **shipped** (policy) | H3 REJECT + ROCm report | Keep in metal_vae design rules |
| K5 | comfy-kitchen Metal backend | **open** | Backends today: eager/cuda/triton/hip | After E4 proven in Continuity |

---

## 6. Tooling / methodology (force compounding)

| ID | Opportunity | Status | Evidence | Revisit / next |
|----|-------------|--------|----------|----------------|
| T1 | Science ladder A–E | **shipped** | test_metal_vae_science.py | Add phases F+ as opts land |
| T2 | Auto-append CSV of timings per run | **open** | — | `results/metal_vae_*.csv` for trend |
| T3 | Full-encoder layer-by-layer profile | **open** | — | Hook every ResnetBlock once V1 unblocked |
| T4 | PSNR / finite gates on every candidate | **shipped** (B) | atol=0 for inplace | Keep gates when landing keep→shipped |
| T5 | This opportunity log | **shipped** | this file | Update on every science run / land |

---

## 7. Stacking policy (small adds up)

1. **Never delete a `keep` row** — land it or leave it `keep`.
2. **`parked` needs a revisit trigger** (torch version, shape class, RAM, or new API).
3. Prefer landing several `keep` items together (e.g. E1+E2+E12) and re-measure the bundle — interactions matter.
4. End-to-end (V1) is the judge for “adds up”; cell benches are the scout.
5. When in doubt, add an `open` row here before writing code.

---

## 8. Run log (append-only)

### 2026-09-24 — first science ladder (MPS, torch 2.10.0)

- A short: H1 SUPPORT, H2 SUPPORT, H3 REJECT  
- C: inplace SiLU −1..−6%; weight cache −20..−42%  
- D medium: pre_share 14–20% (conv-dominated)  
- E: toy fp16 cells finite (small / ref2vaish / long_t) — **do not** clear E8  
- Landed: E1 one-shot fp32 promote in `metal._patch_h3_video_vae_mps_encode`  
- Queued keeps: E2, E12  
- Priority opens: E4 (small maps), D1, V1, T2, R8  

---

## 9. Suggested next measurement queue

1. Land E2 (inplace SiLU) behind identity gate; re-run C.  
2. Add T2 CSV logging.  
3. Prototype E4 Metal/MLX fused GN+SiLU+pad on `SHAPES_SHORT` only.  
4. Unblock V1 (newer Comfy core or vendored `vae.py`) + restart permission → wall clock.  
5. D1 tile-batch decode microbench.  
6. R8 audio VAE sibling suite.  
7. Revisit E7/E8 on next torch bump.
