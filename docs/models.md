# Model downloads

Nothing is bundled. You put files where ComfyUI already looks and pick them
on the node's **weights** pill. If a render needs a file it doesn't have, it
is refused before the queue starts, and the message names the field and the
folder.

A prestartup also searches the Hugging Face hub cache and a sibling
[h3-ws](https://github.com/lmangani/h3-ws) `models/` tree when those exist, so
a file you already downloaded does not have to be copied. New files go in
`ComfyUI/models/` (Desktop: `~/Documents/ComfyUI/models`).

A few notes that apply everywhere:

- Pick **one** weight per slot. The quantizations in a repo (`bf16`,
  `fp8_scaled`, `int8_convrot`, `nvfp4`) are alternatives, not a set. On
  Apple GPU this fork wants **bf16** (or `fp16` / `fp32` for the VAEs).
- `fp8` only speeds up sampling on NVIDIA cards with hardware fp8 matmul.
  On Apple GPU it is not a speed path — do not pick `fp8_scaled` H3 files.
- `nvfp4` and FastH3 / FastVideo student DiTs are CUDA stacks. They decode
  as noise on Metal. NVIDIA users should be on
  [the original Continuity](https://github.com/roadmaus/ComfyUI-Continuity).
- GGUF files work if [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) is
  installed. Drop the `.gguf` in the same folder and pick it like any other
  file.

All folders below are under `ComfyUI/models/`.

## MiniMax H3 (video with sound, and stills)

Packed ComfyUI safetensors from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) — not the
native MiniMaxAI trees (those are for h3.c). Leave attention on **default**.

| Slot | File | Folder |
|---|---|---|
| FL2VA checkpoint | `minimax_h3_fl2va_pruned_bf16.safetensors` | `diffusion_models` |
| Ref2VA checkpoint | `minimax_h3_ref2va_pruned_bf16.safetensors` | `diffusion_models` |
| Text encoder | `qwen3vl_32b_minimax_h3_bf16.safetensors` | `text_encoders` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `vae` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `vae` |

You need both checkpoints: H3 routes between them based on what you attach.

Tighter on RAM: the matching `*_pruned_int8_convrot.safetensors` DiTs and
`qwen3vl_32b_minimax_h3_int8_convrot.safetensors` encoder from the same repo.
ComfyUI emulates those on M-series; they are not the FastH3 student.

### Turbo (TaoMate 3-step)

This fork's turbo default is **not** the LightX2V files in Comfy-Org's
`loras/`. It is TaoMate, stacked on the **base** FL2VA above — not on a
FastH3 student.

From [Robert1212star/TaoMate-H3-3Step-ComfyUI](https://huggingface.co/Robert1212star/TaoMate-H3-3Step-ComfyUI):

| File | Folder |
|---|---|
| `taomate_h3_3step_comfy.safetensors` | `loras` |

Strength 0.8, Euler / simple, **3 steps** — that is the distill, same as
h3-ws. Draft / med / good hide on TaoMate because they would all write 3.
Another turbo LoRA (LightX2V, Tutu, PDD) still owns its own quality table.

Optional filename presets still match LightX2V
([lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo),
the `*_comfyui_bf16.safetensors` files) and Tutu
([tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA](https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA)
under `comfyui/`).

### Fetch

The Comfy-Org repo is laid out like `models/` already. From the ComfyUI root:

```bash
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors \
  diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir models

hf download Robert1212star/TaoMate-H3-3Step-ComfyUI \
  taomate_h3_3step_comfy.safetensors \
  --local-dir models/loras
```

Or `python3 scripts/download_metal_h3.py` from this pack (writes the same
five Comfy-Org files plus TaoMate into `~/Documents/ComfyUI/models`).

Optional: [`taeh3.safetensors`](https://github.com/madebyollin/taehv/blob/main/safetensors/taeh3.safetensors)
in `vae_approx` gives H3 a properly decoded live preview.

Optional: a **guide LoRA** for the sampler row's guide pill — a file trained
with the source clip as an aligned guide. Alissonerdx's
[Minimax-H3-ComfyUI](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI)
carries `minimax_h3_lms_v1.0_r64.safetensors` (a sharpener) and
`minimax_h3_style_transfer_v1.0_r64.safetensors` under `loras/`; both go in
`models/loras` and both were trained against Ref2VA.

Optional: a **trained latent upscaler** for the resolution pill's two passes.
LBH-123-AI's
[Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
(`minimax_h3_latent_upscaler_3d_fp16.safetensors`, 691 MB) goes in
`models/latent_upscale_models`. Pick it under the node's weights control, then
switch the resolution popover's *drawn up by* to **trained**: the refine pass
draws the first pass up with it instead of bicubic, and holds a face at a
lighter refine (0.30, 3 steps) than bicubic can. Nothing changes without it.

Optional: a **VDN-H3 stage** for the sampler row's VDN pill. A stage is a
directory, not a file, and it goes under `models/vdn/` with its layout intact
(`model_spec.json`, `linear_branch/`, `adapters/`). The bf16 release is
[OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3)
(`stage-dmd-step-250/` is the 8-step model, `stage-b-step-2000/` the 50-step
one); the INT8 ConvRot repack of the 8-step stage at
[drbaph/vdn-minimax-h3-int8-convrot-comfyui](https://huggingface.co/drbaph/vdn-minimax-h3-int8-convrot-comfyui)
is half the size with identical output and is the one to take on 24 GB and
below:

```bash
hf download drbaph/vdn-minimax-h3-int8-convrot-comfyui --local-dir ComfyUI/models/vdn/vdn-minimax-h3-int8-convrot-comfyui
```

The directory name is what the pill lists. A stage is branch weights and two
adapters over the H3 checkpoints you already have, not a base of its own, and
the weights are under the MiniMax H3 Community License — read it first.

## LTX 2.5 (video with sound)

From [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5):

| Slot | File | Folder |
|---|---|---|
| Transformer | `ltx-2.5-22b-distilled-transformer-*.safetensors` | `diffusion_models` |
| Text encoder | `gemma4-12b-with-proj-ltx-2.5-*.safetensors` | `text_encoders` |
| Video VAE | `ltx-2.5-video-vae-bf16.safetensors` | `vae` |
| Audio VAE | `ltx-2.5-audio-vae-bf16.safetensors` | `vae` |

Optional files, each unlocking one control:

| Unlocks | File | Folder |
|---|---|---|
| The seconds pill's **auto** | `ltx-2.5-duration-head-bf16.safetensors` | `model_patches` |
| The resolution pill's **two passes** | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `latent_upscale_models` |
| **ReDetail** | [LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler) | `loras` |
| References | [LTX-2.3-22b-IC-LoRA-Ingredients](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients) | `loras` |

About that last row: citing a reference on LTX 2.5 needs the Ingredients
IC-LoRA, which is what makes a reference sheet mean anything to the
transformer. Lightricks hasn't released a 2.5 Ingredients yet, so use the 2.3
one. Most 2.3 IC-LoRAs load and work on 2.5, and that file is what this pack
was built and tested against. A shot with no references never asks for it.

## Krea 2 (stills)

From [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2):

| Slot | File | Folder |
|---|---|---|
| Checkpoint | `krea2_raw_*.safetensors` | `diffusion_models` |
| Turbo checkpoint | `krea2_turbo_*.safetensors` | `diffusion_models` |
| Text encoder | `qwen3vl_4b_*.safetensors` | `text_encoders` |
| VAE | `qwen_image_vae.safetensors` | `vae` |

## Ideogram 4.0 (stills)

From [Comfy-Org/Ideogram-4](https://huggingface.co/Comfy-Org/Ideogram-4):

| Slot | File | Folder |
|---|---|---|
| Checkpoint | `ideogram4_*.safetensors` | `diffusion_models` |
| Unconditional checkpoint (optional) | `ideogram4_unconditional_*.safetensors` | `diffusion_models` |
| Text encoder | `qwen3vl_8b_*.safetensors` | `text_encoders` |
| VAE | `flux2-vae.safetensors` | `vae` |

## Qwen Image Edit (stills, edited from a picture)

From [Comfy-Org/Qwen-Image-Edit_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI):

| Slot | File | Folder |
|---|---|---|
| Checkpoint | `qwen_image_edit_2511_*.safetensors` (or `2509`) | `diffusion_models` |
| Text encoder | `qwen_2.5_vl_7b_*.safetensors` | `text_encoders` |
| VAE | `qwen_image_vae.safetensors` | `vae` |

The turbo pill wants a
[Lightning LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)
in `loras`, matched to the edition of your checkpoint. There is no distilled
checkpoint for this family; the LoRA is the whole speed axis. The VAE is the
same file Krea 2 loads.

## Flux 2 Klein (stills, edited from a picture)

Published by [Black Forest Labs](https://huggingface.co/black-forest-labs) at
two sizes: 4B (Apache 2.0) or 9B (non-commercial).

| Slot | File | Folder |
|---|---|---|
| Checkpoint | `flux-2-klein-base` (4B or 9B) | `diffusion_models` |
| Turbo checkpoint | the 4-step distilled file, same sizes | `diffusion_models` |
| Text encoder | `qwen_3_4b` or `qwen_3_8b`, matched to the checkpoint size | `text_encoders` |
| VAE | `flux2-vae.safetensors` | `vae` |

The VAE is the same file Ideogram loads. The turbo pill swaps in the distilled
checkpoint; there is no turbo LoRA for this family.

## Cutouts and the faces pass

| For | File | Folder |
|---|---|---|
| One-click cutouts | [`birefnet.safetensors`](https://huggingface.co/Comfy-Org/BiRefNet) | `background_removal` |
| Click-to-select cutouts, the faces pass, and the Matte tracing | [`sam3.1_multiplex_fp16.safetensors`](https://huggingface.co/Comfy-Org/sam3.1) | `checkpoints` |

BiRefNet mattes the most prominent subject with no clicks, which is enough for
a picture with one subject in it. SAM 3 is for saying which subject you mean,
and it is also the file the per-frame faces pass and the ControlNet bench's
Matte tracing need.

## The Refine button

Any Qwen3-VL 4B or 8B in `text_encoders`, for example
`qwen3vl_4b_bf16.safetensors`, or any Qwen3.5 (0.8B to 27B) that ComfyUI
loads as a text encoder. If you already have Krea 2 or Ideogram 4.0
installed, you're done: their text encoders are those exact files.

H3's own 32B encoder is not a candidate. It is truncated and has no head to
decode text with, and the picker says so if you choose it.

Or skip the file: Refine can point at any OpenAI-compatible server instead.
See [the-node.md](the-node.md#refine).

## The ControlNet bench

The five arithmetic tracings need nothing. Depth and Pose run a model:

| Tracing | File | Folder |
|---|---|---|
| Depth | any Depth Anything 3 model | `geometry_estimation` |
| Pose | `sdpose_wholebody_fp16.safetensors`, plus any SD 1.5 VAE | `checkpoints`, `vae` |

Matte uses the SAM 3 checkpoint from the cutouts table above.

## The upscale bench

| Backend | File | Folder |
|---|---|---|
| Sharpen | any GAN upscaler spandrel loads (`RealESRGAN_x4plus`, `4x-UltraSharp`, DAT, SwinIR, SPAN) | `upscale_models` |
| Restore | `seedvr2_3b_int8_convrot.safetensors` and `seedvr2_ema_vae_fp16.safetensors` (from Comfy-Org/SeedVR2) | `diffusion_models`, `vae` |
| Refine (DLSS 5) | `dlssnr-weights-logical.safetensors`, extracted on the settings page from your own `nvngx_dlssnr.dll` — see below | `dlss` |

## The neural refiner (DLSS 5)

No download. The weights are NVIDIA's, inside `nvngx_dlssnr.dll` file
version 310.8.0.0 (SHA-256 `ceb6432f…2650`), which NVIDIA ships in its
Streamline SDK and with games that carry DLSS 5. This pack neither hosts nor
bundles that file or anything derived from it: the settings page checks the
DLL you point it at by hash and runs the
[MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS) port's extraction tool on it
locally, writing `models/dlss/dlssnr-weights-logical.safetensors`. The port's
code is carried in the pack, so nothing else is installed. See
[tools.md](tools.md#neural-refiner-dlss-5).
