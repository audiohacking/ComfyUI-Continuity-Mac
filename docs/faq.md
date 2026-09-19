# FAQ and troubleshooting

## Settings

The gear on the node's rail opens the pack's settings.

- **Where files go** is a per-machine setting (the Folders tab), not part of
  the workflow, with `%year%`-style tokens. Every family files into a folder
  of its own, and each has a row to override.
- **MP4 quality** is a setting too, on the same page. Two people opening the
  same workflow get the same shot without having to agree on how many
  megabytes it takes.
- **Language** follows ComfyUI's own locale: English, Japanese, Korean,
  Simplified Chinese. Corrections are one-line edits in
  `web/creator/locales/`.
- **Rendering** holds the drift levers for long strips: the turbo lead-in,
  the seam handoff and the DLSS 5 pass. See [Seams and
  drift](timeline.md#seams-and-drift) for what each was measured to do.
- **Appearance** has a text size, and the pack takes its colours from
  ComfyUI's palette.
- **Stored data** lists everything the pack has written down, with a count
  beside each one and a press to remove it: the preset library scope by scope,
  the stars and the LoRA notes this browser holds, the reference cache, the
  refiner's server, and the settings themselves. Nothing there deletes a
  render, a reference or a workflow: those are files.

## Common errors

### I installed Continuity and now no node shows up at all

Look in `ComfyUI/custom_nodes` for a second copy: the CUDA pack
(`continuity`, `ComfyUI-Continuity`, `ComfyUI-MiniMax-Creator`) sitting
beside this fork (`continuity-mac`). The node ids are the same on purpose,
so saved workflows keep loading. Two folders registering those ids means
neither node shows up. The startup console log says so.

This fork is the one Mac install. Delete the other copy and restart. Search
for **Continuity Metal**. NVIDIA users should be on
[the original](https://github.com/roadmaus/ComfyUI-Continuity), not here.
Nothing you made is in either folder, since presets, settings, favourites and
LoRA memory sit in ComfyUI's `user/` directory. If the copy you want gone came
from the ComfyUI Manager, uninstall it there.

### "Render refused, naming a field and a folder"

Not a bug: a weight file is missing. Put the file it names in the folder it
names. [models.md](models.md) has every file.

### H3 video is only noise / static

This pack forces the Metal path. After restart the console must say
`forced MPS H3 attention` and must **not** say
`[AppleSilicon-FP8/rope-fast] fused RoPE active`. Three separate MPS bugs
all look like noise and none of them log an error:

- AppleSilicon-FP8 fused RoPE takes `L` from `x.shape[-2]`. H3 Q/K is
  `[B, S, heads, dim]`, so every token is rotated by head index. Leave
  that kernel off until a wrap permutes to `[B, heads, S, dim]`.
- Sub-quadratic attention seeds scores from `torch.empty`; MPS
  `baddbmm(beta=0)` broadcasts those NaNs ([ComfyUI#15804](https://github.com/Comfy-Org/ComfyUI/issues/15804)). H3 is bf16, and
  ComfyUI's macOS upcast only covered fp16.
- A machine with hundreds of GB of unified memory never chunks, so the QK
  matrix crosses MPS's 32-bit index wall and corrupts silently
  ([ComfyUI#14837](https://github.com/Comfy-Org/ComfyUI/issues/14837)).

Do not switch to pytorch SDPA: H3's packed sequence tried to allocate a
several-hundred-GB buffer and aborted. After RoPE the layout *is* what
SDPA expects (`[B, heads, S, dim]`), so the next speed path is
[mtlflashattn](https://github.com/pawel-mazurkiewicz/mtlflashattn) (never
forms QK) gated so a kernel miss cannot fall back to dense SDPA — not
turning stock SDPA on. The live preview also needs madebyollin's
[`taeh3.safetensors`](https://github.com/madebyollin/taehv/blob/main/safetensors/taeh3.safetensors)
(~22 MB, keys `decoder.1.weight`) in `models/vae_approx` — a 320 MB SD VAE
dumped under that name is latent2rgb mush, not the shot.

Leave attention on **default**. Sage, kitchen int8, SLA, Spectrum and
fp16 accumulation are NVIDIA paths and this pack refuses them on Apple
GPU; chunked FFN and the step caches still run. Turbo on this fork
offers TaoMate (`taomate_h3_3step_comfy.safetensors` at strength 0.8,
Euler/simple, **3 steps** — re-throw turbo after a restart to pick that
up). LightX2V and Tutu still use the family's 4 / 6 / 8. Do not pick
FastH3, NVFP4, ConvRot or `fp8_scaled` H3 files — those are the CUDA
packed stack. h3-ws runs native MiniMax-H3 FL2VA; ComfyUI wants the
matching Comfy-Org `*_pruned_bf16` DiTs and
`qwen3vl_32b_minimax_h3_bf16.safetensors`. The weights live under
ComfyUI's `models/` tree (and the Hugging Face hub cache / h3-ws, which
the pack also searches).

### MPS OOM with ~56 GB allocated and ~400 GB "other"

The model is not 400 GB. H3 on other boxes runs in well under
128 GB. AppleSilicon-FP8 caps the MPS allocator at 80–100% of
Apple's recommended_max so a 16 GB Mac does not swap; on a 512 GB
Mac that reserve is ~407 GB of empty pool, and a 544 MB VAE tile
is refused. This pack overrides that watermark. Restart once.

### Ref2VA comes out black (or dies at save with AAC NaN)

The Metal attention patches already apply to both FL2VA and Ref2VA.
What Ref2VA does that FL2VA does not is encode the reference *video*
through the H3 video VAE. That encode is NaN on MPS; the pack used
to cache it and the sampler then produced a black clip (and a
soundtrack AAC refused). Multi-frame video encode stays on the GPU
and runs the encoder in fp32. Restart once, then re-queue — the
poisoned cache entries are dropped automatically. The first
re-encode writes a finite cache entry; after that it hits again.

### Ref2VA dies at save with `Input contains (near) NaN/+-Inf`

If the clip is not black, this is the muxer: AAC refused a
non-finite soundtrack. The pack replaces those samples so the mp4
still writes. Restart once if you have not since that fix.

### CUDA OOM with `HostBuffer.read_file_slice` on a long render

Recent ComfyUI streams weights with Dynamic VRAM by default. Start ComfyUI
with `--disable-dynamic-vram`
([ComfyUI#15255](https://github.com/Comfy-Org/ComfyUI/issues/15255)).

### fp8 isn't any faster

fp8 only speeds up sampling on cards with hardware fp8 matmul (RTX 40-series
and later). On older cards it still halves the checkpoint's memory.

### References refused on Ideogram 4.0

Ideogram reads no reference conditioning, and a render that silently ignored
your images would be worse than one that says so. Switch the model pill to
another stills family, or clear the references.

### References do nothing on LTX 2.5

Citing a reference on LTX 2.5 needs the Ingredients IC-LoRA in `models/loras`.
Lightricks hasn't released a 2.5 version, so use
[the 2.3 one](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients),
which is what this pack is tested against.

### The refiner refuses H3's text encoder

By design. H3's 32B encoder is truncated to its hidden states and has no head
to decode text with. Use a Qwen3-VL 4B or 8B (the Krea 2 and Ideogram
encoders are exactly that), any Qwen3.5 text encoder, or point the refiner at
a server.

### My 6-second H3 video is 5.9 or 6.1 seconds

H3's frame count has to satisfy `n % 17 == 5` at 24 fps, so not every whole
second exists. The pill shows whole seconds and the compiler lands on the
nearest legal count.

### GGUF files don't show up

They appear once [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) is
installed. Same folder as the safetensors, picked the same way.

### An accelerator pill is missing

The cache pills, sage attention and the device chips belong to optional packs
(see the Thanks list in the README). They light up when the pack is
installed. `easy` (core's EasyCache) and `kitchen` (core's int8 attention)
need nothing installed, though `kitchen` only appears on builds that ship the
kernel.

## Other questions

### Was this pack called something else?

Yes, MiniMax Creator, back when MiniMax H3 was the only family it drove.
GitHub redirects the old address, so an existing clone still pulls, but it is
worth repointing:

```
git remote set-url origin https://github.com/roadmaus/ComfyUI-Continuity.git
```

Saved workflows, node ids, widget names and output folders are all unchanged.
Old graphs load and old files stay where they are.

### Does anything leave my machine?

No. Rendering is local open weights through ComfyUI core, and nothing is
uploaded. The one exception is opt-in: the refiner can run on a server of
your own - LM Studio, Ollama, or a hosted API with your key - and those
requests go to the server you chose, references included when the model can
see them.

### Where do renders go?

`output/continuity/`, filed per family (`renders/ltx25/`, `stills/krea2/`),
with takes under `takes/` and upscales under `upscaled/`. All of it
overridable in settings.

### Can I add a model family?

A family is a package under `creator/families/` with a `declare.py` the
registry picks up; adding one doesn't mean touching the node. If there is a
model you want in here, open an issue.
