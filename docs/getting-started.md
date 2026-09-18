# Getting started

## What you need

- Apple Silicon Mac and a recent ComfyUI (Desktop or a local checkout on MPS).
  The model families themselves ship with ComfyUI core (H3 is
  `comfy_extras/nodes_minimax_h3.py`, and the others are core nodes too).
  This pack drives them, it does not carry them.
- No extra Python packages. Cloning the repo is the whole install.
- The weight files for at least one family. See [models.md](models.md).
  NVIDIA / CUDA machines should use
  [the original Continuity](https://github.com/roadmaus/ComfyUI-Continuity)
  instead of this fork.

## Install

ComfyUI Desktop's custom-nodes folder is usually
`~/Documents/ComfyUI/custom_nodes`. Remove any existing Continuity or
MiniMax Creator folder first — two copies of this pack register the same
node ids and neither node shows up.

```
cd ~/Documents/ComfyUI/custom_nodes
git clone https://github.com/audiohacking/ComfyUI-Continuity-Mac continuity
```

That leaves one folder, `continuity`. Restart ComfyUI. Search the node
list for **Continuity Mac**. Dropping that node onto an empty canvas is
enough — the UI mounts on the node itself. An optional starter graph
lives under `example_workflows/`.

Presets and settings live in ComfyUI's `user/` directory, not in the pack
folder.

## Download one family's weights

Pick the family you want to start with and put its files where
[models.md](models.md) says. This fork does not download them. On Apple
Silicon it also searches the Hugging Face hub cache and a sibling h3-ws
`models/` tree when those exist. The minimum for a family you do not already
have is small: H3 video is five files, LTX 2.5 video is four, a Krea 2 still
is four.

The Comfy-Org and Lightricks repositories are laid out like the `models/`
folder already, so a file at `diffusion_models/krea2_raw_bf16.safetensors` in
the repo goes to `ComfyUI/models/diffusion_models/`. Download by path and you
can't put it in the wrong place.

## First render

1. Add the node: double-click the canvas and search for "Continuity Mac".
2. Click the model pill and pick your family and checkpoint. The weights pill
   next to it is where you point each slot at the files you downloaded. Picks
   are remembered per family, so this is a one-time chore.
3. Type a prompt in the box.
4. Press Render.

The finished clip or still lands in `output/continuity/`, filed under the
family that made it, and shows up in the node's own gallery.

If a file is missing, the render is refused before the queue starts, with a
message naming the field and the folder it looks in. That message is the fix:
put the named file in the named folder.

## The fullscreen editor

`Ctrl+Shift+M` opens the node as the whole window. Two views over the same
state:

- **Simple** is one column, for when the piece is one prompt.
- **Full** puts the pre-stage, the shot and the picture side by side, for when
  the piece is built out of parts.

Switching views mid-sentence keeps the sentence. It is the same node either
way.

## Where to go next

- Attaching pictures, clips and sound to a prompt: [the-node.md](the-node.md)
- More than one shot: [timeline.md](timeline.md)
- What each model family can and can't do: [families.md](families.md)
