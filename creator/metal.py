"""Apple Metal extras for this Mac fork: where H3 files live, and turbo.

Continuity still drives ComfyUI core. On this machine the packed files sit in
ComfyUI's own `models/` tree, may also exist under the Hugging Face hub cache,
and LoRAs may live in the sibling h3-ws checkout. `register_search_paths`
adds every one of those that exists so a picker sees the file once, wherever
it landed.

Turbo on this fork is TaoMate's 3-step adapter — the file already in
`models/loras/taomate_h3_3step_comfy.safetensors`.
"""

from __future__ import annotations

import os
from pathlib import Path

HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
COMFY_MODELS = Path.home() / "Documents" / "ComfyUI" / "models"
SHARED_MODELS = Path.home() / "ComfyUI-Shared" / "models"

# Packed Comfy-Org MiniMax-H3 as huggingface_hub stores it.
HF_COMFY_ORG_H3 = "models--Comfy-Org--MiniMax-H3"

TURBO_LORA = "taomate_h3_3step_comfy.safetensors"

# Guess-time exclusions that are still true on a Mac: the two VAEs share a
# folder, and the T=1 image decoder is not the video VAE.
SLOT_AVOID = {
    "vae": [r"t1[_-]?image", r"image[_-]vae", r"audio"],
}

# Filename presets the turbo switch matches. TaoMate first: it is this fork's
# default distill, 3 Euler steps at 0.8 on the base (or student) DiT.
LORA_PRESETS = (
    {"match": r"taomate", "strength": 0.8,
     "shift_video": 12, "shift_audio": 3,
     "row": {"sampler_name": "euler", "scheduler": "simple"},
     "steps": {"draft": 3, "medium": 3, "good": 3},
     "note": "TaoMate 3-step: 3 Euler steps at strength 0.8."},
    {"match": "lightx2v", "strength": 0.6, "shift_video": 6, "shift_audio": 3},
    {"match": r"tutu|20to8-nfe|20to8_nfe", "strength": 0.8,
     "shift_video": 12, "shift_audio": 3,
     "row": {"sampler_name": "euler", "scheduler": "beta"},
     "steps": {"draft": 8, "medium": 8, "good": 8},
     "note": "Tutu 20→8 NFE: 8 Euler steps at strength 0.8."},
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
        # h3-ws keeps a packed FastH3 student under fasth3-live-ref.
        if _add("diffusion_models", Path(root) / "fasth3-live-ref"):
            added.append(f"diffusion_models:{Path(root) / 'fasth3-live-ref'}")
    return added
