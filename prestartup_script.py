"""Register every on-disk H3 location before ComfyUI walks model folders.

ComfyUI runs this file before importing the node pack. The files this Mac
fork needs may sit in ComfyUI/models, the Hugging Face hub cache, a sibling
h3-ws checkout, or (legacy) ComfyUI-Shared.

`shield_h3_mps` has to run here: AppleSilicon-FP8 installs fused RoPE and
fused RMSNorm from its `__init__`, after every prestartup. The env flags
must already be off. Attention monkeypatches wait until the pack import,
when ComfyUI core is loaded — this file must not import torch first.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from creator import metal  # noqa: E402

metal.register_search_paths()
metal.shield_h3_mps()
print("[Continuity Mac] AppleSilicon-FP8 fused RoPE and fused RMSNorm off. "
      "H3 Q/K is [B,S,heads,dim]; that RoPE kernel indexes by heads. "
      "Attention patches apply when this pack imports.",
      flush=True)
