"""Register every on-disk H3 location before ComfyUI walks model folders.

ComfyUI runs this file before importing the node pack. The files this Mac
fork needs may sit in ComfyUI/models, the Hugging Face hub cache, a sibling
h3-ws checkout, or (legacy) ComfyUI-Shared.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from creator import metal  # noqa: E402

metal.register_search_paths()
