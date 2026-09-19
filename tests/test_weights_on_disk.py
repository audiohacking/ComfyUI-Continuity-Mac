"""`models.check` refuses a needed file that is not on disk.

A remembered SAM3 (or any other) pick that is not under models/ used to pass
the "has a filename been picked" gate and only fail inside the face pass —
after a full sample. The check has to open the path before anything is queued.

    python3 tests/test_weights_on_disk.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import layout  # noqa: E402
from harness import check, passed  # noqa: E402

models = layout.load("models", "h3_models")
core = models.models
h3slots = models.h3_models


class _Weights:
    def __init__(self, picked):
        self.slots = h3slots.SLOTS
        self._picked = picked

    def get(self, name):
        return self._picked.get(name)


fake = types.ModuleType("folder_paths")
fake._paths = {}


def get_full_path(folder, name):
    return fake._paths.get((folder, name))


fake.get_full_path = get_full_path
sys.modules["folder_paths"] = fake

# Empty pick with the face pass's own sentence.
try:
    core.check(_Weights({"sam3": None}), ["sam3"])
except ValueError as err:
    empty = str(err)
else:
    empty = ""
check("face pass with no SAM3 pick uses the slot's missing sentence",
      "face pass" in empty and "SAM3" in empty, True)

# Remembered name, nothing on disk.
fake._paths.clear()
try:
    core.check(_Weights({"sam3": "sam3.1_multiplex_fp16.safetensors"}), ["sam3"])
except ValueError as err:
    gone = str(err)
else:
    gone = ""
check("face pass refuses a SAM3 pick that is not on disk",
      "not in models/checkpoints" in gone
      and "sam3.1_multiplex_fp16.safetensors" in gone, True)

# Same pick, file present — check returns without raising.
fake._paths[("checkpoints", "sam3.1_multiplex_fp16.safetensors")] = (
    "/tmp/sam3.1_multiplex_fp16.safetensors")
# os.path.isfile needs a real file.
open(fake._paths[("checkpoints", "sam3.1_multiplex_fp16.safetensors")], "wb").close()
try:
    core.check(_Weights({"sam3": "sam3.1_multiplex_fp16.safetensors"}), ["sam3"])
    ok = True
except ValueError:
    ok = False
finally:
    os.unlink(fake._paths[("checkpoints", "sam3.1_multiplex_fp16.safetensors")])
check("face pass accepts a SAM3 pick that is on disk", ok, True)

check("sam3 is an opt-in H3 slot", h3slots.SLOTS["sam3"].optional, True)

# The frontend mirror: a remembered pick that is not in the catalog counts as
# missing, and the face toggle stays off until SAM3 is listed.
layout.skip_without_node()
out = layout.run(r'''
const S = await import("./web/creator/state.js");
const models = { fl2va: "a.safetensors", sam3: "sam3.1_multiplex_fp16.safetensors" };
const required = ["fl2va", "sam3"];
console.log(JSON.stringify({
  emptyOnly: S.missingModels(models, required, "h3"),
  absent: S.missingModels(models, required, "h3",
                          { fl2va: ["a.safetensors"], sam3: [] }),
  present: S.missingModels(models, required, "h3", {
    fl2va: ["a.safetensors"],
    sam3: ["sam3.1_multiplex_fp16.safetensors"],
  }),
  readyNull: S.faceDetectorReady(null),
  readyUnknown: S.faceDetectorReady({}),
  readyEmpty: S.faceDetectorReady({ sam3: [] }),
  readyOk: S.faceDetectorReady({ sam3: ["x.safetensors"] }),
}));
''')
check("missingModels without catalog ignores disk", out["emptyOnly"], [])
check("missingModels flags a pick not in the catalog", out["absent"], ["sam3"])
check("missingModels clears when the pick is on disk", out["present"], [])
check("faceDetectorReady is null before the catalog answers", out["readyNull"], None)
check("faceDetectorReady is null when sam3 is unanswered", out["readyUnknown"], None)
check("faceDetectorReady is false with an empty sam3 list", out["readyEmpty"], False)
check("faceDetectorReady is true when a SAM3 file is listed", out["readyOk"], True)

passed("all weights-on-disk tests passed")
