"""A sampler row is a family's, on the still side and in the chat.

    python3 tests/test_still_row_switch.py

Krea and Ideogram spell `cfg` the same and run it an order apart, so the row
must not follow the arch pill — and must not be forgotten either, or looking
at Ideogram means re-dialling Krea. `PreStageRow.setArch` sets the leaving
arch's row aside under its name and hands it back on return, the way
`state.setFamily` does for a piece. The chat room's pinned copy goes through
the same switches (`Sync.moveStill` / `moveVideo`), because writing the rail
alone left Ideogram's guidance in force on Krea. Runs the real modules over
the DOM shim. No server.
"""

import layout
from domshim import DOM
from harness import check

layout.skip_without_node()

SCRIPT = r'''
globalThis.app = { extensionManager: { setting: { get: () => "en" } } };
const S = await import("./web/creator/state.js");
const { PreStageRow } = await import("./web/creator/prestage.js");
const { blobIO } = await import("./web/creator/sampling.js");
const { Sync } = await import("./web/creator/chatnode.js");
const out = {};

const ioOver = (state) => blobIO(() => ({}), () => state.sampling, (block) => { state.sampling = block; });
const rowOf = (state) => S.parseSampling(state.sampling);

// ---- the node ---------------------------------------------------------------
{
  const state = S.parsePreStage(JSON.stringify({ arch: "krea2", sampling: { cfg: 3.5, steps: 30 } }));
  const io = ioOver(state);
  const row = new PreStageRow({ state, widgetIO: () => io, commit: () => {} });
  row.setArch("ideogram4");
  out.onIdeogram = { arch: state.arch, row: rowOf(state), spare: state.sampling_spare };
  io.set("cfg", 7);
  row.setArch("krea2");
  out.back = { arch: state.arch, row: rowOf(state), spare: state.sampling_spare };
  const blob = JSON.parse(S.serializePreStage(state));
  out.saved = { spare: blob.sampling_spare, again: S.parsePreStage(S.serializePreStage(state)).sampling_spare };
  out.freshBlob = "sampling_spare" in JSON.parse(S.serializePreStage(S.emptyPreStage()));
  // A thrown switch is released into the stash: what comes back is the
  // dialled row, not the distillation's.
  state.turbo.krea2 = { on: true, quality: "good", lora: null, saved: { steps: 30, cfg: 3.5 } };
  io.set("steps", 8); io.set("cfg", 1);
  row.setArch("ideogram4");
  out.released = { spare: state.sampling_spare.krea2, turbo: state.turbo.krea2.on };
}

// ---- the chat ---------------------------------------------------------------
{
  let rail = { still_arch: "ideogram4", video_family: "h3" };
  const sync = new Sync({ rail: () => rail, setRail: (patch) => { rail = { ...rail, ...patch }; } });
  sync.takeOwn("still");
  sync.copy("still").sampling = { cfg: 7, steps: 28 };
  sync.save("still");
  sync.moveStill("krea2");
  const onKrea = sync.copy("still");
  out.chatStill = { rail: rail.still_arch, copyArch: onKrea.arch, row: sync.row("still").sampling,
                    spare: onKrea.sampling_spare };
  sync.moveStill("ideogram4");
  out.chatStillBack = sync.row("still").sampling;

  sync.takeOwn("video");
  sync.copy("video").sampling = { steps: 20, sampler_name: "res_multistep" };
  sync.save("video");
  sync.moveVideo("ltx25");
  out.chatVideo = { rail: rail.video_family, family: S.pieceFamily(sync.copy("video")),
                    row: sync.row("video").sampling, spare: sync.copy("video").sampling_spare };
  sync.moveVideo("h3");
  out.chatVideoBack = sync.row("video").sampling;

  // An unpinned side moves the rail and nothing else.
  sync.follow("still");
  sync.moveStill("krea2");
  out.unpinned = { rail: rail.still_arch, own: sync.own("still") };
}
console.log(JSON.stringify(out));
'''

with layout.pack(skip=["atlas"]) as target:
    r = layout.in_pack(DOM + SCRIPT, target)

ideo = r["onIdeogram"]
check("switching the node to Ideogram writes Ideogram's own row", ideo["arch"], "ideogram4")
check("...not Krea's guidance", ideo["row"]["cfg"] != 3.5 and ideo["row"]["steps"] != 30, True)
check("...and sets Krea's dialled row aside", ideo["spare"], {"krea2": {"cfg": 3.5, "steps": 30}})
back = r["back"]
check("coming back hands Krea its row", (back["row"]["cfg"], back["row"]["steps"]), (3.5, 30))
check("...with Ideogram's now in the stash", back["spare"]["ideogram4"]["cfg"], 7)
check("...and Krea's out of it", "krea2" in back["spare"], False)
check("the stash survives a save", r["saved"]["spare"], r["saved"]["again"])
check("a node that never switched writes none", r["freshBlob"], False)
check("a thrown turbo switch is released before the row is set aside",
      (r["released"]["spare"]["steps"], r["released"]["spare"]["cfg"], r["released"]["turbo"]),
      (30, 3.5, False))

cs = r["chatStill"]
check("the chat's image pill moves the pinned copy with the rail",
      (cs["rail"], cs["copyArch"]), ("krea2", "krea2"))
check("...so Krea does not sample on Ideogram's cfg", cs["row"].get("cfg") != 7, True)
check("...and Ideogram's row waits under its name", cs["spare"]["ideogram4"]["cfg"], 7)
check("...to come back with it", r["chatStillBack"]["cfg"], 7)
cv = r["chatVideo"]
check("the video pill moves the pinned clip copy onto the family",
      (cv["rail"], cv["family"]), ("ltx25", "ltx25"))
check("...H3's row set aside, not in force", (cv["row"], cv["spare"]),
      ({}, {"h3": {"steps": 20, "sampler_name": "res_multistep"}}))
check("...and back", r["chatVideoBack"], {"steps": 20, "sampler_name": "res_multistep"})
check("an unpinned side just moves the rail", r["unpinned"], {"rail": "krea2", "own": False})
