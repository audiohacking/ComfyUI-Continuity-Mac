"""Reference size survives pool injection and merged/one-pass serialization.

Runs without ComfyUI or model weights: this is the real compiler's payload
round trip, not a render or a measurement of GPU memory.

Continuity Metal locks video to `match` (compile coerces `max`); images still
round-trip both sizes.
"""

import copy

import layout
from harness import check

compiler = layout.load("compile").compile


def video(size):
    return {"handle": "vid-1", "kind": "video", "role": "reference",
            "filename": "motion.mp4", "track": "picture",
            "takes": "motion", "ref_size": size}


for size in ("match", "max"):
    raw = {"handle": "ref-1", "kind": "image", "filename": "reference",
           "ref_size": size}
    parsed = compiler._parse_assets([raw])[0]
    restored = compiler._parse_assets([compiler._asset_dict(parsed)])[0]
    check(f"image/{size} round-trips", restored.ref_size, size)

# Video: both sizes compile to match and stay there after serialize.
for size in ("match", "max"):
    parsed = compiler._parse_assets([video(size)])[0]
    check(f"video/{size} coerces to match", parsed.ref_size, "match")
    restored = compiler._parse_assets([compiler._asset_dict(parsed)])[0]
    check(f"video/{size} stays match after serialize", restored.ref_size, "match")

for size in ("match", "max"):
    piece = {"version": 2, "family": "h3", "aspect": "1:1", "short_edge": 480,
             "assets": [video(size)],
             "segments": [{"duration_s": 3, "prompt": "@vid-1 walks."},
                          {"duration_s": 3, "prompt": "@vid-1 waves."}]}
    for location in ("pool", "local"):
        data = copy.deepcopy(piece)
        if location == "local":
            assets = data.pop("assets")
            for segment in data["segments"]:
                segment["assets"] = copy.deepcopy(assets)
        for mode in ("chain", "merge", "one-pass"):
            selected = copy.deepcopy(data)
            if mode == "merge":
                selected["segments"][1]["merge"] = True
            payloads = ([compiler.single_payload(selected)] if mode == "one-pass"
                        else compiler.timeline_payloads(selected))
            for index, payload in enumerate(payloads):
                compiled = compiler.compile_segment(payload)
                check(f"{location}/{mode}/{size} pass {index} keeps video match",
                      [asset.ref_size for asset in compiled.ref_videos], ["match"])

# Cast motion injects the video without a direct @video citation in the shot.
piece = {"version": 2, "family": "h3", "aspect": "1:1", "short_edge": 480,
         "assets": [{"handle": "img-1", "kind": "image", "filename": "face.png"},
                    video("match")],
         "subjects": [{"handle": "anna", "from": ["img-1"], "motion": "vid-1"}],
         "segments": [{"duration_s": 3, "prompt": "@anna walks."},
                      {"duration_s": 3, "prompt": "@anna waves."}]}
for payload in compiler.timeline_payloads(piece):
    check("cast-injected motion keeps match",
          compiler.compile_segment(payload).ref_videos[0].ref_size, "match")

# Video max is coerced to match on Continuity Metal — never round-trips as max.
check("video max is coerced away",
      "ref_size" not in compiler._asset_dict(
          compiler._parse_assets([video("max")])[0]), True)
check("video match stays the silent default",
      "ref_size" not in compiler._asset_dict(
          compiler._parse_assets([video("match")])[0]), True)
