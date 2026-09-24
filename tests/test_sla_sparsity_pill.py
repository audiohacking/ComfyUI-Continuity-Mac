"""Attention pill on Continuity Mac: `default` only; SLA sparsity is retired.

    python3 tests/test_sla_sparsity_pill.py

SLA's sparsity stepper used to extend the attention pill under `attention=sla`.
Continuity Mac removed sage / kitchen / sla / laya from the combo (patched
rides under default), so the pill no longer grows a sparsity control — a stale
blob naming sla is coerced to default and draws no stepper.

Skips itself if node is not installed.
"""

import layout
from domshim import DOM
from harness import FAILURES, check, passed

layout.skip_without_node()
passed("Continuity Mac attention pill is default-only; SLA sparsity stays off the row")

_pkg = layout.load("canvas", "accel", "sampling", "contextir", "compile",
                   "compile_image", "models", "registry", "manifest",
                   "still", "krea2_still", "ideogram4_still", "grammar",
                   "h3_declare", "h3_models", "h3_grammar", "ltx25_declare",
                   "ltx25_models", "ltx25_sampling")
h3 = _pkg.manifest.describe("h3")
ATTENTION = next(w for w in h3["widgets"] if w["id"] == "attention")
SPARSITY = next(w for w in h3["widgets"] if w["id"] == "sla_sparsity")

CHECK = """
await import("./dom.mjs");
const { samplingBar } = await import("./web/creator/sampling.js");

const [, casesJSON] = process.argv;
const cases = JSON.parse(casesJSON);

const textOf = (node) => {
  const out = [];
  const walk = (n) => {
    if (!n) return;
    if (n.tagName === "SPAN" && !(n.children ?? []).length && n.textContent)
      out.push(String(n.textContent).trim());
    (n.children ?? []).forEach(walk);
  };
  walk(node);
  return out;
};

const widgets = {
  attention: { name: "attention", value: "default",
               options: { values: ["default"] } },
};

const out = {};
for (const [label, blob] of Object.entries(cases)) {
  out[label] = textOf(samplingBar({
    widgets,
    value: (name, fallback) => (name in blob ? blob[name] : fallback),
    set: () => {},
    family: "h3",
  }));
}
console.log(JSON.stringify(out));
"""

CASES = {
    "default": {},
    "retired_sla": {"attention": "sla", "sla_sparsity": 0.7},
}

with layout.pack(skip=["atlas"]) as target:
    drawn = layout.in_pack(
        CHECK.replace("await import(\"./dom.mjs\");", DOM), target, CASES)


def says(case, needle):
    return any(text == needle for text in drawn[case])


check("the attention widget offers only default",
      ATTENTION["options"], ["default"])
check("sla_sparsity remains on the manifest (blob compat) but gated on sla",
      (SPARSITY["group"], SPARSITY.get("requires")),
      ("accel", [{"id": "attention", "value": "sla"}]))
check("default draws attention default",
      says("default", "attention default"), True)
check("no sparsity stepper on the default row",
      any(t.startswith("sparsity") for t in drawn["default"]), False)
check("a retired sla blob still draws no sparsity stepper",
      any(t.startswith("sparsity") for t in drawn["retired_sla"]), False)

if FAILURES:
    raise SystemExit(1)
