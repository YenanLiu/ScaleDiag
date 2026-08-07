"""Experiment D — Linguistic information scaling.

Experiment B (attribute enrichment) came back near-null: adding auto-derived
color/size/shape to an already-complete instruction barely moves accuracy (and
sometimes hurts). That tells us enrichment is the wrong lever. The meaningful
question is the *information axis itself*:

  How does grounding accuracy scale with the amount of linguistic information,
  from none (generic) up to an oracle localisation hint?

We sweep a monotone information ladder on the SAME screenshot + target:

  generic      : target identity removed entirely  -> information FLOOR
  keep_half    : first ~50% of the instruction tokens (partial identity)
  base         : the original instruction
  clarify_pos  : base + coarse oracle region (3x3)        -> mild clarification
  clarify_fine : base + fine oracle location (~% x,y)      -> strong clarification

Read alongside Experiment A (visual/crop) this separates regimes:
  - if generic collapses and base recovers -> the model *uses* identity info;
  - if clarify_fine >> base -> failures are INFORMATION-limited (compute can't
    fix them, a user hint can);
  - if clarify_fine ~= base while crop lifts accuracy -> failures are
    COMPUTE/visual-limited, not linguistic.

Pure-text intervention (same request path as the base grounding call), so it is
model-agnostic and reproducible. Writes out/<model>/exp_language_<dataset>.json.
"""
from __future__ import annotations

import os
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import json
import math
import asyncio
import argparse

from PIL import Image

from grounding import load_items, hit, DATASETS
from client import GroundingPool, wait_ready
from models import get_spec

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = ["generic", "keep_half", "base", "clarify_pos", "clarify_fine"]

GENERIC = "Click the correct interactive UI element on the screen for the task."


def _region_3x3(cx_frac: float, cy_frac: float) -> str:
    col = "left" if cx_frac < 1 / 3 else ("right" if cx_frac > 2 / 3 else "center")
    row = "top" if cy_frac < 1 / 3 else ("bottom" if cy_frac > 2 / 3 else "middle")
    if row == "middle" and col == "center":
        return "center"
    return f"{row} {col}".replace("middle ", "").replace(" center", "")


def keep_half(instr: str) -> str:
    toks = instr.split()
    if len(toks) <= 2:
        return instr
    return " ".join(toks[: max(1, math.ceil(len(toks) / 2))])


def build(cond: str, instr: str, cx_frac: float, cy_frac: float) -> str:
    if cond == "generic":
        return GENERIC
    if cond == "keep_half":
        return keep_half(instr)
    if cond == "base":
        return instr
    if cond == "clarify_pos":
        return instr.rstrip() + f" (It is in the {_region_3x3(cx_frac, cy_frac)} region of the screen.)"
    if cond == "clarify_fine":
        return (instr.rstrip() +
                f" (It is located approximately {round(cx_frac * 100)}% from the left "
                f"and {round(cy_frac * 100)}% from the top of the screen.)")
    raise ValueError(cond)


async def run(spec, dataset: str, limit: int, ports, OUT, chunk=150):
    """Grouped-by-image: all five information-ladder conditions of one screenshot
    are routed to the same server so the image prefill is computed once and reused
    via vLLM prefix caching."""
    items = load_items(dataset, limit=limit)
    fp = os.path.join(OUT, f"exp_language_{dataset}.json")
    if os.environ.get("FORCE", "0") != "1" and os.path.exists(fp):
        try:
            prev = json.load(open(fp))
            if prev.get("n") == len(items):
                print(f"[skip] full result exists {fp} (n={prev['n']})", flush=True)
                return prev
        except Exception:
            pass
    pool = GroundingPool(spec, ports)
    acc = {c: {"n": 0, "hit": 0, "parse_fail": 0} for c in CONDITIONS}

    for c0 in range(0, len(items), chunk):
        chunk_items = items[c0:c0 + chunk]
        groups, metas = [], []
        for it in chunk_items:
            cx, cy = it.gt_center
            cxf, cyf = cx / max(it.img_w, 1), cy / max(it.img_h, 1)
            instrs = [build(c, it.instruction, cxf, cyf) for c in CONDITIONS]
            groups.append((it.img_path, instrs))
            metas.append(it)
        res = await pool.ground_grouped(groups)
        for it, per in zip(metas, res):
            for ci, cond in enumerate(CONDITIONS):
                pt, _raw = per[ci]
                a = acc[cond]
                a["n"] += 1
                if pt is None:
                    a["parse_fail"] += 1
                    continue
                a["hit"] += int(hit(it.bbox, (pt[0] * it.img_w, pt[1] * it.img_h)))
        b = acc["base"]
        print(f"  {dataset:15s} {c0 + len(chunk_items)}/{len(items)}  "
              f"base={b['hit'] / max(1, b['n']):.3f}", flush=True)

    results = {c: {"n": acc[c]["n"], "hit": acc[c]["hit"],
                   "parse_fail": acc[c]["parse_fail"],
                   "acc": acc[c]["hit"] / max(1, acc[c]["n"])} for c in CONDITIONS}
    base = results["base"]["acc"]
    out = {
        "dataset": dataset, "n": len(items), "conditions": results,
        "gains": {
            "reliance_on_identity": base - results["generic"]["acc"],   # base - floor
            "clarify_pos_gain": results["clarify_pos"]["acc"] - base,
            "clarify_fine_gain": results["clarify_fine"]["acc"] - base,
        },
    }
    json.dump(out, open(fp, "w"), indent=1)
    print(f"[written] {fp}", flush=True)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl_8b")
    ap.add_argument("--datasets", nargs="*",
                    default=["screenspot_pro", "ui_vision", "mmbench_gui", "osworld_g"])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--ports", type=int, nargs="*", default=None)
    ap.add_argument("--chunk", type=int, default=150)
    args = ap.parse_args()
    spec = get_spec(args.model)
    OUT = os.path.join(HERE, "..", "out", args.model)
    os.makedirs(OUT, exist_ok=True)
    await wait_ready(args.ports)
    for ds in args.datasets:
        print(f"=== [{args.model}] Experiment D (linguistic info scaling): {ds} ===",
              flush=True)
        await run(spec, ds, args.limit, args.ports, OUT, chunk=args.chunk)


if __name__ == "__main__":
    asyncio.run(main())
