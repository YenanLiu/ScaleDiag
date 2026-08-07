"""Experiment B — What kind of instruction detail actually helps grounding?

We keep the image fixed (full resolution) and rewrite the instruction, adding
fine-grained descriptive attributes that we derive *from the ground truth*
(so no external annotator is needed and the added information is always true):

  position : which 3x3 screen region the target sits in     ("top-right", ...)
  color    : dominant colour of the target patch            ("blue", "gray", ...)
  size     : target footprint relative to the screen        ("small"/"medium"/"large")
  shape    : bounding-box aspect                            ("square"/"wide"/"tall")

Conditions: the original instruction (simple baseline), each single attribute,
and several multi-attribute combinations. Comparing accuracy across conditions
isolates which descriptive cues substantially move GUI grounding.

Writes out/exp_instruction_<dataset>.json.
"""
from __future__ import annotations

import os
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import json
import asyncio
import argparse
from typing import List

import numpy as np
from PIL import Image

from grounding import load_items, hit
from client import GroundingPool, wait_ready
from models import get_spec

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))

# named colours for nearest-match (RGB)
COLORS = {
    "black": (20, 20, 20), "white": (240, 240, 240), "gray": (128, 128, 128),
    "red": (200, 40, 40), "green": (40, 160, 60), "blue": (40, 90, 200),
    "yellow": (230, 210, 40), "orange": (230, 140, 30), "purple": (140, 60, 180),
    "cyan": (40, 190, 200), "pink": (230, 130, 180), "brown": (140, 90, 50),
}

CONDITIONS = ["base", "+pos", "+color", "+size", "+shape",
              "+pos+color", "+pos+size", "+pos+color+size", "+all"]


def pos_label(cx_frac, cy_frac) -> str:
    col = "left" if cx_frac < 1 / 3 else ("right" if cx_frac > 2 / 3 else "center")
    row = "top" if cy_frac < 1 / 3 else ("bottom" if cy_frac > 2 / 3 else "middle")
    if row == "middle" and col == "center":
        return "center"
    return f"{row}-{col}".replace("middle-", "").replace("-center", "")


def color_label(img: Image.Image, bbox) -> str:
    vals = [int(v) for v in bbox]
    if len(vals) == 4:
        x0, y0, x1, y1 = vals
    else:
        xs, ys = vals[0::2], vals[1::2]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = max(x0 + 1, x1), max(y0 + 1, y1)
    patch = img.crop((x0, y0, x1, y1)).resize((16, 16))
    arr = np.asarray(patch).reshape(-1, 3).astype(float)
    med = np.median(arr, axis=0)
    best = min(COLORS, key=lambda c: float(np.sum((np.array(COLORS[c]) - med) ** 2)))
    return best


def size_label(area_frac) -> str:
    if area_frac < 5e-4:
        return "small"
    if area_frac < 5e-3:
        return "medium"
    return "large"


def shape_label(bbox) -> str:
    w = max(1.0, bbox[2] - bbox[0])
    h = max(1.0, bbox[3] - bbox[1])
    r = w / h
    if r > 1.6:
        return "wide"
    if r < 0.62:
        return "tall"
    return "square"


def compose(instr: str, attrs: dict, cond: str) -> str:
    parts = []
    if "pos" in cond:
        parts.append(f"located in the {attrs['pos']} of the screen")
    desc = []
    if "size" in cond:
        desc.append(attrs["size"])
    if "color" in cond:
        desc.append(attrs["color"])
    if "shape" in cond:
        desc.append(attrs["shape"])
    clause = ""
    if desc or parts:
        d = ("a " + " ".join(desc) + " element") if desc else "the target"
        loc = (" " + " ".join(parts)) if parts else ""
        clause = f" (It is {d}{loc}.)"
    return instr.rstrip() + clause


async def run(spec, dataset: str, limit: int, ports, OUT, conditions=None,
              chunk=150):
    """Attribute-injection conditions. Requests are grouped by image so all
    conditions of one screenshot share a single image prefill (vLLM prefix cache)."""
    conditions = conditions if conditions is not None else CONDITIONS
    items = load_items(dataset, limit=limit)
    fp = os.path.join(OUT, f"exp_instruction_{dataset}.json")
    # resume: skip only if a *full* result (same item count) already exists
    if os.environ.get("FORCE", "0") != "1" and os.path.exists(fp):
        try:
            prev = json.load(open(fp))
            if prev.get("n") == len(items):
                print(f"[skip] full result exists {fp} (n={prev['n']})", flush=True)
                return prev
        except Exception:
            pass
    pool = GroundingPool(spec, ports)
    acc = {c: {"n": 0, "hit": 0, "parse_fail": 0} for c in conditions}

    def instrs_for(it, img):
        cx, cy = it.gt_center
        attrs = {
            "pos": pos_label(cx / it.img_w, cy / it.img_h),
            "color": color_label(img, it.bbox),
            "size": size_label(it.target_area_frac),
            "shape": shape_label(it.bbox),
        }
        seq = []
        for cond in conditions:
            ck = "+pos+color+size+shape" if cond == "+all" else cond
            seq.append(it.instruction if cond == "base"
                       else compose(it.instruction, attrs, ck))
        return seq

    for c0 in range(0, len(items), chunk):
        chunk_items = items[c0:c0 + chunk]
        groups, metas = [], []
        for it in chunk_items:
            try:
                img = Image.open(it.img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (128, 128))
            groups.append((it.img_path, instrs_for(it, img)))
            img.close()
            metas.append(it)
        res = await pool.ground_grouped(groups)  # per group: list aligned to conditions
        for it, per in zip(metas, res):
            for ci, cond in enumerate(conditions):
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
                   "acc": acc[c]["hit"] / max(1, acc[c]["n"])} for c in conditions}
    out = {"dataset": dataset, "n": len(items), "conditions": results}
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
        print(f"=== [{args.model}] Experiment B (instruction attributes): {ds} ===",
              flush=True)
        await run(spec, ds, args.limit, args.ports, OUT, chunk=args.chunk)


if __name__ == "__main__":
    asyncio.run(main())
