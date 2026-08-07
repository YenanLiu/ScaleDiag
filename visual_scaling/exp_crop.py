"""Experiment A — Visual scaling by crop size.

For each dataset we crop the screenshot to a sequence of sizes, keeping the
image's native aspect ratio, centred on the target, and measure grounding
accuracy at each size. This isolates the *resolution* lever: as the crop
shrinks, the target occupies more pixels (higher effective resolution) but less
surrounding context is visible. The resulting accuracy-vs-crop-size curve shows
where visual scaling helps and where it saturates or reverses, per benchmark
resolution.

frac = crop area / full-image area. frac=1.0 is the full-image baseline.
Crops are GT-centred (oracle localisation) so the curve is the resolution
ceiling attributable purely to crop size, not to where we crop.

Writes out/exp_crop_<dataset>.json.
"""
from __future__ import annotations

import os
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import json
import asyncio
import argparse

from PIL import Image

from grounding import load_items, aspect_crop, remap_point_from_crop, hit, DATASETS
from client import GroundingPool, wait_ready
from models import get_spec
from analyze_target_size import eff_sizes

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))

FRACS = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]


async def run(spec, dataset: str, limit: int, ports, OUT):
    items = load_items(dataset, limit=limit)
    fp = os.path.join(OUT, f"exp_crop_{dataset}.json")
    if os.environ.get("FORCE", "0") != "1" and os.path.exists(fp):
        try:
            prev = json.load(open(fp))
            if prev.get("n") == len(items):
                print(f"[skip] full result exists {fp} (n={prev['n']})", flush=True)
                return prev
        except Exception:
            pass
    pool = GroundingPool(spec, ports)
    # cache opened images
    imgs = {}

    def get_img(p):
        if p not in imgs:
            try:
                imgs[p] = Image.open(p).convert("RGB")
            except Exception:
                imgs[p] = Image.new("RGB", (128, 128))
        return imgs[p]

    per_frac = {f: {"n": 0, "hit": 0, "parse_fail": 0} for f in FRACS}
    # per-sample records so accuracy can be re-binned by the *effective target
    # size the model actually sees* (occupancy / short-edge px / vision tokens),
    # not just by crop ratio. These are deterministic from geometry+smart-resize
    # (see analyze_target_size.eff_sizes) but we pair them with the actual hit.
    samples = []
    # build all jobs for one frac at a time to bound memory
    for f in FRACS:
        jobs = []
        meta = []
        for it in items:
            img = get_img(it.img_path)
            cx, cy = it.gt_center
            box = aspect_crop(it.img_w, it.img_h, cx, cy, f)
            crop = img.crop(box)
            jobs.append((crop, it.instruction))
            meta.append((it, box))
        res = await pool.ground_many(jobs)
        for (it, box), (pt, raw) in zip(meta, res):
            per_frac[f]["n"] += 1
            es, et, oc = eff_sizes(it, f, spec.factor, spec.min_pixels,
                                   spec.max_pixels)
            rec = {"id": str(it.id), "dataset": dataset, "frac": f,
                   "ui_type": it.ui_type,
                   "occ_frac": oc, "eff_short_px": es, "eff_tokens": et,
                   "parse_fail": int(pt is None), "hit": 0}
            if pt is None:
                per_frac[f]["parse_fail"] += 1
            else:
                px = remap_point_from_crop(pt, box, it.img_w, it.img_h)
                rec["hit"] = int(hit(it.bbox, px))
                per_frac[f]["hit"] += rec["hit"]
            samples.append(rec)
        acc = per_frac[f]["hit"] / max(per_frac[f]["n"], 1)
        print(f"  {dataset} frac={f:<7} acc={acc:.3f} "
              f"(n={per_frac[f]['n']}, pf={per_frac[f]['parse_fail']})", flush=True)

    result = {
        "dataset": dataset, "n": len(items), "fracs": FRACS,
        "resolution_class": DATASETS[dataset]["res"],
        "curve": {str(f): {"acc": per_frac[f]["hit"] / max(per_frac[f]["n"], 1),
                           **per_frac[f]} for f in FRACS},
        "samples": samples,
    }
    json.dump(result, open(fp, "w"), indent=1)
    print(f"[written] {fp}")
    return result


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl_8b")
    ap.add_argument("--datasets", nargs="*",
                    default=["screenspot_pro", "ui_vision", "mmbench_gui",
                             "osworld_g", "screenspot_v2"])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--ports", type=int, nargs="*", default=None)
    args = ap.parse_args()
    spec = get_spec(args.model)
    OUT = os.path.join(HERE, "..", "out", args.model)
    os.makedirs(OUT, exist_ok=True)
    await wait_ready(args.ports)
    for ds in args.datasets:
        print(f"=== [{args.model}] Experiment A (crop sweep): {ds} ===")
        await run(spec, ds, args.limit, args.ports, OUT)


if __name__ == "__main__":
    asyncio.run(main())
