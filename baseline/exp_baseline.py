"""Baseline grounding accuracy on the FULL benchmarks (all samples), per model.
This is the standard multi-benchmark leaderboard number; the A/B/C experiments
then dissect *why* accuracy is where it is. Writes out/<model>/baseline.json.
"""
from __future__ import annotations

import os
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import json
import asyncio
import argparse

from PIL import Image

from grounding import load_items, hit, DATASETS
from client import GroundingPool, wait_ready
from models import get_spec

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))


async def run(spec, dataset, limit, ports, OUT, chunk=192):
    items = load_items(dataset, limit=limit)
    pool = GroundingPool(spec, ports)
    n = h = pf = 0
    per_type = {}
    # stream in chunks so we never hold the whole (decoded) dataset in RAM
    for c0 in range(0, len(items), chunk):
        batch = items[c0:c0 + chunk]
        jobs = [(it.img_path, it.instruction) for it in batch]  # opened in executor
        res = await pool.ground_many(jobs)
        for it, (pt, raw) in zip(batch, res):
            n += 1
            ok = 0
            if pt is None:
                pf += 1
            else:
                px = (pt[0] * it.img_w, pt[1] * it.img_h)
                ok = int(hit(it.bbox, px))
                h += ok
            d = per_type.setdefault(it.ui_type, [0, 0])
            d[0] += ok
            d[1] += 1
    out = {"dataset": dataset, "n": n, "hit": h, "parse_fail": pf,
           "acc": h / max(n, 1),
           "by_ui_type": {k: {"acc": v[0] / max(v[1], 1), "n": v[1]}
                          for k, v in per_type.items()}}
    fp = os.path.join(OUT, f"baseline_{dataset}.json")
    json.dump(out, open(fp, "w"), indent=1)
    print(f"  [{spec.key}] {dataset:16s} acc={out['acc']:.3f} (n={n}, pf={pf})",
          flush=True)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl_8b")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    ap.add_argument("--limit", type=int, default=-1)  # -1 = full benchmark
    ap.add_argument("--ports", type=int, nargs="*", default=None)
    args = ap.parse_args()
    spec = get_spec(args.model)
    OUT = os.path.join(HERE, "..", "out", args.model)
    os.makedirs(OUT, exist_ok=True)
    await wait_ready(args.ports)
    print(f"=== [{args.model}] Baseline (full-benchmark) ===")
    for ds in args.datasets:
        await run(spec, ds, args.limit, args.ports, OUT)


if __name__ == "__main__":
    asyncio.run(main())
