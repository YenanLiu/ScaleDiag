"""Deterministic crop -> effective-target-size translation for Experiment A.

The crop geometry (grounding.aspect_crop) and the smart-resize applied before
the image is sent (client.prep_b64 -> grounding.smart_resize) are BOTH
deterministic. So for every sample and every crop fraction we can compute,
WITHOUT running any model, the size the target actually has in the pixels the
model sees:

  - eff_short_px  : target bbox short edge, in the smart-resized input (px)
  - eff_tokens    : number of vision tokens the target covers (area/factor^2)
  - occ_frac      : target area as a fraction of the shown (cropped) image

This converts the paper's "peak crop ratio = 6-25% area" into an absolute
target-size statement, and exposes that "crop ratio" conflates two regimes:
downscaled (cropping restores pixels) vs already-native (cropping only raises
relative occupancy, not pixel count).

Two modes:
  (default) annotation-only translation table  ->  crop frac -> target size
  --from-runs  read out/<model>/exp_crop_*.json per-sample logs and bin the
               ACTUAL accuracy by effective target size (occupancy / short-edge
               px), pooled across benchmarks -> the unified "how big is best"
               curve. Requires an exp_crop run produced after per-sample logging
               was added (result["samples"]).

Run:  python analyze_target_size.py --model qwen3vl_8b
      python analyze_target_size.py --from-runs --model qwen3vl_8b
Annotation-only mode needs no GPU / no serving.
"""
from __future__ import annotations

import os
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import glob
import json
import argparse
import statistics as st
from typing import List

from grounding import load_items, aspect_crop, smart_resize, DATASETS
from models import get_spec

HERE = os.path.dirname(os.path.abspath(__file__))

FRACS = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
DATASETS_ORDER = ["screenspot_pro", "ui_vision", "mmbench_gui",
                  "osworld_g", "screenspot_v2"]


def eff_sizes(it, frac, factor, min_px, max_px):
    """Return (eff_short_px, eff_tokens, occ_frac) for one item at one frac."""
    cx, cy = it.gt_center
    x0, y0, x1, y1 = aspect_crop(it.img_w, it.img_h, cx, cy, frac)
    cw, ch = max(1, x1 - x0), max(1, y1 - y0)
    rh, rw = smart_resize(ch, cw, factor, min_px, max_px)
    scale_x, scale_y = rw / cw, rh / ch
    bw = max(0.0, it.bbox[2] - it.bbox[0])
    bh = max(0.0, it.bbox[3] - it.bbox[1])
    tw, th = bw * scale_x, bh * scale_y          # target size in resized input
    eff_short = min(tw, th)
    eff_tokens = (tw / factor) * (th / factor)   # merged vision tokens
    occ_frac = (bw * bh) / float(cw * ch)        # occupancy of shown image
    return eff_short, eff_tokens, occ_frac


def med(xs: List[float]) -> float:
    return st.median(xs) if xs else float("nan")


def bin_by(records, key, edges):
    """Bucket per-sample records by record[key] into [edges] and report acc."""
    buckets = [[] for _ in range(len(edges) + 1)]
    for r in records:
        v = r[key]
        b = len(edges)
        for i, e in enumerate(edges):
            if v < e:
                b = i
                break
        buckets[b].append(r["hit"])
    labels = ([f"<{edges[0]:g}"]
              + [f"{edges[i-1]:g}-{edges[i]:g}" for i in range(1, len(edges))]
              + [f">={edges[-1]:g}"])
    print(f"{'bin':>14} | {'n':>6} | {'acc%':>6}")
    for lab, hits in zip(labels, buckets):
        acc = 100 * sum(hits) / max(1, len(hits))
        print(f"{lab:>14} | {len(hits):>6} | {acc:>6.1f}")


def bin_mean(records, key, edges, valkey):
    """Bucket records by record[key]; report n and mean(record[valkey]) per bin."""
    buckets = [[] for _ in range(len(edges) + 1)]
    for r in records:
        v = r[key]
        b = len(edges)
        for i, e in enumerate(edges):
            if v < e:
                b = i
                break
        buckets[b].append(r[valkey])
    labels = ([f"<{edges[0]:g}"]
              + [f"{edges[i-1]:g}-{edges[i]:g}" for i in range(1, len(edges))]
              + [f">={edges[-1]:g}"])
    print(f"{'bin':>14} | {'n':>7} | {'mean':>7}")
    rows = []
    for lab, vals in zip(labels, buckets):
        mv = sum(vals) / len(vals) if vals else float("nan")
        rows.append((lab, len(vals), mv))
        print(f"{lab:>14} | {len(vals):>7} | {mv:>7.3f}")
    return rows


OCC_EDGES = [0.05, 0.1, 0.25, 0.5, 1, 2, 5]          # occupancy bin edges (%)
NATIVE_EDGES = [0.001, 0.005, 0.02]                  # native occ frac strata
NATIVE_LABELS = ["<0.1%", "0.1-0.5%", "0.5-2%", ">=2%"]
FR_ORDER = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]


def compute_within(models, datasets):
    """Difficulty-controlled crop analysis (returns structured aggregates).

    Naive pooling of acc-vs-occupancy is confounded: naturally-large targets are
    both high-occupancy AND easy. Here each item is its OWN control: every crop
    fraction is paired to the SAME item's full-frame (frac=1.0) result and we
    measure delta_hit = hit(frac) - hit(full). This isolates the causal effect of
    cropping-to-higher-occupancy from the item's intrinsic difficulty.
    """
    from collections import defaultdict
    OUTD = os.path.join(HERE, "..", "out")
    groups = defaultdict(dict)   # (model,dataset,id) -> {frac: rec}
    used = []
    for m in models:
        for ds in datasets:
            fp = os.path.join(OUTD, m, f"exp_crop_{ds}.json")
            if not os.path.exists(fp):
                continue
            d = json.load(open(fp))
            if "samples" not in d:
                continue
            used.append((m, ds))
            for r in d["samples"]:
                groups[(m, ds, r["id"])][r["frac"]] = r

    def native_bin(nocc_frac):
        for i, e in enumerate(NATIVE_EDGES):
            if nocc_frac < e:
                return i
        return len(NATIVE_EDGES)

    deltas = []
    strat = defaultdict(lambda: defaultdict(list))
    n_items = n_helped = n_hurt = 0
    best_occ_when_helped = []
    for _key, byfrac in groups.items():
        base = byfrac.get(1.0)
        if base is None or base.get("parse_fail"):
            continue
        bh = base["hit"]
        nb = native_bin(base["occ_frac"])
        n_items += 1
        for frac, r in byfrac.items():
            if r.get("parse_fail"):
                continue
            strat[nb][frac].append(r["hit"])
            if frac != 1.0:
                deltas.append((r["occ_frac"] * 100, frac, r["hit"] - bh))
        cropped = [(f, byfrac[f]) for f in byfrac if f != 1.0
                   and not byfrac[f].get("parse_fail")]
        any_hit = [f for f, rr in cropped if rr["hit"] == 1]
        if bh == 0 and any_hit:
            n_helped += 1
            best_occ_when_helped.append(byfrac[max(any_hit)]["occ_frac"] * 100)
        elif bh == 1 and cropped and all(rr["hit"] == 0 for _, rr in cropped):
            n_hurt += 1

    # gain by achieved-occupancy bin
    gbuck = [[] for _ in range(len(OCC_EDGES) + 1)]
    for occ, _frac, dl in deltas:
        b = len(OCC_EDGES)
        for i, e in enumerate(OCC_EDGES):
            if occ < e:
                b = i
                break
        gbuck[b].append(dl)
    gain_by_occ = [(len(v), (sum(v) / len(v) if v else float("nan")))
                   for v in gbuck]
    # gain by crop fraction
    fbuck = defaultdict(list)
    for _occ, frac, dl in deltas:
        fbuck[frac].append(dl)
    gain_by_frac = {f: (len(fbuck[f]), sum(fbuck[f]) / len(fbuck[f]))
                    for f in sorted(fbuck)}
    # stratified raw accuracy by frac
    strat_acc = {}
    for nb in range(len(NATIVE_EDGES) + 1):
        if nb not in strat:
            continue
        row = {"n_items": len(strat[nb].get(1.0, []))}
        for f in FR_ORDER:
            hits = strat[nb].get(f, [])
            row[f] = (100 * sum(hits) / len(hits)) if hits else None
        strat_acc[nb] = row
    bo = sorted(best_occ_when_helped)

    def q(p):
        return bo[min(len(bo) - 1, int(p * len(bo)))] if bo else float("nan")

    return {
        "used": used, "n_items": n_items,
        "gain_by_occ": gain_by_occ, "gain_by_frac": gain_by_frac,
        "strat_acc": strat_acc, "n_helped": n_helped, "n_hurt": n_hurt,
        "rescue_median": q(0.5), "rescue_iqr": (q(0.25), q(0.75)),
        "n_rescue": len(bo),
    }


def within_item(models, datasets):
    """Print the difficulty-controlled analysis (thin wrapper over compute_within)."""
    a = compute_within(models, datasets)
    labels = ([f"<{OCC_EDGES[0]:g}"]
              + [f"{OCC_EDGES[i-1]:g}-{OCC_EDGES[i]:g}"
                 for i in range(1, len(OCC_EDGES))]
              + [f">={OCC_EDGES[-1]:g}"])
    print(f"pooled models x datasets = {a['used']}")
    print(f"paired items (with valid full-frame baseline) = {a['n_items']}\n")
    print("=== (A) within-item crop GAIN over full frame, by ACHIEVED occupancy (%) ===")
    print("    delta_hit = hit(cropped) - hit(full-frame), same item")
    print(f"{'occ% bin':>14} | {'n':>7} | {'mean':>7}")
    for lab, (n, mv) in zip(labels, a["gain_by_occ"]):
        print(f"{lab:>14} | {n:>7} | {mv:>7.3f}")
    print("\n=== (B) within-item crop GAIN by crop fraction ===")
    print(f"{'frac':>8} | {'n':>7} | {'mean delta':>10}")
    for f in sorted(a["gain_by_frac"], reverse=True):
        n, mv = a["gain_by_frac"][f]
        print(f"{f:>8} | {n:>7} | {mv:>10.3f}")
    print("\n=== (C) raw accuracy vs crop fraction, STRATIFIED by native occupancy ===")
    print(f"{'native occ':>12} | {'n_items':>7} | " +
          " | ".join(f"f={f:g}" for f in FR_ORDER))
    for nb, row in a["strat_acc"].items():
        cells = [f"{row[f]:5.1f}" if row[f] is not None else "  -  "
                 for f in FR_ORDER]
        print(f"{NATIVE_LABELS[nb]:>12} | {row['n_items']:>7} | " +
              " | ".join(cells))
    print("\n=== (D) operating point, difficulty-controlled ===")
    print(f"items where cropping RESCUES a full-frame miss : {a['n_helped']} "
          f"({100*a['n_helped']/max(1,a['n_items']):.1f}%)")
    print(f"items where cropping BREAKS a full-frame hit   : {a['n_hurt']} "
          f"({100*a['n_hurt']/max(1,a['n_items']):.1f}%)")
    lo, hi = a["rescue_iqr"]
    print(f"'just-enough' occupancy when cropping helps (median) = "
          f"{a['rescue_median']:.2f}%  IQR=[{lo:.2f}, {hi:.2f}]%  (n={a['n_rescue']})")


def from_runs(model, datasets):
    """Re-bin actual accuracy by effective target size, pooled across sets."""
    recs = []
    for ds in datasets:
        fp = os.path.join(HERE, "..", "out", model, f"exp_crop_{ds}.json")
        if not os.path.exists(fp):
            print(f"[skip] {fp} missing")
            continue
        d = json.load(open(fp))
        if "samples" not in d:
            print(f"[skip] {fp} has no per-sample logs (re-run exp_crop.py)")
            continue
        recs.extend(d["samples"])
    if not recs:
        print("No per-sample records found. Re-run exp_crop.py first.")
        return
    valid = [r for r in recs if not r["parse_fail"]]
    print(f"model={model}  pooled samples={len(recs)} "
          f"(parse-ok={len(valid)}) across {len(datasets)} datasets\n")
    print("=== accuracy vs target OCCUPANCY (target area / shown image, %) ===")
    bin_by(valid, "occ_frac", [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05])
    print("\n=== accuracy vs effective target SHORT EDGE (px in resized input) ===")
    bin_by(valid, "eff_short_px", [16, 24, 32, 48, 64, 96, 128])
    print("\n=== accuracy vs effective target TOKENS (merged vision tokens) ===")
    bin_by(valid, "eff_tokens", [1, 2, 4, 8, 16, 32, 64])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl_8b")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--datasets", nargs="*", default=DATASETS_ORDER)
    ap.add_argument("--from-runs", action="store_true",
                    help="bin actual accuracy from out/<model>/exp_crop_*.json")
    ap.add_argument("--within", action="store_true",
                    help="difficulty-controlled within-item crop analysis")
    ap.add_argument("--models", nargs="*",
                    default=["qwen3vl_8b", "maiui_8b", "qwen25vl_7b", "gta1_7b"],
                    help="models to pool for --within")
    args = ap.parse_args()

    if args.within:
        within_item(args.models, args.datasets)
        return
    if args.from_runs:
        from_runs(args.model, args.datasets)
        return

    spec = get_spec(args.model)
    factor, min_px, max_px = spec.factor, spec.min_pixels, spec.max_pixels
    print(f"model={args.model}  factor={factor}  min_px={min_px}  "
          f"max_px={max_px}  (token = {factor}x{factor}px)\n")

    for ds in args.datasets:
        try:
            items = load_items(ds, limit=args.limit)
        except Exception as e:
            print(f"[skip {ds}] {e}")
            continue
        print(f"### {ds}  (n={len(items)}, res={DATASETS[ds]['res']})")
        print(f"{'frac':>8} | {'occ%(med)':>9} | {'short_px(med)':>13} | "
              f"{'tokens(med)':>11} | {'%downscaled':>11}")
        for f in FRACS:
            occ, shorts, toks, ndown = [], [], [], 0
            for it in items:
                if it.img_w <= 0 or it.img_h <= 0:
                    continue
                # is the crop downscaled by smart-resize? (crop area > max_px)
                s = f ** 0.5
                cw, ch = it.img_w * s, it.img_h * s
                if cw * ch > max_px:
                    ndown += 1
                es, et, oc = eff_sizes(it, f, factor, min_px, max_px)
                shorts.append(es); toks.append(et); occ.append(oc * 100)
            n = max(1, len(shorts))
            print(f"{f:>8} | {med(occ):>8.2f} | {med(shorts):>13.0f} | "
                  f"{med(toks):>11.0f} | {100*ndown/n:>10.0f}%")
        print()


if __name__ == "__main__":
    main()
