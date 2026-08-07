"""Experiment U — Uncertainty: can the model's own dispersion tell us WHICH errors
are information-limited (a clarified instruction fixes them) vs. irreducible?

This is the uncertainty counterpart to the compute-vs-information thesis. For each
item we draw K samples at temperature T under several *instruction conditions*
(the GPT-5.2 rewrites), and study three things:

1) Sampling uncertainty as an error detector (selective prediction).
   Signal = sampling dispersion (mean pairwise distance of the K predicted points).
   - AUROC(dispersion -> greedy error): is uncertainty predictive of being wrong?
   - Risk-coverage: if we abstain on the most-dispersed items, how much does
     accuracy on the retained set rise? (the practical payoff of the signal)

2) Sampling headroom decomposition (ties to Exp E / GRPO).
   - oracle@K - greedy  = total best-of-N headroom.
   - sc_medoid - greedy = the verifier-free (self-consistency) part.
   High dispersion + oracle@K HITS => compute/sampling-limited (BoN/GRPO can fix).
   High dispersion + oracle@K MISSES => information-limited / irreducible.

3) THE key analysis — decompose uncertainty by what a clarified instruction does.
   Comparing a base condition to an information-rich rewrite (e.g. `fun`):
   - info_limited_rate = P(correct under fun | wrong under base)  -> errors that
     ADDING INFORMATION fixes.
   - residual_rate     = P(wrong under fun | wrong under base)    -> errors info
     does NOT fix (visual/compute-limited or irreducible).
   - Does uncertainty COLLAPSE when we add information? We compare per-item
     dispersion(base) vs dispersion(fun), and specifically whether the items whose
     dispersion collapses are the ones information fixes. If so, "the resolvable
     part of a model's uncertainty is exactly its information-limited part."

Writes out/<model>/exp_uncertainty_<dataset>.json (aggregates + compact per-item
arrays so figures/decomposition can be recomputed offline).
"""
from __future__ import annotations

import os
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "instruction"))
import json
import asyncio
import argparse

import numpy as np
from PIL import Image

from grounding import load_items, hit
from client import GroundingPool, wait_ready
from models import get_spec
from exp_sample import _medoid, _dispersion

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))

# instruction conditions to study: `ori` is the plain instruction; the rest are
# GPT-5.2 rewrites (fun = functional/info-rich, app = appearance, spa = spatial).
DEFAULT_CONDS = ["base", "fun", "app"]
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def _auroc(scores, labels):
    """AUROC for `scores` predicting positive `labels` (1). Rank-sum (Mann-Whitney).
    Higher score => more likely positive. NaN if only one class present."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {}
    start = 0
    for i, c in enumerate(counts):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[v] for v in inv])
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg))


def _risk_coverage(correct, disp):
    """Sort by ascending dispersion (most confident first); accuracy on the most
    confident `c` fraction, for c in COVERAGES."""
    c = np.asarray(correct, int)
    d = np.asarray(disp, float)
    order = np.argsort(d, kind="mergesort")  # low dispersion (confident) first
    c_sorted = c[order]
    out = {}
    n = len(c)
    for cov in COVERAGES:
        k = max(1, int(round(cov * n)))
        out[f"{cov:.1f}"] = float(c_sorted[:k].mean())
    return out


async def _measure(pool, items, jobs, k, temperature):
    """Return per-item dict arrays: greedy_correct, oracle, sc_correct, pass1, disp."""
    greedy = await pool.ground_many(jobs)
    sampled = await pool.ground_repeated(jobs, k=k, temperature=temperature)
    g_ok, oracle, sc_ok, pass1, disp, pf = [], [], [], [], [], 0
    for it, (gp, _), samples in zip(items, greedy, sampled):
        if gp is None:
            g_ok.append(0); pf += 1
        else:
            g_ok.append(int(hit(it.bbox, (gp[0] * it.img_w, gp[1] * it.img_h))))
        pts = [p for p, _ in samples]
        hits = [int(hit(it.bbox, (p[0] * it.img_w, p[1] * it.img_h)))
                if p is not None else 0 for p in pts]
        pass1.append(sum(hits) / len(hits) if hits else 0.0)
        oracle.append(int(any(hits)))
        mi = _medoid(pts)
        sc_ok.append(int(hit(it.bbox, (pts[mi][0] * it.img_w, pts[mi][1] * it.img_h)))
                     if (mi is not None and pts[mi] is not None) else 0)
        disp.append(_dispersion(pts))
    return {"greedy": g_ok, "oracle": oracle, "sc": sc_ok, "pass1": pass1,
            "disp": disp, "parse_fail": pf}


def _cond_summary(m):
    n = len(m["greedy"])
    ag = np.mean(m["greedy"]); ao = np.mean(m["oracle"]); asc = np.mean(m["sc"])
    return {
        "n": n,
        "acc_greedy": float(ag),
        "acc_pass1_mean": float(np.mean(m["pass1"])),
        "acc_self_consistency": float(asc),
        "acc_oracle_at_k": float(ao),
        "mean_dispersion": float(np.mean(m["disp"])),
        "sampling_headroom": float(ao - ag),          # oracle@K - greedy
        "free_sc_gain": float(asc - ag),              # self-consistency - greedy
        # uncertainty as an error detector
        "auroc_disp_error": _auroc(m["disp"], [1 - c for c in m["greedy"]]),
        "risk_coverage": _risk_coverage(m["greedy"], m["disp"]),
        # among high-dispersion (top-quartile) items, does oracle@K still hit?
        # (compute/sampling-limited) vs miss (information/irreducible)
        **_hi_disp_split(m),
        "parse_fail": m["parse_fail"],
    }


def _hi_disp_split(m):
    d = np.asarray(m["disp"], float)
    if len(d) < 4:
        return {"hi_disp_oracle_hit": float("nan"), "hi_disp_frac": 0.0}
    thr = np.quantile(d, 0.75)
    hi = d >= thr
    orc = np.asarray(m["oracle"], int)
    return {
        "hi_disp_frac": float(hi.mean()),
        # of the most-uncertain items, share BoN could still fix (compute-limited)
        "hi_disp_oracle_hit": float(orc[hi].mean()) if hi.any() else float("nan"),
    }


def _decompose(base, other):
    """base/other are per-item measurement dicts. Decompose base errors by whether
    the info-rich condition fixes them, and whether uncertainty collapses."""
    bg = np.asarray(base["greedy"], int)
    og = np.asarray(other["greedy"], int)
    bd = np.asarray(base["disp"], float)
    od = np.asarray(other["disp"], float)
    bwrong = bg == 0
    n_bwrong = int(bwrong.sum())
    fixed = int(((bg == 0) & (og == 1)).sum())      # info fixed it
    still = int(((bg == 0) & (og == 0)).sum())      # residual / irreducible
    d_fixed = (bd - od)[(bg == 0) & (og == 1)]      # uncertainty change on fixed
    d_still = (bd - od)[(bg == 0) & (og == 0)]      # uncertainty change on residual
    return {
        "n_base_wrong": n_bwrong,
        "info_limited_rate": (fixed / n_bwrong) if n_bwrong else float("nan"),
        "residual_rate": (still / n_bwrong) if n_bwrong else float("nan"),
        "acc_gain_greedy": float(og.mean() - bg.mean()),
        "mean_disp_drop_overall": float((bd - od).mean()),
        # the key comparison: does uncertainty collapse MORE on info-fixable errors?
        "disp_drop_on_fixed": float(d_fixed.mean()) if len(d_fixed) else float("nan"),
        "disp_drop_on_residual": float(d_still.mean()) if len(d_still) else float("nan"),
    }


async def run(spec, dataset, limit, ports, OUT, k, temperature, conds, rewrites,
              chunk=200):
    items = load_items(dataset, limit=limit)
    pool = GroundingPool(spec, ports)

    def get_img(p):
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            return Image.new("RGB", (128, 128))

    def instr_for(it, cond):
        if cond == "base":
            return it.instruction
        rec = rewrites.get((dataset, str(it.id))) if rewrites else None
        return (rec or {}).get(cond) or it.instruction

    # accumulate per-condition arrays across chunks (only tiny scalars are kept;
    # decoded images are released per chunk to bound memory on full datasets)
    keys = ["greedy", "oracle", "sc", "pass1", "disp"]
    per_cond_meas = {c: {kk: [] for kk in keys} | {"parse_fail": 0} for c in conds}
    for c0 in range(0, len(items), chunk):
        chunk_items = items[c0:c0 + chunk]
        imgs = [get_img(it.img_path) for it in chunk_items]
        for cond in conds:
            jobs = [(imgs[i], instr_for(it, cond)) for i, it in enumerate(chunk_items)]
            m = await _measure(pool, chunk_items, jobs, k, temperature)
            for kk in keys:
                per_cond_meas[cond][kk].extend(m[kk])
            per_cond_meas[cond]["parse_fail"] += m["parse_fail"]
        for im in imgs:
            try:
                im.close()
            except Exception:
                pass
        print(f"  {dataset:15s} chunk {c0 + len(chunk_items)}/{len(items)}", flush=True)

    summaries = {}
    for cond in conds:
        summaries[cond] = _cond_summary(per_cond_meas[cond])
        s = summaries[cond]
        print(f"  {dataset:15s} [{cond:4s}] greedy={s['acc_greedy']:.3f} "
              f"sc={s['acc_self_consistency']:.3f} oracle@{k}={s['acc_oracle_at_k']:.3f} "
              f"disp={s['mean_dispersion']:.3f} AUROC={s['auroc_disp_error']:.3f}",
              flush=True)

    # decomposition base vs each info condition
    decomp = {}
    if "base" in per_cond_meas:
        for cond in conds:
            if cond == "base":
                continue
            decomp[f"base_vs_{cond}"] = _decompose(per_cond_meas["base"],
                                                   per_cond_meas[cond])
            d = decomp[f"base_vs_{cond}"]
            print(f"    decomp base->{cond}: info_limited={d['info_limited_rate']:.3f} "
                  f"residual={d['residual_rate']:.3f} "
                  f"disp_drop fixed={d['disp_drop_on_fixed']:.3f} "
                  f"vs residual={d['disp_drop_on_residual']:.3f}", flush=True)

    out = {
        "dataset": dataset, "n": len(items), "k": k, "temperature": temperature,
        "conditions": summaries,
        "decomposition": decomp,
        # compact per-item arrays (for offline figures / re-analysis)
        "per_item": {c: {"greedy": per_cond_meas[c]["greedy"],
                         "oracle": per_cond_meas[c]["oracle"],
                         "sc": per_cond_meas[c]["sc"],
                         "disp": [round(x, 5) for x in per_cond_meas[c]["disp"]]}
                     for c in conds},
    }
    fp = os.path.join(OUT, f"exp_uncertainty_{dataset}.json")
    json.dump(out, open(fp, "w"), indent=1)
    print(f"[written] {fp}", flush=True)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl_8b")
    ap.add_argument("--datasets", nargs="*",
                    default=["screenspot_pro", "ui_vision", "osworld_g"])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--conds", nargs="*", default=DEFAULT_CONDS)
    ap.add_argument("--ports", type=int, nargs="*", default=None)
    ap.add_argument("--rewrites", default=os.path.join(HERE, "..", "instruction", "rewrites_v2",
                                                       "all_instructions.json"))
    args = ap.parse_args()
    spec = get_spec(args.model)
    OUT = os.path.join(HERE, "..", "out", args.model)
    os.makedirs(OUT, exist_ok=True)
    rewrites = None
    need_rewrite = any(c != "base" for c in args.conds)
    if args.rewrites and os.path.exists(args.rewrites):
        from rewrite_instructions import load_consolidated
        rewrites = load_consolidated(args.rewrites)
        print(f"[rewrites] loaded {len(rewrites)} entries")
    elif need_rewrite:
        raise SystemExit(f"conds {args.conds} need rewrites but {args.rewrites} missing")
    await wait_ready(args.ports)
    for ds in args.datasets:
        print(f"=== [{args.model}] Experiment U (uncertainty K={args.k} T={args.temperature}): {ds} ===")
        await run(spec, ds, args.limit, args.ports, OUT, args.k, args.temperature,
                  args.conds, rewrites)


if __name__ == "__main__":
    asyncio.run(main())
