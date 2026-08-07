"""Experiment E — Test-time sampling & the RL/GRPO headroom.

Question: does drawing several samples (best-of-N / self-consistency) buy more
accuracy than a single greedy decode, and how much of that is reachable WITHOUT
an external verifier (i.e. what a GRPO-style policy could internalise)?

For each item we measure, at sampling temperature T with K draws:
  greedy        : temperature 0, single decode              (current policy pass@1)
  sc_medoid     : self-consistency = the sample point that   (verifier-FREE gain:
                  minimises summed distance to the other K-1  pick the mode)
  pass@1 (mean) : average single-sample hit rate at temp T   (policy at temp T)
  oracle@K      : ANY of the K samples hits                  (UPPER BOUND for BoN)
  dispersion    : mean pairwise distance of the K points     (sampling uncertainty)

Interpretation (ties to the two-regime story):
  - oracle@K - greedy   = total sampling headroom (what BoN + a perfect verifier
    could reach). A GRPO/RLVR policy tries to convert this into pass@1.
  - sc_medoid - greedy  = the FREE, verifier-less gain already available at test
    time (self-consistency), and a proxy for how "self-consistent-correct" the
    model is (GRPO reinforces exactly these self-agreeing correct samples).
  - large oracle gap + small dispersion on visually-hard (icon/high-res) items
    => compute/visual-limited, sampling helps; large dispersion that does NOT
    hit even at K (low oracle@K) => information-limited, sampling can't help.

Writes out/<model>/exp_sample_<dataset>.json.
"""
from __future__ import annotations

import os
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import json
import asyncio
import argparse
import itertools

from PIL import Image

from grounding import load_items, hit
from client import GroundingPool, wait_ready
from models import get_spec

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))


def _medoid(points):
    """Return index of the point minimising summed L2 distance to the others."""
    valid = [(i, p) for i, p in enumerate(points) if p is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0][0]
    best_i, best_d = valid[0][0], float("inf")
    for i, p in valid:
        d = 0.0
        for j, q in valid:
            if i == j:
                continue
            d += ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def _dispersion(points):
    valid = [p for p in points if p is not None]
    if len(valid) < 2:
        return 0.0
    tot = cnt = 0.0
    for a, b in itertools.combinations(valid, 2):
        tot += ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        cnt += 1
    return tot / cnt


async def run(spec, dataset, limit, ports, OUT, k, temperature):
    items = load_items(dataset, limit=limit)
    pool = GroundingPool(spec, ports)

    def get_img(p):
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            return Image.new("RGB", (128, 128))

    jobs = [(get_img(it.img_path), it.instruction) for it in items]

    # greedy (temperature 0)
    greedy = await pool.ground_many(jobs)
    # K samples at temperature T
    sampled = await pool.ground_repeated(jobs, k=k, temperature=temperature)

    n = len(items)
    acc_greedy = acc_sc = acc_oracle = 0
    pass1_sum = 0.0
    disp_sum = 0.0
    pf_greedy = 0
    # per-item distribution of "how many of the K samples are correct" (0..K).
    # This is the quantity that governs GRPO/RLVR: a group's reward variance (and
    # hence its policy-gradient signal) is nonzero only for 0<m<K, and maximal
    # near m=K/2; m=0 is information-limited (RL can't help), m=K is already solved.
    correct_count_hist = [0] * (k + 1)
    greedy_hit_by_cc = [0] * (k + 1)   # of items with m correct, how many greedy already hits
    sc_hit_by_cc = [0] * (k + 1)       # ... how many coordinate-medoid consensus hits
    for it, (gp, _), samples in zip(items, greedy, sampled):
        # greedy
        g_ok = 0
        if gp is None:
            pf_greedy += 1
        else:
            g_ok = int(hit(it.bbox, (gp[0] * it.img_w, gp[1] * it.img_h)))
            acc_greedy += g_ok
        pts = [p for p, _ in samples]
        hits = [int(hit(it.bbox, (p[0] * it.img_w, p[1] * it.img_h)))
                if p is not None else 0 for p in pts]
        # pass@1 (expected single-sample), oracle@K
        if hits:
            pass1_sum += sum(hits) / len(hits)
        acc_oracle += int(any(hits))
        # self-consistency medoid
        sc_ok = 0
        mi = _medoid(pts)
        if mi is not None and pts[mi] is not None:
            sc_ok = int(hit(it.bbox, (pts[mi][0] * it.img_w, pts[mi][1] * it.img_h)))
            acc_sc += sc_ok
        disp_sum += _dispersion(pts)
        # bin this item by its number-correct-out-of-K
        nc = sum(hits)
        if nc <= k:
            correct_count_hist[nc] += 1
            greedy_hit_by_cc[nc] += g_ok
            sc_hit_by_cc[nc] += sc_ok

    learnable = sum(correct_count_hist[1:k])  # 0<m<K: nonzero GRPO reward variance
    res = {
        "dataset": dataset, "n": n, "k": k, "temperature": temperature,
        "acc_greedy": acc_greedy / max(n, 1),
        "acc_pass1_mean": pass1_sum / max(n, 1),
        "acc_self_consistency": acc_sc / max(n, 1),
        "acc_oracle_at_k": acc_oracle / max(n, 1),
        "mean_dispersion_frac": disp_sum / max(n, 1),
        "parse_fail_greedy": pf_greedy,
        "sampling_headroom": acc_oracle / max(n, 1) - acc_greedy / max(n, 1),
        "free_sc_gain": acc_sc / max(n, 1) - acc_greedy / max(n, 1),
        # --- GRPO / RLVR headroom decomposition (per-item correct-count) ---
        "correct_count_hist": correct_count_hist,        # index m = #items with exactly m/K correct
        "greedy_hit_by_correct_count": greedy_hit_by_cc,
        "sc_hit_by_correct_count": sc_hit_by_cc,
        "frac_info_limited": correct_count_hist[0] / max(n, 1),   # m=0: RL can't help
        "frac_already_solved": correct_count_hist[k] / max(n, 1),  # m=K: nothing to learn
        "frac_grpo_learnable": learnable / max(n, 1),            # 0<m<K: usable gradient
    }
    print(f"  {dataset:16s} greedy={res['acc_greedy']:.3f}  sc={res['acc_self_consistency']:.3f}"
          f"  oracle@{k}={res['acc_oracle_at_k']:.3f}  disp={res['mean_dispersion_frac']:.3f}"
          f"  (headroom={res['sampling_headroom']:+.3f}, free_sc={res['free_sc_gain']:+.3f})"
          f"  [cc_hist={res['correct_count_hist']} learnable={res['frac_grpo_learnable']:.3f}"
          f" info_limited={res['frac_info_limited']:.3f}]",
          flush=True)
    fp = os.path.join(OUT, f"exp_sample_{dataset}.json")
    json.dump(res, open(fp, "w"), indent=1)
    print(f"[written] {fp}")
    return res


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl_8b")
    ap.add_argument("--datasets", nargs="*",
                    default=["screenspot_pro", "ui_vision", "osworld_g"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--ports", type=int, nargs="*", default=None)
    args = ap.parse_args()
    spec = get_spec(args.model)
    OUT = os.path.join(HERE, "..", "out", args.model)
    os.makedirs(OUT, exist_ok=True)
    await wait_ready(args.ports)
    for ds in args.datasets:
        print(f"=== [{args.model}] Experiment E (sampling K={args.k} T={args.temperature}): {ds} ===")
        await run(spec, ds, args.limit, args.ports, OUT, args.k, args.temperature)


if __name__ == "__main__":
    asyncio.run(main())
