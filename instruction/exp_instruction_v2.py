"""Experiment B (v2) — instruction information scaling with high-quality,
image-grounded rewrites.

Uses the GPT-5.2 rewrites produced from a RED-BOX-marked screenshot
(rewrite_instructions.py, --out ../rewrites_v2), which include four single
perspectives plus every pair/triple/quad combination. This lets us trace
grounding accuracy as a function of how many perspectives (how much
information) the instruction carries: base -> single -> pair -> triple -> quad.

Speed: all 16 instruction variants of one screenshot are grounded as a *group*
routed to the same vLLM server (client.ground_grouped), so the expensive 4K
image prefix is prefill-cached and reused across variants instead of recomputed
16x. Items are processed in chunks to bound memory / in-flight requests.

Writes out/<model>/exp_instruction_v2_<dataset>.json:
  {dataset, n, conditions: {cond: {n, hit, parse_fail, acc, coverage}}}
where cond is "base" or "llm_<view>" for each of the 15 views.
"""
from __future__ import annotations

import os
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import json
import asyncio
import argparse

from grounding import load_items, hit
from client import GroundingPool, wait_ready
from models import get_spec
from rewrite_instructions import VIEWS, load_consolidated

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ["base"] + VIEWS  # base = original instruction, then the 15 variants


async def run(spec, dataset, limit, ports, OUT, rewrites, chunk=150):
    items = load_items(dataset, limit=limit)
    pool = GroundingPool(spec, ports)

    # accumulators per condition
    acc = {c: {"n": 0, "hit": 0, "parse_fail": 0, "coverage": 0} for c in CONDS}

    def instr_for(it, cond):
        if cond == "base":
            return it.instruction, True
        rec = rewrites.get((dataset, str(it.id)))
        txt = (rec or {}).get(cond) or ""
        return (txt, True) if txt else (it.instruction, False)

    for c0 in range(0, len(items), chunk):
        chunk_items = items[c0:c0 + chunk]
        groups = []              # (img_path, [instr per cond])
        have = []                # parallel: [bool coverage per cond]
        for it in chunk_items:
            instrs, cov = [], []
            for c in CONDS:
                txt, ok = instr_for(it, c)
                instrs.append(txt)
                cov.append(ok)
            groups.append((it.img_path, instrs))
            have.append(cov)
        res = await pool.ground_grouped(groups)  # per-group list aligned to CONDS
        for it, per, cov in zip(chunk_items, res, have):
            for ci, c in enumerate(CONDS):
                pt, _raw = per[ci]
                a = acc[c]
                a["n"] += 1
                if cov[ci]:
                    a["coverage"] += 1
                if pt is None:
                    a["parse_fail"] += 1
                    continue
                a["hit"] += int(hit(it.bbox, (pt[0] * it.img_w, pt[1] * it.img_h)))
        print(f"  {dataset:15s} {c0 + len(chunk_items)}/{len(items)}  "
              f"base={acc['base']['hit']/max(1,acc['base']['n']):.3f}", flush=True)

    conditions = {}
    for c in CONDS:
        a = acc[c]
        conditions[c if c == "base" else f"llm_{c}"] = {
            "n": a["n"], "hit": a["hit"], "parse_fail": a["parse_fail"],
            "acc": a["hit"] / max(1, a["n"]), "coverage": a["coverage"]}
    out = {"dataset": dataset, "n": len(items), "conditions": conditions}
    fp = os.path.join(OUT, f"exp_instruction_v2_{dataset}.json")
    json.dump(out, open(fp, "w"), indent=1)
    # short scaling summary: best single vs best pair vs best triple vs quad
    def best(views):
        vals = [conditions[f"llm_{v}"]["acc"] for v in views
                if f"llm_{v}" in conditions]
        return max(vals) if vals else float("nan")
    from rewrite_instructions import SINGLE_VIEWS, PAIR_VIEWS, TRIPLE_VIEWS, QUAD_VIEW
    print(f"[written] {fp}  base={conditions['base']['acc']:.3f} "
          f"best1={best(SINGLE_VIEWS):.3f} best2={best(PAIR_VIEWS):.3f} "
          f"best3={best(TRIPLE_VIEWS):.3f} quad={best(QUAD_VIEW):.3f}", flush=True)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3vl_8b")
    ap.add_argument("--datasets", nargs="*",
                    default=["screenspot_v2", "mmbench_gui", "osworld_g",
                             "screenspot_pro", "ui_vision"])
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--ports", type=int, nargs="*", default=None)
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--rewrites", default=os.path.join(HERE, "rewrites_v2",
                                                        "all_instructions.json"))
    args = ap.parse_args()
    spec = get_spec(args.model)
    OUT = os.path.join(HERE, "..", "out", args.model)
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(args.rewrites):
        raise SystemExit(f"missing rewrite file {args.rewrites}")
    rewrites = load_consolidated(args.rewrites)
    print(f"[rewrites] loaded {len(rewrites)} entries from {args.rewrites}", flush=True)
    await wait_ready(args.ports)
    for ds in args.datasets:
        print(f"=== [{args.model}] Experiment B v2 (marked-image rewrites): {ds} ===",
              flush=True)
        await run(spec, ds, args.limit, args.ports, OUT, rewrites, chunk=args.chunk)


if __name__ == "__main__":
    asyncio.run(main())
