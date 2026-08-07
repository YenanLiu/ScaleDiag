"""Rewrite GUI-grounding instructions with GPT-5.2, grounded on a MARKED image.

v2 motivation
-------------
v1 handed GPT only *derived* attributes (colour / coarse position / size / shape).
Those are noisy: on ScreenSpot-Pro ~84% of targets collapsed to "white"/"black",
and the coarse ui-type field mislabels many text buttons as icons. GPT then
faithfully wrote those wrong cues into the instruction.

v2 fixes the grounding: we draw a bright red box around the ground-truth element
on (a) a downscaled full screenshot and (b) a zoomed crop, and send BOTH images
to GPT-5.2. GPT now *sees* the exact target, so it can describe the real colour,
text, icon and neighbours; the derived attributes are passed only as soft hints
and GPT is told to trust the pixels when they disagree.

v2 also tests linguistic *scaling* properly. Instead of only four single-view
instructions (each of which drops information relative to the original), we ask
for every combination of the four perspectives:

  singles (4): app, spa, fun, goa
  pairs   (6): app_spa, app_fun, app_goa, spa_fun, spa_goa, fun_goa
  triples (4): app_spa_fun, app_spa_goa, app_fun_goa, spa_fun_goa
  quad    (1): app_spa_fun_goa

so eval can trace accuracy as a function of how many perspectives (how much
information) the instruction carries.

  app : appearance  — on-element text / icon / colour + coarse location
  spa : spatial     — screen region + relation to neighbouring elements
  fun : functional  — what the element does when clicked
  goa : goal        — the user's higher-level intent

Output (per dataset, resumable):
  <out>/<dataset>.json         : {id: {ori, <15 views>, attrs, meta}}
  <out>/<dataset>_split.json   : flat list (one row per view)
  <out>/all_instructions.json  : consolidated {data: {dataset: {id: rec}}}

Config (env):
  OPENAI_API_KEY   (required)
  OPENAI_BASE_URL  (optional; defaults to https://api.openai.com/v1)

Usage:
  python rewrite_instructions.py --datasets screenspot_v2 mmbench_gui \
      --limit -1 --out ../rewrites_v2 --concurrency 16
"""
from __future__ import annotations

import os
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import json
import base64
import asyncio
import argparse
from io import BytesIO
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw

from grounding import load_items, Item
from exp_instruction import (
    pos_label, color_label, size_label, shape_label, COLORS,
)

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))

# perspective views, then every combination (ori is kept verbatim, not generated)
SINGLE_VIEWS = ["app", "spa", "fun", "goa"]
PAIR_VIEWS = ["app_spa", "app_fun", "app_goa", "spa_fun", "spa_goa", "fun_goa"]
TRIPLE_VIEWS = ["app_spa_fun", "app_spa_goa", "app_fun_goa", "spa_fun_goa"]
QUAD_VIEW = ["app_spa_fun_goa"]
VIEWS = SINGLE_VIEWS + PAIR_VIEWS + TRIPLE_VIEWS + QUAD_VIEW  # 15

_PERSPECTIVE_DEF = {
    "app": "appearance — identify it by its visible look: the on-element "
           "text/label, icon/symbol, and colour, plus a coarse screen location",
    "spa": "spatial — identify it by WHERE it is: its screen region and its "
           "relation to neighbouring elements (left/right/above/below of ...)",
    "fun": "functional — identify it by WHAT IT DOES when clicked (no appearance "
           "or location cues)",
    "goa": "goal — the user's higher-level intent that this click serves",
}


def _combo_line(key: str) -> str:
    parts = key.split("_")
    names = " + ".join(_PERSPECTIVE_DEF[p].split(" — ")[0] for p in parts)
    return (f"  {key}: fuse {names} into ONE natural, fluent instruction that "
            f"weaves those cues together (not a list).")


SYSTEM = (
    "You are an expert annotator building GUI-grounding instructions. In the "
    "screenshot you are shown, exactly ONE target UI element is outlined by a "
    "bright RED rectangle; a zoomed-in crop of that same target (also red-boxed) "
    "is provided for detail. Describe THIS element.\n"
    "You also get the original instruction and some auto-derived hints (colour, "
    "position, size, shape, ui-type). The hints may be wrong — TRUST WHAT YOU SEE "
    "in the image over the hints when they disagree (e.g. fix the element type or "
    "colour).\n\n"
    "Write instructions from four perspectives:\n"
    f"  app: {_PERSPECTIVE_DEF['app']}.\n"
    f"  spa: {_PERSPECTIVE_DEF['spa']}.\n"
    f"  fun: {_PERSPECTIVE_DEF['fun']}.\n"
    f"  goa: {_PERSPECTIVE_DEF['goa']}.\n"
    "Then also write every COMBINATION that merges these perspectives into a "
    "single richer instruction:\n"
    + "\n".join(_combo_line(k) for k in PAIR_VIEWS + TRIPLE_VIEWS + QUAD_VIEW)
    + "\n\n"
    "Rules: every value is a natural, fluent, unambiguous English instruction a "
    "person could follow WITHOUT seeing the red box. NEVER mention the box, the "
    "red rectangle, a marker, coordinates, percentages, pixels, bounding boxes, "
    "or the words 'ground truth'/'attribute'. Keep every view about the SAME "
    "single target. Combination views must genuinely include all their named "
    "cues. Singles must stay single-perspective (e.g. `fun` must not reveal "
    "colour or location).\n"
    "Return ONLY a JSON object whose keys are EXACTLY: "
    + ", ".join(VIEWS) + "."
)


def precise_pos(cx_frac: float, cy_frac: float) -> str:
    return (f"{round(cx_frac * 100)}% from the left edge, "
            f"{round(cy_frac * 100)}% from the top edge")


def derive_attrs(it: Item, img: Image.Image) -> dict:
    cx, cy = it.gt_center
    cxf, cyf = cx / max(1, it.img_w), cy / max(1, it.img_h)
    return {
        "coarse_pos": pos_label(cxf, cyf),
        "precise_pos": precise_pos(cxf, cyf),
        "color": color_label(img, it.bbox),
        "size": size_label(it.target_area_frac),
        "shape": shape_label(it.bbox),
        "ui_type": it.ui_type,
        "application": it.application,
        "platform": it.platform,
        "group": it.group,
    }


def _norm_box(bbox, w, h):
    vals = [float(v) for v in bbox]
    if len(vals) == 4:
        x0, y0, x1, y1 = vals
    else:
        xs, ys = vals[0::2], vals[1::2]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(w), max(x0 + 1, x1)), min(float(h), max(y0 + 1, y1))
    return x0, y0, x1, y1


def _b64(im: Image.Image) -> str:
    buf = BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _draw_marker(im: Image.Image, box, thick_frac: float, min_lw: int,
                 dilate_frac: float, min_gap: int, color=(255, 0, 0)):
    """Draw a bold frame that sits *outside* the element (dilated outward by a
    small gap) so it never occludes the target. Line width scales with resolution;
    a white halo underneath keeps it visible on light or dark backgrounds."""
    W, H = im.size
    x0, y0, x1, y1 = box
    lw = max(min_lw, round(thick_frac * max(W, H)))
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    gap = max(min_gap, round(dilate_frac * min(bw, bh)))
    # outward-dilated frame, clamped so the lines stay on-canvas and visible
    inset = lw  # keep the outer (halo) edge inside the image
    rx0 = max(inset, x0 - gap); ry0 = max(inset, y0 - gap)
    rx1 = min(W - 1 - inset, x1 + gap); ry1 = min(H - 1 - inset, y1 + gap)
    if rx1 <= rx0 or ry1 <= ry0:  # tiny image / huge box -> fall back to raw box
        rx0, ry0, rx1, ry1 = x0, y0, x1, y1
    d = ImageDraw.Draw(im)
    d.rectangle([rx0 - lw, ry0 - lw, rx1 + lw, ry1 + lw],
                outline=(255, 255, 255), width=lw)
    d.rectangle([rx0, ry0, rx1, ry1], outline=color, width=lw)


def marked_full(img: Image.Image, bbox, max_side: int = 1600, color=(255, 0, 0)) -> str:
    """Downscaled full screenshot with the target framed (dilated)."""
    w, h = img.size
    x0, y0, x1, y1 = _norm_box(bbox, w, h)
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    else:
        im = img.copy()
    _draw_marker(im, [x0 * scale, y0 * scale, x1 * scale, y1 * scale],
                 thick_frac=0.006, min_lw=4, dilate_frac=0.12, min_gap=3, color=color)
    return _b64(im)


def marked_crop(img: Image.Image, bbox, pad_frac: float = 1.4,
                max_side: int = 672, color=(255, 0, 0)) -> str:
    """Zoomed crop around the target (with local context) + red frame. We
    thumbnail first, then draw, so the frame stays bold at the final resolution."""
    w, h = img.size
    x0, y0, x1, y1 = _norm_box(bbox, w, h)
    bw, bh = x1 - x0, y1 - y0
    px, py = bw * pad_frac, bh * pad_frac
    cx0, cy0 = int(max(0, x0 - px)), int(max(0, y0 - py))
    cx1, cy1 = int(min(w, x1 + px)), int(min(h, y1 + py))
    patch = img.crop((cx0, cy0, max(cx0 + 1, cx1), max(cy0 + 1, cy1))).convert("RGB")
    ow, oh = patch.size
    patch.thumbnail((max_side, max_side))
    s = patch.size[0] / max(1, ow)  # thumbnail scale (uniform)
    _draw_marker(patch, [(x0 - cx0) * s, (y0 - cy0) * s,
                         (x1 - cx0) * s, (y1 - cy0) * s],
                 thick_frac=0.016, min_lw=3, dilate_frac=0.14, min_gap=3, color=color)
    return _b64(patch)


def build_user_content(it: Item, attrs: dict, full_b64: Optional[str],
                       crop_b64: Optional[str]) -> list:
    content = []
    if full_b64 is not None:
        content.append({"type": "text",
                        "text": "Full screenshot (target = red box):"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{full_b64}"}})
    if crop_b64 is not None:
        content.append({"type": "text",
                        "text": "Zoomed crop of the same target (red box):"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}})
    facts = (
        f"Original instruction: {it.instruction}\n"
        f"Application/context: {attrs['application']} on {attrs['platform']} "
        f"({attrs['group']})\n"
        f"Hint element type (may be wrong): {attrs['ui_type']}\n"
        f"Hint dominant colour (may be wrong): {attrs['color']}\n"
        f"Hint shape: {attrs['shape']}; hint relative size: {attrs['size']}\n"
        f"Hint coarse screen region: {attrs['coarse_pos']}\n"
    )
    content.append({"type": "text", "text": facts})
    return content


# bounded thread pool for blocking image decode + marker drawing (off event loop)
_EXEC = ThreadPoolExecutor(max_workers=16)


def _prep(it: Item, max_side: int, with_images: bool):
    """Blocking: open the screenshot, derive hints, draw the marked full+crop,
    then release the full image (no cross-item cache -> bounded memory)."""
    try:
        img = Image.open(it.img_path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (max(1, it.img_w), max(1, it.img_h)))
    attrs = derive_attrs(it, img)
    full_b64 = crop_b64 = None
    if with_images:
        try:
            full_b64 = marked_full(img, it.bbox, max_side=max_side)
            crop_b64 = marked_crop(img, it.bbox)
        except Exception:
            full_b64 = crop_b64 = None
    img.close()
    return attrs, full_b64, crop_b64


async def _call(client, model, it: Item, attrs, full_b64, crop_b64, retries=3):
    content = build_user_content(it, attrs, full_b64, crop_b64)
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": content}],
        response_format={"type": "json_object"},
        max_tokens=1600,
    )
    for attempt in range(retries):
        try:
            r = await client.chat.completions.create(**kwargs)
            txt = r.choices[0].message.content or "{}"
            obj = json.loads(txt)
            return {v: (obj.get(v) or "").strip() for v in VIEWS}
        except Exception as e:
            msg = str(e)
            if attempt == 0 and "response_format" in msg:
                kwargs.pop("response_format", None)
            elif attempt == retries - 1:
                return {"_error": msg[:300]}
            await asyncio.sleep(3.0 * (2 ** attempt))  # 3s, 6s, 12s
    return {"_error": "exhausted"}


async def run(client, sem, model, dataset, limit, out_dir, max_side, with_images):
    items = load_items(dataset, limit=limit)
    fp = os.path.join(out_dir, f"{dataset}.json")
    done: Dict[str, dict] = {}
    if os.path.exists(fp):
        done = json.load(open(fp))

    def _complete(rec):  # cached record counts as done only if it has content
        return rec is not None and any(rec.get(v) for v in SINGLE_VIEWS)

    todo = [it for it in items if not _complete(done.get(str(it.id)))]
    n_ok0 = sum(1 for it in items if _complete(done.get(str(it.id))))
    print(f"=== {dataset}: {len(items)} items, {n_ok0} done, {len(todo)} to do "
          f"(incl. retry of errored) ===", flush=True)

    loop = asyncio.get_event_loop()

    async def work(it: Item):
        async with sem:
            attrs, full_b64, crop_b64 = await loop.run_in_executor(
                _EXEC, _prep, it, max_side, with_images)
            views = await _call(client, model, it, attrs, full_b64, crop_b64)
        return it, attrs, views

    n_ok = n_err = 0
    tasks = [asyncio.create_task(work(it)) for it in todo]
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        it, attrs, views = await fut
        if "_error" in views:
            n_err += 1
        else:
            n_ok += 1
        rec = {
            "ori": it.instruction,
            **{v: views.get(v, "") for v in VIEWS},
            "attrs": attrs,
            "img_filename": os.path.basename(it.img_path),
            "bbox": it.bbox,
            "img_size": [it.img_w, it.img_h],
            "ui_type": it.ui_type,
        }
        if "_error" in views:
            rec["_error"] = views["_error"]
        done[str(it.id)] = rec
        if i % 25 == 0 or i == len(tasks):
            json.dump(done, open(fp, "w"), ensure_ascii=False, indent=1)
            print(f"  [{dataset}] {i}/{len(tasks)}  ok={n_ok} err={n_err}", flush=True)
    json.dump(done, open(fp, "w"), ensure_ascii=False, indent=1)

    # flat split (one row per view)
    split = []
    for _id, rec in done.items():
        for k in ["ori"] + VIEWS:
            txt = rec.get(k) or ""
            if not txt:
                continue
            split.append({
                "id": _id, "img_filename": rec["img_filename"],
                "bbox": rec["bbox"], "img_size": rec["img_size"],
                "ui_type": rec["ui_type"], "instruction": txt,
                "instruction_type": f"{k}_instruction",
            })
    sp = os.path.join(out_dir, f"{dataset}_split.json")
    json.dump(split, open(sp, "w"), ensure_ascii=False, indent=1)
    print(f"[written] {fp}  (+{len(split)} rows -> {sp})", flush=True)


ALL_DATASETS = ["screenspot_pro", "ui_vision", "mmbench_gui", "osworld_g", "screenspot_v2"]


def consolidate(out_dir: str, datasets: List[str], model: str, base_url: str) -> str:
    data: Dict[str, dict] = {}
    total = 0
    for ds in datasets:
        fp = os.path.join(out_dir, f"{ds}.json")
        if not os.path.exists(fp):
            continue
        recs = json.load(open(fp))
        clean = {k: v for k, v in recs.items()
                 if any(v.get(x) for x in SINGLE_VIEWS)}
        data[ds] = clean
        total += len(clean)
    obj = {
        "meta": {"model": model, "base_url": base_url, "views": ["ori"] + VIEWS,
                 "n_total": total, "datasets": {d: len(v) for d, v in data.items()}},
        "data": data,
    }
    fp = os.path.join(out_dir, "all_instructions.json")
    json.dump(obj, open(fp, "w"), ensure_ascii=False, indent=1)
    print(f"[consolidated] {total} items across {len(data)} datasets -> {fp}", flush=True)
    return fp


def load_consolidated(path: str) -> Dict[tuple, dict]:
    """Eval-side helper: returns {(dataset, str(id)): record}. `record` has keys
    ori + the 15 perspective/combination variants + attrs."""
    obj = json.load(open(path))
    out = {}
    for ds, recs in obj["data"].items():
        for _id, rec in recs.items():
            out[(ds, str(_id))] = rec
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=ALL_DATASETS)
    ap.add_argument("--limit", type=int, default=-1)  # -1 = full
    ap.add_argument("--model", default=os.environ.get("REWRITE_MODEL", "gpt-5.2-2025-12-11"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=1600,
                    help="downscale the marked full screenshot to this long side")
    ap.add_argument("--no-images", action="store_true",
                    help="text-only hints (v1 behaviour); default sends marked images")
    ap.add_argument("--out", default=os.path.join(HERE, "rewrites_v2"))
    ap.add_argument("--consolidate-only", action="store_true")
    args = ap.parse_args()

    from openai import AsyncOpenAI
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    os.makedirs(args.out, exist_ok=True)

    if args.consolidate_only:
        consolidate(args.out, args.datasets, args.model, base_url)
        return

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set. export it (and OPENAI_BASE_URL "
                         "if you use a proxy) before running.")
    client = AsyncOpenAI(api_key=key, base_url=base_url, max_retries=8, timeout=180)
    sem = asyncio.Semaphore(args.concurrency)
    print(f"model={args.model}  base_url={base_url}  images={not args.no_images}  "
          f"max_side={args.max_side}  concurrency={args.concurrency}  "
          f"views={len(VIEWS)}", flush=True)
    for ds in args.datasets:
        await run(client, sem, args.model, ds, args.limit, args.out,
                  args.max_side, not args.no_images)
        consolidate(args.out, args.datasets, args.model, base_url)


if __name__ == "__main__":
    asyncio.run(main())
