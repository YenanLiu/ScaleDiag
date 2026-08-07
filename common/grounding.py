"""Standalone GUI-grounding core: dataset loading, hit test, and
aspect-ratio-preserving crop helpers.

Self-contained (no external project dependencies). All datasets use the shared
ScreenSpot-Pro annotation schema:
    {"img_filename", "bbox":[x0,y0,x1,y1] pixels, "instruction", "img_size":[W,H],
     optional "ui_type"/"group"/"platform"/"application"}
"""
from __future__ import annotations

import os
import glob
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- registry ---
# Override with env GUI_DATA_ROOT (absolute path to a folder that contains
# annotations/ and images/ for each benchmark). See configs/paths.env.example.
DATA_ROOT = os.environ.get("GUI_DATA_ROOT", os.path.expanduser("~/gui_grounding_data"))

DATASETS: Dict[str, Dict[str, Any]] = {
    # Each entry expects ScreenSpot-Pro schema annotations + image folder.
    "screenspot_pro": {
        "ann": os.path.join(DATA_ROOT, "screenspot_pro", "annotations"),
        "img": os.path.join(DATA_ROOT, "screenspot_pro", "images"),
        "res": "high",
    },
    "ui_vision": {
        "ann": os.path.join(DATA_ROOT, "ui_vision", "annotations"),
        "img": os.path.join(DATA_ROOT, "ui_vision", "images"),
        "res": "high",
    },
    "osworld_g": {
        "ann": os.path.join(DATA_ROOT, "osworld_g", "OSWorld-G_sspro_format.json"),
        "img": os.path.join(DATA_ROOT, "osworld_g", "images"),
        "res": "mid",
    },
    "mmbench_gui": {
        "ann": os.path.join(DATA_ROOT, "mmbench_gui", "MMbench_GUI_sspro_format.json"),
        "img": os.path.join(DATA_ROOT, "mmbench_gui", "images"),
        "res": "mid",
    },
    "screenspot_v2": {
        "ann": os.path.join(DATA_ROOT, "screenspot_v2", "annotations"),
        "img": os.path.join(DATA_ROOT, "screenspot_v2", "images"),
        "res": "low",
    },
}


@dataclass
class Item:
    id: Any
    dataset: str
    img_path: str
    bbox: List[float]           # pixel [x0,y0,x1,y1]
    instruction: str
    img_w: int
    img_h: int
    ui_type: str = "unknown"
    group: str = "unknown"
    platform: str = "unknown"
    application: str = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def gt_center(self) -> Tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2

    @property
    def target_area_frac(self) -> float:
        w = max(0.0, self.bbox[2] - self.bbox[0])
        h = max(0.0, self.bbox[3] - self.bbox[1])
        return w * h / max(1.0, self.img_w * self.img_h)


def _norm_bbox(raw) -> List[float]:
    """Normalize a bbox to [x0,y0,x1,y1]. Some datasets (e.g. osworld_g) store
    extra polygon vertices after the box; derive the axis-aligned box via
    min/max over x- and y-coords so all experiments see a clean 4-tuple."""
    vals = [float(v) for v in raw]
    if len(vals) == 4:
        return vals
    xs, ys = vals[0::2], vals[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def _as_str(v) -> str:
    """Coerce a metadata field to a hashable string (some datasets store lists)."""
    if isinstance(v, (list, tuple)):
        return "/".join(str(x) for x in v) if v else "unknown"
    return str(v)


def load_items(dataset: str, limit: int = -1, seed: int = 0) -> List[Item]:
    cfg = DATASETS[dataset]
    ann, img_dir = cfg["ann"], cfg["img"]
    files = sorted(glob.glob(os.path.join(ann, "*.json"))) if os.path.isdir(ann) else [ann]
    items: List[Item] = []
    for fp in files:
        data = json.load(open(fp))
        for it in data:
            fn = it.get("img_filename") or it.get("image_path")
            if not fn or "bbox" not in it or "instruction" not in it:
                continue
            wh = it.get("img_size") or [0, 0]
            items.append(Item(
                id=it.get("id", f"{os.path.basename(fp)}:{len(items)}"),
                dataset=dataset,
                img_path=os.path.join(img_dir, fn),
                bbox=_norm_bbox(it["bbox"]),
                instruction=it["instruction"],
                img_w=int(wh[0]), img_h=int(wh[1]),
                ui_type=_as_str(it.get("ui_type", "unknown")),
                group=_as_str(it.get("group", "unknown")),
                platform=_as_str(it.get("platform", "unknown")),
                application=_as_str(it.get("application", "unknown")),
            ))
    if seed >= 0:
        import random
        random.Random(seed).shuffle(items)
    if limit > 0:
        items = items[:limit]
    return items


# ------------------------------------------------------------ prompt/parse ---
SYSTEM_PROMPT = (
    "You are a GUI grounding model. Given a screenshot and an instruction, "
    "output one click location.\n\nOutput Format:\nReturn a JSON object within "
    "<tool_call></tool_call> tags with the following format:\n<tool_call>\n"
    '{"arguments": {"coordinate": [x, y]}}\n</tool_call>\n\nRules:\n'
    "- The coordinate [x, y] must be integers in the range [0, 1000]\n"
    "- Use top-left origin (0,0 is top-left corner)\n"
    "- x increases to the right, y increases downward\n"
    "- Return exactly one coordinate point\n\nExample:\n<tool_call>\n"
    '{"arguments": {"coordinate": [500, 300]}}\n</tool_call>\n'
)


def smart_resize(height: int, width: int, factor: int = 28,
                 min_pixels: int = 3136, max_pixels: int = 12_845_056
                 ) -> Tuple[int, int]:
    """Qwen2.5-VL smart-resize: returns (resized_h, resized_w), each a multiple
    of `factor`, with area within [min_pixels, max_pixels]. Idempotent."""
    import math
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def hit(bbox_px, point_px) -> bool:
    x, y = point_px
    return bbox_px[0] <= x <= bbox_px[2] and bbox_px[1] <= y <= bbox_px[3]


# --- aspect-preserving cropping --
def aspect_crop(img_w: int, img_h: int, cx: float, cy: float, frac: float
                ) -> Tuple[int, int, int, int]:
    """A crop centred at (cx,cy) whose area is `frac` of the full image and
    whose aspect ratio equals the full image's. Clamped inside the image.
    frac in (0,1]; frac=1 -> full image."""
    frac = max(1e-4, min(1.0, frac))
    s = frac ** 0.5
    cw, ch = img_w * s, img_h * s
    x0 = cx - cw / 2
    y0 = cy - ch / 2
    # shift to stay inside
    x0 = min(max(0, x0), img_w - cw)
    y0 = min(max(0, y0), img_h - ch)
    return int(round(x0)), int(round(y0)), int(round(x0 + cw)), int(round(y0 + ch))


def remap_point_from_crop(px_norm, crop_box, full_w, full_h):
    """Map a [0,1] point predicted on the crop back to full-image pixels."""
    x0, y0, x1, y1 = crop_box
    x = x0 + px_norm[0] * (x1 - x0)
    y = y0 + px_norm[1] * (y1 - y0)
    return x, y
