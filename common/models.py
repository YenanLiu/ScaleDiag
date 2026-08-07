"""Model registry + per-model prompt / coordinate decoders.

Different GUI-grounding models speak different coordinate dialects:
  - norm1000    : coordinate is [0,1000] normalized  (Qwen3-VL, MAI-UI)
  - abs_resized : coordinate is absolute pixels in the smart-resized input
                  space (Qwen2.5-VL, GTA1, GUI-G2, and UI-Ins)

For abs_resized models the client must send the image at the model's own
smart-resize fixed point so the returned pixels map back exactly; we read each
model's processor bounds (min/max pixels, patch*merge factor) from its
preprocessor_config.json. For norm1000 models the mapping is resize-independent.
"""
from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

MODELS_ROOT = os.environ.get("MODELS_ROOT", os.path.expanduser("~/models"))

# ---- coordinate parsers ----

def _last_coord_bracket(text: str) -> Optional[Tuple[float, float]]:
    """last `coordinate ... [x, y]` (reasoning + tool_call models)."""
    if "coordinate" not in text:
        m = re.findall(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", text)
        if m:
            return float(m[-1][0]), float(m[-1][1])
        return None
    seg = text.rsplit("coordinate", 1)[1]
    m = re.search(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", seg)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _first_coord_bracket(text: str) -> Optional[Tuple[float, float]]:
    if "coordinate" in text:
        seg = text.split("coordinate", 1)[1]
    else:
        seg = text
    m = re.search(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", seg)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _paren_xy(text: str) -> Optional[Tuple[float, float]]:
    m = re.search(r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)", text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _bbox4_center(text: str) -> Optional[Tuple[float, float]]:
    m = re.search(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", text)
    if not m:
        return None
    x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
    return (x1 + x2) / 2, (y1 + y2) / 2


PARSERS = {
    "coord_last": _last_coord_bracket,
    "coord_first": _first_coord_bracket,
    "paren": _paren_xy,
    "bbox4": _bbox4_center,
}

# ---- prompt fragments -----------------------------------------------------

GROUNDING_REASONING_SYS = (
    "You are a GUI grounding model. Given a screenshot and an instruction, "
    "output one click location.\n\nOutput Format:\nReturn a JSON object within "
    "<tool_call></tool_call> tags with the following format:\n<tool_call>\n"
    '{"arguments": {"coordinate": [x, y]}}\n</tool_call>\n\nRules:\n'
    "- The coordinate [x, y] must be integers in the range [0, 1000]\n"
    "- Use top-left origin (0,0 is top-left corner)\n"
    "- x increases to the right, y increases downward\n"
    "- Return exactly one coordinate point\n"
)

QWEN25VL_GROUNDING_SYS = (
    "You are a helpful assistant.\n"
    "You are a GUI agent. You are given a task and your action history, with "
    "screenshots. You need to perform the next action to complete the task. "
    "\n\n## Output Format\nReturn a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n```\n<tool_call>\n"
    '{"name": "grounding", "arguments": <args-json-object>}\n</tool_call>\n```\n\n'
    "<args-json-object> represents the following item of the action space:\n\n"
    '## Action Space\n{"action": "click", "coordinate": [x, y]}'
)

# Qwen3-VL computer_use tool prompt (coordinate space is 1000x1000)
QWEN3VL_COMPUTERUSE_SYS = (
    "You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions "
    "to assist with the user query.\n\nYou are provided with function signatures "
    "within <tools></tools> XML tags:\n<tools>\n"
    '{"type": "function", "function": {"name": "computer_use", "description": '
    '"Use a mouse and keyboard to interact with a computer. The screen\'s '
    'resolution is 1000x1000. Make sure to click elements with the cursor tip in '
    'the center of the element.", "parameters": {"properties": {"action": '
    '{"description": "The action to perform.", "enum": ["left_click"], "type": '
    '"string"}, "coordinate": {"description": "(x, y): the x and y coordinates, '
    'each in the range [0, 1000].", "type": "array"}}, "required": ["action"], '
    '"type": "object"}}}\n</tools>\n\nFor each function call, return a json object '
    "with function name and arguments within <tool_call></tool_call> XML tags:\n"
    '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)

MAIUI_SYS = (
    "You are a GUI grounding agent. \n## Task\nGiven a screenshot and the user's "
    "grounding instruction. Your task is to accurately locate a UI element based "
    "on the user's instructions.\nFirst, you should carefully examine the "
    "screenshot and analyze the user's instructions, translate the user's "
    "instruction into a effective reasoning process, and then provide the final "
    "coordinate.\n## Output Format\nReturn a json object with a reasoning process "
    "in <grounding_think></grounding_think> tags, a [x,y] format coordinate within "
    "<answer></answer> XML tags:\n<grounding_think>...</grounding_think>\n<answer>\n"
    '{"coordinate": [x,y]}\n</answer>\n## Input instruction\n'
)


@dataclass
class ModelSpec:
    key: str
    path: str
    tp: int = 1
    coord_space: str = "abs_resized"   # or "norm1000"
    parser: str = "coord_last"
    system: str = ""
    img_first: bool = True
    max_tokens: int = 256
    #: optional forced assistant prefix (continue_final_message) to lock the
    #: output format for chatty Instruct models; parsed as prefill+completion.
    prefill: str = ""
    display: str = ""
    # filled from preprocessor_config.json
    factor: int = 28
    min_pixels: int = 3136
    max_pixels: int = 12_845_056

    def load_processor_bounds(self):
        p = os.path.join(self.path, "preprocessor_config.json")
        if os.path.exists(p):
            c = json.load(open(p))
            ps = c.get("patch_size", 14)
            ms = c.get("merge_size", 2)
            self.factor = int(ps) * int(ms)
            # old format: min_pixels/max_pixels; new (Qwen3-VL/3.5): size.{shortest,longest}_edge
            mn = c.get("min_pixels") or c.get("size", {}).get("shortest_edge")
            if mn:
                self.min_pixels = int(mn)
            # cap max_pixels to stay well under served max-model-len
            mp = c.get("max_pixels") or c.get("size", {}).get("longest_edge")
            if mp:
                self.max_pixels = min(int(mp), 12_845_056)
        return self

    def parse(self, text: str):
        return PARSERS[self.parser](text)


# Set MODELS_ROOT or per-model absolute paths. Or register a one-off model via
# env MODEL_KEY / MODEL_PATH (see get_spec).
REGISTRY: Dict[str, ModelSpec] = {
    "qwen3vl_8b": ModelSpec(
        key="qwen3vl_8b", path=f"{MODELS_ROOT}/Qwen3-VL-8B-Instruct", tp=1,
        coord_space="norm1000", parser="coord_last",
        system=QWEN3VL_COMPUTERUSE_SYS, img_first=True, max_tokens=32,
        prefill='<tool_call>\n{"name": "computer_use", "arguments": '
                '{"action": "left_click", "coordinate": [',
        display="Qwen3-VL-8B-Instruct"),
    "maiui_8b": ModelSpec(
        key="maiui_8b", path=f"{MODELS_ROOT}/MAI-UI-8B", tp=1,
        coord_space="norm1000", parser="coord_last",
        system=MAIUI_SYS, img_first=False, max_tokens=512,
        display="MAI-UI-8B"),
    "qwen3vl_32b": ModelSpec(
        key="qwen3vl_32b", path=f"{MODELS_ROOT}/Qwen3-VL-32B-Instruct", tp=2,
        coord_space="norm1000", parser="coord_last",
        # 32B tends to emit tokens before the coordinate on dense/high-res
        # screenshots; 32 new tokens truncates before the point (parse_fail
        # 75-94% on hi-res). Give it reasoning headroom; coord_last grabs the
        # final [x,y] regardless of any preamble.
        system=QWEN3VL_COMPUTERUSE_SYS, img_first=True, max_tokens=256,
        prefill='<tool_call>\n{"name": "computer_use", "arguments": '
                '{"action": "left_click", "coordinate": [',
        display="Qwen3-VL-32B-Instruct"),
    # ---- Qwen3.5 series (qwen3_5 arch; Qwen3VLProcessor; computer_use 1000x1000)
    # NOTE: model_type `qwen3_5` (hybrid linear-attention VL) is NOT supported by
    # vLLM<=0.11.0. Serving requires a newer vLLM/SGLang; set VLLM=<env>/bin/vllm.
    "qwen35_4b": ModelSpec(
        key="qwen35_4b", path=f"{MODELS_ROOT}/Qwen3.5-4B", tp=1,
        coord_space="norm1000", parser="coord_last",
        system=QWEN3VL_COMPUTERUSE_SYS, img_first=True, max_tokens=32,
        prefill='<tool_call>\n{"name": "computer_use", "arguments": '
                '{"action": "left_click", "coordinate": [',
        display="Qwen3.5-4B"),
    "qwen35_9b": ModelSpec(
        key="qwen35_9b", path=f"{MODELS_ROOT}/Qwen3.5-9B", tp=1,
        coord_space="norm1000", parser="coord_last",
        system=QWEN3VL_COMPUTERUSE_SYS, img_first=True, max_tokens=32,
        prefill='<tool_call>\n{"name": "computer_use", "arguments": '
                '{"action": "left_click", "coordinate": [',
        display="Qwen3.5-9B"),
    # closest dense to the requested "32B" is 27B (no 32B in Qwen3.5 series)
    "qwen35_27b": ModelSpec(
        key="qwen35_27b", path=f"{MODELS_ROOT}/Qwen3.5-27B", tp=2,
        coord_space="norm1000", parser="coord_last",
        system=QWEN3VL_COMPUTERUSE_SYS, img_first=True, max_tokens=32,
        prefill='<tool_call>\n{"name": "computer_use", "arguments": '
                '{"action": "left_click", "coordinate": [',
        display="Qwen3.5-27B"),
    # extra grounding models available if needed:
    "qwen25vl_7b": ModelSpec(
        key="qwen25vl_7b", path=f"{MODELS_ROOT}/Qwen2.5-VL-7B-Instruct", tp=1,
        coord_space="abs_resized", parser="coord_first",
        system=QWEN25VL_GROUNDING_SYS, img_first=True, max_tokens=64,
        display="Qwen2.5-VL-7B-Instruct"),
    "gta1_7b": ModelSpec(
        key="gta1_7b", path=f"{MODELS_ROOT}/GTA1-7B", tp=1,
        coord_space="abs_resized", parser="paren",
        system="", img_first=True, max_tokens=64, display="GTA1-7B"),
}


def get_spec(key: str) -> ModelSpec:
    """Resolve a ModelSpec.

    Priority:
      1. If MODEL_PATH is set, build/override a spec for MODEL_KEY (default=key)
         so users can point at any local checkpoint without editing this file.
      2. Else look up REGISTRY[key] (paths under MODELS_ROOT).
    """
    env_path = os.environ.get("MODEL_PATH", "").strip()
    env_key = os.environ.get("MODEL_KEY", key).strip() or key
    if env_path:
        if env_key in REGISTRY:
            spec = REGISTRY[env_key]
            spec.path = env_path
        else:
            # default to Qwen3-VL computer_use dialect
            spec = ModelSpec(
                key=env_key, path=env_path, tp=int(os.environ.get("MODEL_TP", "1")),
                coord_space=os.environ.get("MODEL_COORD", "norm1000"),
                parser=os.environ.get("MODEL_PARSER", "coord_last"),
                system=QWEN3VL_COMPUTERUSE_SYS, img_first=True,
                max_tokens=int(os.environ.get("MODEL_MAX_TOKENS", "256")),
                prefill=os.environ.get(
                    "MODEL_PREFILL",
                    '<tool_call>\n{"name": "computer_use", "arguments": '
                    '{"action": "left_click", "coordinate": [',
                ),
                display=env_key,
            )
            REGISTRY[env_key] = spec
        return spec.load_processor_bounds()
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown model key {key!r}. Set MODEL_PATH, or add it to REGISTRY. "
            f"Known: {sorted(REGISTRY)}"
        )
    return REGISTRY[key].load_processor_bounds()
