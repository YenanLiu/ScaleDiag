# ScaleDiag

**ScaleDiag** is a public evaluation framework for GUI click-grounding. It diagnoses **where failures come from** by scaling three axes (plus a plain baseline):

| Folder | Experiment | Question |
|--------|------------|----------|
| `baseline/` | Full-image accuracy | How accurate is the model as usually reported? |
| `visual_scaling/` | Crop / effective resolution | Does making the target larger in pixels help? |
| `instruction/` | Instruction information | Does adding / rewriting language fix errors? |
| `sampling/` | Multi-sample test-time compute | Does best-of-K / self-consistency help? |
| `serving/` | vLLM pool | How to serve the model for the scripts above |

Results land in `out/<model_key>/*.json`.

Smoke-tested with `Qwen3-VL-8B-Instruct`, `--datasets screenspot_v2 --limit 3 --ports 8001` (baseline / crop / instruction / language / sample / uncertainty all OK).

---

## 0. Setup

```bash
pip install -r requirements.txt
# Also need a vLLM install that can serve your VL checkpoint.
cp configs/paths.env.example configs/paths.env
# edit GUI_DATA_ROOT, MODELS_ROOT, VLLM
source configs/paths.env
```

### Data layout (`GUI_DATA_ROOT`)

Each benchmark uses the ScreenSpot-Pro item schema:

`img_filename`, `instruction`, `bbox=[x1,y1,x2,y2]`, `img_size=[W,H]`, optional `ui_type`.

```
$GUI_DATA_ROOT/
  screenspot_pro/{annotations/*.json, images/}
  ui_vision/{annotations/*.json, images/}
  osworld_g/{OSWorld-G_sspro_format.json, images/}
  mmbench_gui/{MMbench_GUI_sspro_format.json, images/}
  screenspot_v2/{annotations/*.json, images/}
```

`annotations/` may be a directory of JSON lists, or (for osworld/mmbench) a single consolidated JSON file.

### Model path

Either put checkpoints under `MODELS_ROOT` with the names in `common/models.py`, **or** override every run:

```bash
export MODEL_KEY=qwen3vl_8b
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
```

Non–Qwen3-VL dialects: set `MODEL_COORD` / `MODEL_PARSER` / `MODEL_PREFILL`, or add a `ModelSpec` in `common/models.py`.

---

## 1. Serve the model (`serving/`)

Starts one vLLM worker per GPU (or TP groups). Default ports = `BASE_PORT+1, +2, …` (e.g. `8001`).

```bash
source configs/paths.env
MODEL=$MODEL_PATH GPUS="0 1 2 3" bash serving/run_serve.sh
# stop:
bash serving/stop_pool.sh
```

Useful env vars: `TP`, `BASE_PORT` (default 8000), `MAXLEN`, `GPUFRAC`, `VLLM`, `NAME` (served name, default `grounder`).

Eval scripts take `--ports 8001 8002 …`. If omitted, they auto-detect any healthy ports in `8001–8007`.

Minimal single-GPU smoke serve:

```bash
MODEL=$MODEL_PATH GPUS="0" bash serving/run_serve.sh
```

---

## 2. Baseline (`baseline/`)

**Code:** `baseline/exp_baseline.py`

**What it tests:** Standard full-screenshot grounding accuracy (and per-`ui_type` breakdown). No crop, no instruction rewrite, greedy decode.

**Run:**

```bash
source configs/paths.env
python baseline/exp_baseline.py --model qwen3vl_8b --datasets screenspot_v2 --limit 3 --ports 8001
# full bench: --limit -1
# or: bash baseline/run.sh --datasets screenspot_v2 --limit 3 --ports 8001
```

**Output:** `out/<model>/baseline_<dataset>.json` → `{acc, n, parse_fail, by_ui_type}`.

---

## 3. Visual scaling (`visual_scaling/`)

### How cropping works

Implemented in `common/grounding.py` → `aspect_crop`:

1. Pick the ground-truth click target center `(cx, cy)`.
2. Choose a crop whose **area** is `frac` of the full image (`frac ∈ {1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125}`).
3. Keep the **same aspect ratio** as the full screenshot (side scale = `√frac`).
4. Center the window on `(cx, cy)`, then clamp so it stays inside the image.
5. Send **only the crop** to the model; map the predicted `[0,1]` point back to full-image pixels with `remap_point_from_crop`, then `hit(bbox, point)`.

So `frac=1.0` is the usual full-image baseline; smaller `frac` makes the widget occupy more of the pixels the model sees (higher effective resolution / occupancy), at the cost of context. Crops are **GT-centred** (oracle location) so the curve isolates resolution, not “did we crop the right place”.

After the model’s own smart-resize, `analyze_target_size.eff_sizes` also records occupancy / short-edge px / vision tokens for each sample.

### Eval script

**Code:** `visual_scaling/exp_crop.py`

**What it tests:** Accuracy vs crop `frac` (resolution lever).

```bash
python visual_scaling/exp_crop.py --model qwen3vl_8b \
  --datasets screenspot_v2 --limit 3 --ports 8001
```

**Output:** `out/<model>/exp_crop_<dataset>.json`  
→ `curve[frac].acc` plus per-sample `hit` / `occ_frac` / `eff_short_px`.

### Offline size table (no GPU)

**Code:** `visual_scaling/analyze_target_size.py` (also imported by the crop runner for `eff_sizes`)

```bash
python visual_scaling/analyze_target_size.py --model qwen3vl_8b \
  --datasets screenspot_v2 --limit 20
# after a crop run:  --from-runs
# multi-model within-item analysis:  --within
```

---

## 4. Instruction (`instruction/`)

Three related probes; image is always the **full** screenshot.

### 4a. Attribute injection (Exp B)

**Code:** `instruction/exp_instruction.py`

**What it tests:** Append GT-derived cues (position / color / size / shape) to the original instruction. Isolates whether cheap descriptive attributes help.

Conditions: `base`, `+pos`, `+color`, `+size`, `+shape`, and combinations up to `+all`.

```bash
python instruction/exp_instruction.py --model qwen3vl_8b \
  --datasets screenspot_v2 --limit 3 --ports 8001
```

**Output:** `out/<model>/exp_instruction_<dataset>.json`

### 4b. Language / information ladder (Exp D)

**Code:** `instruction/exp_language.py`

**What it tests:** Monotone linguistic information on the **same** screenshot:

| Condition | Instruction |
|-----------|-------------|
| `generic` | Fixed generic “click the UI element…” (info floor) |
| `keep_half` | First ~50% of tokens of the original instruction |
| `base` | Original instruction |
| `clarify_pos` | Base + coarse 3×3 region (oracle) |
| `clarify_fine` | Base + approx. `%` from left/top (oracle) |

```bash
python instruction/exp_language.py --model qwen3vl_8b \
  --datasets screenspot_v2 --limit 3 --ports 8001
```

**Output:** `out/<model>/exp_language_<dataset>.json`

### 4c. Multi-perspective rewrites (Exp F) — data + eval

**Data prep code:** `instruction/rewrite_instructions.py`

**How rewrite data is built:**

1. Draw a bright red box on the GT widget (full screenshot + zoomed crop).
2. Call an OpenAI-compatible VLM/LLM with both images.
3. Ask for 15 perspective combinations: `app` / `spa` / `fun` / `goa` and all pairs / triples / quad.
4. Write `instruction/rewrites_v2/<dataset>.json` and consolidated `all_instructions.json`.

```bash
export OPENAI_API_KEY=sk-...
# optional: export OPENAI_BASE_URL=https://api.openai.com/v1
python instruction/rewrite_instructions.py \
  --datasets screenspot_v2 --limit 10 \
  --out instruction/rewrites_v2 --concurrency 8
```

**Eval code:** `instruction/exp_instruction_v2.py` — accuracy for `base` vs each `llm_<view>`.

```bash
python instruction/exp_instruction_v2.py --model qwen3vl_8b \
  --datasets screenspot_v2 --limit 3 --ports 8001 \
  --rewrites instruction/rewrites_v2/all_instructions.json
```

---

## 5. Sampling (`sampling/`)

### 5a. Greedy / medoid / oracle@K (Exp E)

**Code:** `sampling/exp_sample.py`

**What it tests:** Test-time sampling headroom on the full image + original instruction.

| Metric | Meaning |
|--------|---------|
| `greedy` | Temperature 0, one decode |
| `sc_medoid` | Among K samples (temp `T`), pick the geometric medoid (verifier-free) |
| `oracle@K` | Hit if **any** of K samples is correct (BoN upper bound) |
| `dispersion` | Mean pairwise distance of the K points (uncertainty) |

```bash
python sampling/exp_sample.py --model qwen3vl_8b \
  --datasets screenspot_v2 --limit 3 --ports 8001 --k 4 --temperature 0.7
```

**Output:** `out/<model>/exp_sample_<dataset>.json`

### 5b. Uncertainty (Exp U)

**Code:** `sampling/exp_uncertainty.py`

**What it tests:** Whether sample dispersion predicts errors (AUROC / risk-coverage), and how that interacts with richer instructions (needs rewrite file for non-`base` conditions).

```bash
# base instruction only (no rewrite file needed):
python sampling/exp_uncertainty.py --model qwen3vl_8b \
  --datasets screenspot_v2 --limit 3 --ports 8001 \
  --k 4 --temperature 0.7 --conds base

# with rewrites (e.g. base vs fun vs app):
python sampling/exp_uncertainty.py --model qwen3vl_8b \
  --datasets screenspot_v2 --limit 3 --ports 8001 \
  --k 8 --conds base fun app \
  --rewrites instruction/rewrites_v2/all_instructions.json
```

---

## 6. Swap only the model (typical loop)

```bash
source configs/paths.env
export MODEL_KEY=qwen3vl_8b
export MODEL_PATH=/path/to/Your-Checkpoint

MODEL=$MODEL_PATH GPUS="0 1 2 3" bash serving/run_serve.sh

python baseline/exp_baseline.py --model $MODEL_KEY --limit -1 --ports 8001 8002 8003 8004
python visual_scaling/exp_crop.py --model $MODEL_KEY --limit 300 --ports 8001 8002 8003 8004
python instruction/exp_instruction.py --model $MODEL_KEY --limit 300 --ports 8001 8002 8003 8004
python instruction/exp_language.py --model $MODEL_KEY --limit 300 --ports 8001 8002 8003 8004
python sampling/exp_sample.py --model $MODEL_KEY --limit 300 --k 8 --ports 8001 8002 8003 8004

bash serving/stop_pool.sh
```

Thin wrappers under each folder (`run.sh`, `run_attr.sh`, …) source `configs/paths.env` and call the same Python entrypoints.

---

## Layout

```
ScaleDiag/
  common/           grounding.py  client.py  models.py
  serving/          serve_pool.sh  stop_pool.sh  run_serve.sh
  baseline/         exp_baseline.py
  visual_scaling/   exp_crop.py  analyze_target_size.py
  instruction/      exp_instruction.py  exp_language.py
                    rewrite_instructions.py  exp_instruction_v2.py
  sampling/         exp_sample.py  exp_uncertainty.py
  configs/          paths.env.example
  out/              run outputs (gitignored)
```
