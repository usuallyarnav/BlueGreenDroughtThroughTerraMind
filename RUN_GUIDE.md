# GWSat v3 — Run Guide  (FIXED)
## What changed and why it matters

---

## 🚨 Critical Issues Fixed

### 1. TerraMind was being used as a checkbox

**Before:** The model silently fell back to a DeiT-tiny timm proxy when terratorch wasn't installed — and kept calling it "TerraMind" in all output. `results.json` said TerraMind. It wasn't.

**After:** `model.backend_name` tells you exactly what loaded:

| Value | Meaning |
|-------|---------|
| `terratorch` | ✅ Real IBM TerraMind (terratorch package) |
| `hf_transformers` | ✅ Real TerraMind via HuggingFace |
| `timm_proxy` | ⚠️ DeiT-tiny PROXY — NOT TerraMind |
| `edge_cnn` | ❌ CNN fallback — all encoder strategies failed |

Every results file now includes `"backend"` and `"note"` keys. A `timm_proxy` run explicitly says "DeiT-tiny proxy — NOT TerraMind".

### 2. No true holdout validation

**Before:** `test.pt` was split from the same synthetic distribution as `train.pt`. That's not holdout — it's a held-out sample of the same random process.

**After:** `validate.py` builds a separate holdout set with **completely different seeds** (offset +99999), runs your real `scene.pt` tiles, and runs 5-fold CV with confidence intervals.

### 3. Export was backend-unaware

**Before:** `export.py` had a `_Wrapper` that called `spectral_idx` manually, which broke for the TerraMind path.

**After:** `_ExportWrapper` just calls `_forward_features()` (the shared method), which handles all backends correctly. The timm resize lives inside `TerramindEncoder.forward()` so ONNX captures it as part of the graph.

---

## Step 0 — Install TerraMind (do this first)

```bash
# Option A — official (recommended)
pip install terratorch

# Option B — from source
pip install git+https://github.com/IBM/terratorch.git

# Verify it loaded (you'll see this in any script output):
# ✅ TerraMind loaded via terratorch (embed_dim=192)
# vs
# ⚠️ TerraMind unavailable — using DeiT-tiny PROXY via timm.
```

If you see "PROXY", your results are from DeiT-tiny, not TerraMind. Fix the install before claiming TerraMind performance.

---

## Step 1 — Train

```bash
python train.py --epochs 40 --n_per_class 300

# Output will clearly show:
# *** ACTIVE BACKEND: terratorch ***    ← good
# *** ACTIVE BACKEND: timm_proxy ***   ← means terratorch isn't installed
```

---

## Step 2 — Validate (the new critical step)

```bash
# Holdout synthetic only (fast, ~2 min)
python validate.py --no_cv

# With your real Telangana scene
python validate.py --real_pt scene.pt --real_label 1 --no_cv
# (real_label: 0=Stable, 1=Moderate, 2=Critical — use your best estimate)

# Full validation with 5-fold CV (slow, ~20 min)
python validate.py --real_pt scene.pt --real_label 1

# Output: validation_results.json
```

### What validate.py tests

| Test | Data source | Why |
|------|------------|-----|
| NDVI baseline | Holdout synthetic (unseen seeds) | Establishes physics-only floor |
| Multi-index baseline | Holdout synthetic | Better physics-only comparison |
| GWSat model | Holdout synthetic | True generalization test |
| Real scene patches | Your actual TIF data | Out-of-distribution test |
| 5-fold CV | Full synthetic pool | Confidence interval on F1 |

---

## Step 3 — Convert TIF files

```bash
python tif_to_pt.py --folder ./Telangana_Instant_Data --out scene.pt
python validate.py --real_pt scene.pt --real_label 1 --no_cv
```

---

## Step 4 — Export

```bash
# Export (filename includes backend name automatically)
python export.py --verify

# Output:
# checkpoints/gwsat_terratorch.onnx      ← real TerraMind
# checkpoints/gwsat_timm-proxy.onnx      ← proxy (weaker)
# checkpoints/gwsat_terratorch.meta.json ← metadata

# INT8 for Jetson
python export.py --quantize --verify
trtexec --onnx=checkpoints/gwsat_terratorch.onnx \
        --saveEngine=checkpoints/gwsat.engine --int8 --workspace=256
```

---

## Step 5 — Demo

```bash
python demo_app.py --share
```

The demo now shows `backend=terratorch` (or `timm_proxy`) in the output so judges know exactly what they're evaluating.

---

## Reading validation_results.json

```json
{
  "backend": "terratorch",
  "holdout_synthetic": {
    "ndvi":  {"f1_macro": 0.18, ...},
    "multi_index": {"f1_macro": 0.34, ...},
    "gwsat": {"f1_macro": 0.85, ...},
    "delta_vs_ndvi":  0.67,
    "delta_vs_multi": 0.51
  },
  "real_scene": {
    "scene_prediction": "Moderate",
    "scene_confidence": 0.71,
    "ground_truth": "Moderate",
    "correct": true,
    "patch_metrics": {"f1_macro": 0.79, ...}
  },
  "cross_validation": {
    "cv_f1_mean": 0.84,
    "cv_f1_std":  0.03
  }
}
```

**A result is only trustworthy if:**
- `backend` is `terratorch` or `hf_transformers`
- `holdout_synthetic.gwsat.f1_macro` > `holdout_synthetic.ndvi.f1_macro` by a meaningful margin
- `real_scene` exists and shows reasonable patch metrics

---

## Expected honest results

```
Backend: terratorch (real TerraMind)
────────────────────────────────────────────────────
NDVI threshold (holdout)         F1≈0.18   Acc≈0.37
Multi-index physics (holdout)    F1≈0.34   Acc≈0.47
GWSat [terratorch] (holdout)     F1≈0.83   Acc≈0.85
5-fold CV                        F1≈0.83 ± 0.03

Backend: timm_proxy (DeiT-tiny — NOT TerraMind)
────────────────────────────────────────────────────
(Numbers will be similar because training data is synthetic
and the model learns the physics features regardless of backbone.
The difference matters most on real out-of-distribution data.)
```