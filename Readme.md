# GWSat — Groundwater Stress Satellite Classifier

> "Downlink the answer, not the data."

GWSat detects pre-drought groundwater stress from Sentinel-2 imagery **weeks before NDVI drops** — using IBM TerraMind as a frozen geospatial encoder fused with hand-crafted SWIR physics indices.

**Team:** NP Complete | **Target region:** Telangana, India | **Hardware target:** Jetson Orin NX / MNR Cubesat

---

## What This Does

GWSat classifies Sentinel-2 satellite tiles into three groundwater stress levels:

| Class | Label | Groundwater Depth | Meaning |
|-------|-------|-------------------|---------|
| 0 | ✅ Stable | < 5 m | Normal range, standard monitoring |
| 1 | ⚠️ Moderate | 5–10 m | Early stress — weeks before visual crop damage |
| 2 | 🚨 Critical | > 10 m | Stomatal closure confirmed — crop failure risk in 2–4 weeks |

The key insight: **SWIR bands (B11, B12) detect stomatal closure from water stress before NDVI detects canopy damage.** GWSat explicitly encodes this physics — NDVI alone cannot.

---

## Results

All numbers below are from real held-out Sentinel-2 patches (never seen during training), running on confirmed **real TerraMind** (`backend: terratorch`).

| Method | F1 Macro | Accuracy | Notes |
|--------|----------|----------|-------|
| NDVI Threshold (baseline) | 0.4756 | 0.5345 | Misses Moderate entirely (0% recall) |
| **GWSat [terratorch]** | **0.8602** | **0.8448** | TerraMind + SWIR physics fusion |

**Δ vs NDVI baseline: +0.3846 F1 Macro**

Per-class breakdown:

| Class | n | Accuracy | Precision | Recall |
|-------|---|----------|-----------|--------|
| Stable | 22 | 0.8182 | 0.7826 | 0.8182 |
| Moderate | 21 | 0.7619 | 0.8000 | 0.7619 |
| Critical | 15 | **1.0000** | **1.0000** | **1.0000** |

Critical class: **zero misses** — the most dangerous class is never overlooked.

NDVI baseline gets 0% recall on Moderate (classifies every Moderate patch as Stable), which is exactly the failure mode this system fixes.

**Model size:** 273 KB head checkpoint + ~13 MB TerraMind encoder  
**CPU latency:** 3.8 ms per 64×64 patch  
**Alert payload:** 64 bytes (vs 131 KB raw tile — 2,048× compression)

---

## Architecture

```
Raw Input [B, 8, 64, 64]  ← 8 Sentinel-2 bands [B4, B5, B6, B7, B8, B8A, B11, B12]
          │
          ▼
  SpectralAttention          (SE-style per-band weighting)
          │
    ┌─────┴──────────────────────────────────────┐
    │  TerraMind path (primary)                  │  EdgeBackbone path (fallback)
    │  TerramindEncoder                          │  SpectralIndexLayer → [B, 14, H, W]
    │  [B, 8, H, W] → [B, 192]                  │  DSConv blocks      → [B, 256]
    └───────────────────┬────────────────────────┘
                        │  deep_feat [B, embed_dim]
                        │
  _physics_scalars()    │  ← computed from raw bands independently
  [B, 6] ───────────────┤  (NDVI, REI, LSWI, NDWI, SWIR_ratio, CWC)
                        │
                        ▼
            PhysicsFusionHead
            Linear(embed_dim+6 → 256) → LayerNorm → GELU → Dropout
            Linear(256 → 64)          → LayerNorm → GELU → Dropout
            Linear(64 → 3)
                        │
                        ▼
            3-class logits → {Stable, Moderate, Critical}
```

**Key design choices:**
- TerraMind encoder is **frozen** — only the fusion head is trained. Prevents overfitting on the small dataset.
- Physics scalars are extracted from raw bands (before attention weighting) so they measure true physical signal.
- For scene inference: tiled 64×64 sliding window (stride 32) with conservative alert rule — if ≥15% of vegetated patches are Critical, escalate regardless of majority vote.

### Spectral Physics Fusion

Six indices appended to the TerraMind embedding before classification:

| Index | Formula | Physical meaning |
|-------|---------|-----------------|
| NDVI | (B8−B4)/(B8+B4) | Greenness — drops late, fooled by irrigation |
| RedEdge Chl | (B8A/B5)−1 | Chlorophyll — drops early under stress |
| LSWI | (B8−B11)/(B8+B11) | Direct leaf water content |
| NDWI | (B8A−B11)/(B8A+B11) | Open water signal |
| SWIR Ratio | B11/B12 | ABA proxy — stomatal closure signal |
| CWC | (B8−B12)/(B8+B12) | Canopy water content |

The SWIR Ratio and LSWI are the primary discriminators between Moderate and Critical drought — the physics that NDVI misses.

---

## Quick Start (under 10 minutes)

```bash
# 1. Clone and install
git clone https://github.com/LeafyChan/Blue-Green-Drought.git
cd Blue-Green-Drought
pip install -r requirements.txt

# 2. Install real TerraMind (recommended)
pip install git+https://github.com/IBM/terratorch.git
# Verify: look for "✅ TerraMind loaded via terratorch (embed_dim=192)"
# If you see "⚠️ DeiT-tiny PROXY" — fix the install before running.

# 3. Run demo on synthetic tiles (no data files needed)
python infer.py --demo

# 4. Or launch the Gradio demo
python demo_app.py
# → Open http://localhost:7860
```

---

## Full Pipeline

### Step 0 — Install dependencies

```bash
pip install -r requirements.txt

# TerraMind (required for real results):
pip install git+https://github.com/IBM/terratorch.git
# OR:
pip install terratorch
```

If you see `⚠️ TerraMind unavailable — using DeiT-tiny PROXY`, your results are from DeiT-tiny, not TerraMind. The `backend` field in every output file will tell you exactly what loaded.

### Step 1 — Build dataset

```bash
python build_real_dataset.py
```

If you have real Telangana TIF files:

```bash
# Split multi-band or per-band TIF files into processed/ folder
python split_bands.py --input_dir raw/

# Convert to .pt tensors
python tif_to_pt.py --folder ./Telangana_Instant_Data --out scene.pt
```

This produces `data/train.pt`, `data/val.pt`, and `data/test.pt`. The test split uses completely different seeds from training — it is a true holdout, not a sample of the same random process.

### Step 2 — Train

Standard training (cross-entropy + AdamW):

```bash
python train.py --epochs 60 --batch_size 32 --lr 3e-4
```

If Moderate recall is 0% (common with imbalanced data), use the fixed trainer instead:

```bash
python train_moderate_fix.py --epochs 80 --moderate_weight 5.0 --focal_gamma 2.0
# Uses: Focal loss + Mixup augmentation + cosine LR + Moderate weight ×5
```

Only `PhysicsFusionHead` and `SpectralAttention` weights are trained. The TerraMind encoder stays frozen.

The output will clearly show:
```
*** ACTIVE BACKEND: terratorch ***    ← real TerraMind
*** ACTIVE BACKEND: timm_proxy ***   ← DeiT-tiny proxy (terratorch not installed)
```

### Step 3 — Validate

```bash
# Fast: evaluate on held-out test split only
python validate.py

# With additional out-of-distribution TIF scene folders
# (folder path must contain 'stable', 'moderate', or 'critical' to auto-infer label)
python validate.py --scene_dirs path/to/extra_stable path/to/extra_critical

# Custom checkpoint
python validate.py --checkpoint checkpoints/best_head.pth
```

Output: `validation_results.json`

### Step 4 — Get all metrics

Run all of the following to get the full picture:

```bash
python validate.py
cat validation_results.json
cat results_moderate_fixed.json
```

**Model size:**

```bash
ls -lh checkpoints/best_head.pth
ls -lh checkpoints/gwsat_terratorch.onnx 2>/dev/null || echo "not exported yet"
```

**CPU inference latency:**

```bash
python3 -c "
import torch, time
from model import GWSatModel
model = GWSatModel(device='cpu')
model.load_head('checkpoints/best_head.pth')
model.eval()
x = torch.randn(1, 8, 64, 64)
for _ in range(3): model(x)
t0 = time.perf_counter()
for _ in range(50):
    with torch.no_grad(): model(x)
ms = (time.perf_counter() - t0) / 50 * 1000
print(f'CPU latency: {ms:.1f} ms per patch')
"
```

**Confidence scores across the test set:**

```bash
python3 -c "
import torch, numpy as np
from model import GWSatModel
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GWSatModel(device=device)
model.load_head('checkpoints/best_head.pth')
model.eval()

data = torch.load('data/test.pt', weights_only=False)
X, y = data['X'].float(), data['y'].numpy()

all_probs, all_preds = [], []
for i in range(0, len(X), 32):
    with torch.no_grad():
        logits = model(X[i:i+32].to(device))
    probs = F.softmax(logits, dim=1).cpu()
    all_probs.append(probs)
    all_preds.extend(probs.argmax(1).numpy())

all_probs = torch.cat(all_probs).numpy()
all_preds = np.array(all_preds)
confidences = all_probs.max(axis=1)

print(f'Confidence  max={confidences.max():.4f}  min={confidences.min():.4f}  mean={confidences.mean():.4f}')
print()
labels = ['Stable','Moderate','Critical']
for c in range(3):
    mask = y == c
    conf_c = confidences[mask]
    correct = (all_preds[mask] == c)
    print(f'{labels[c]:<10}  n={mask.sum()}  conf_mean={conf_c.mean():.3f}  '
          f'correct_conf={conf_c[correct].mean() if correct.any() else 0:.3f}  '
          f'wrong_conf={conf_c[~correct].mean() if (~correct).any() else 0:.3f}')
"
```

**F1 vs NDVI delta (clean summary):**

```bash
python3 -c "
import json
r = json.load(open('validation_results.json'))
ts = r['test_split']
gf = ts['gwsat']['f1_macro']
nf = ts['ndvi']['f1_macro']
ga = ts['gwsat']['accuracy']
print(f'GWSat  F1={gf:.4f}  Acc={ga:.4f}')
print(f'NDVI   F1={nf:.4f}')
print(f'Delta +{gf-nf:.4f}')
print(f'Backend: {r[\"backend\"]}')
"
```

### Step 5 — Export to ONNX

```bash
# Full-precision ONNX (filename automatically includes backend name)
python export.py --verify

# INT8 quantized for Jetson deployment
python export.py --quantize --verify

# TensorRT engine (run on Jetson)
trtexec --onnx=checkpoints/gwsat_terratorch.onnx \
        --saveEngine=checkpoints/gwsat.engine \
        --int8 --workspace=256
```

Output files:
- `checkpoints/gwsat_terratorch.onnx` — real TerraMind
- `checkpoints/gwsat_timm-proxy.onnx` — proxy, clearly labelled
- `checkpoints/gwsat_terratorch.meta.json` — metadata

### Step 6 — Run demo

```bash
python demo_app.py

# Shareable public URL
python demo_app.py --share
```

The demo accepts synthetic sample tiles, a folder of TIFs, or a `.pt` tensor file. Output includes backend name, stress class, confidence, spectral index values, and a spatial heatmap.

---

## Inference modes

```bash
# Demo on synthetic tiles (no data needed)
python infer.py --demo

# Full Sentinel-2 TIF scene (tiled, no resize aliasing)
python infer.py --scene B4.tif B5.tif B6.tif B7.tif B8.tif B8A.tif B11.tif B12.tif

# Single .pt tile
python infer.py --tile sample_input/sample_class2.pt

# GEE batch tensor
python infer.py --tensor data/inference/telangana_demo.pt
```

Scene mode tiles the image into 64×64 patches instead of resizing — this preserves the SWIR stress signal that gets destroyed by global averaging.

Minimal ONNX inference (for deployment):

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("checkpoints/gwsat_terratorch.onnx")
# Band order: [B4, B5, B6, B7, B8, B8A, B11, B12], values in [0, 1]
input_patch = np.random.randn(1, 8, 64, 64).astype(np.float32)
logits = session.run(None, {"s2_patch": input_patch})[0]
labels = ["Stable", "Moderate", "Critical"]
print(labels[logits.argmax()])
```

---

## Input / Output format

**Input:** 8-band Sentinel-2 L2A patch, shape `[8, 64, 64]`, band order `[B4, B5, B6, B7, B8, B8A, B11, B12]`, reflectance values `[0.0, 1.0]` (divide raw DN by 10000).

**Output:**

```json
{
    "stress_class": 1,
    "confidence": 0.87,
    "probabilities": {
        "Stable": 0.08,
        "Moderate": 0.87,
        "Critical": 0.05
    },
    "spectral_indices": {
        "NDVI": 0.42,
        "RedEdge_Index": 0.21,
        "LeafWaterStress_LSWI": -0.03,
        "SWIR_ratio": 1.21,
        "IR_Pressure_Index": 0.84
    },
    "raw_tile_bytes": 131072,
    "alert_bytes": 64
}
```

Scene inference additionally returns `patch_distribution`, `n_patches`, `skipped_patches`, and a spatial heatmap tensor.

---

## Backend transparency

The `backend_name` property tells you exactly what encoder loaded. Every results file includes this explicitly — there is no silent fallback.

| Value | Meaning |
|-------|---------|
| `terratorch` | ✅ Real IBM TerraMind via terratorch package |
| `hf_transformers` | ✅ Real IBM TerraMind via HuggingFace |
| `timm_proxy` | ⚠️ DeiT-tiny proxy — NOT TerraMind |
| `edge_cnn` | ❌ CNN fallback — all encoder strategies failed |

```python
from model import GWSatModel
model = GWSatModel(device="cpu")
print(model.backend_name)   # → 'terratorch'
```

Results with `timm_proxy` will be similar on synthetic data (the head learns the physics features regardless of backbone) but will underperform on real out-of-distribution imagery where TerraMind's geospatial pre-training matters.

---

## On-orbit continual learning

`ocl.py` implements memory-bounded continual learning for cubesat deployment. The production model serves all inference (frozen). A shadow head trains on labelled corrections. At orbit boundaries, `MissionSupervisor` gates the swap: requires F1 improvement > 0.02 and no single class dropping > 0.10. A `RollbackBank` stores the last 3 checkpoints.

Memory budget: ~18 MB for a 150-tile correction buffer. Not yet tested on physical Jetson hardware.

---

## File reference

| File | Purpose |
|------|---------|
| `model.py` | Core architecture — TerraMind encoder + Physics Fusion Head |
| `train.py` | Standard training loop (head only, encoder frozen) |
| `train_weighted.py` | Weighted CE to address Moderate class imbalance |
| `train_moderate_fix.py` | Focal loss + Mixup — fixes 0% Moderate recall |
| `validate.py` | Holdout validation + NDVI baseline comparison |
| `infer.py` | All inference modes: demo, scene, tile, tensor |
| `inference.py` | Minimal ONNX runtime example |
| `data_pipeline.py` | Synthetic Sentinel-2 tile generator |
| `build_real_dataset.py` | Builds train/val/test .pt tensors from raw data |
| `tif_to_pt.py` | Converts GeoTIFF to PyTorch tensors (auto band detection) |
| `split_bands.py` | Splits multi-band or per-band TIF files into processed/ |
| `calculate_ndvi.py` | Patch-level spectral analysis and NDVI baseline reporting |
| `demo_app.py` | Gradio web demo with heatmap and spectral charts |
| `export.py` | ONNX export with backend-aware wrapper + INT8 quantization |
| `ocl.py` | On-orbit continual learning (shadow training, rollback bank) |
| `visualize.py` | Heatmap and false-colour composite utilities |
| `run.py` | Convenience entry point for common workflows |
| `requirements.txt` | All dependencies with minimum versions |
| `RUN_GUIDE.md` | Detailed fix log and troubleshooting guide |

---

## Edge inference feasibility

| Metric | Value |
|--------|-------|
| Head checkpoint | 273 KB |
| TerraMind-tiny encoder | ~13 MB |
| INT8 quantized ONNX | ~3–5 MB |
| CPU latency (64×64 patch) | **3.8 ms** |
| Jetson latency (INT8, estimated) | ~15–30 ms |
| RAM footprint (inference only) | < 256 MB |
| Alert output size | 64 bytes |

Fits on Jetson Orin NX with INT8 quantization. The TensorRT engine can be pre-compiled on-ground and flashed alongside weights. On-orbit inference requires only the ONNX runtime (< 30 MB).

---

## Honest limitations

- **Training data is real but small.** 12 Sentinel-2 acquisitions (5 Stable, 2 Moderate, 5 Critical). Moderate is underrepresented — the Stable/Moderate boundary is partly seasonal rather than fully physical. The fix (`train_moderate_fix.py`) addresses this with Focal loss and Mixup.
- **No borewell ground-truth co-registration yet.** Labels are derived from CGWB district records and scene-level expert annotation, not per-patch borewell depth measurements. The next step is scraping CGWB open records to co-register 50+ patches with measured GWL depths.
- **OCL not tested on physical hardware.** Power and thermal constraints on-orbit may require further tuning of the update budget.
- **ONNX export on terratorch backend** may require opset≥17 depending on terratorch version. Use `python export.py --verify` to confirm the exported graph runs correctly.

---

## Dependencies

```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
transformers>=4.35.0
huggingface_hub>=0.19.0
timm>=0.9.0
rasterio>=1.3.0
pandas>=2.0.0
Pillow>=10.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
gradio>=4.0.0
matplotlib>=3.7.0
onnx>=1.15.0
onnxruntime>=1.15.0
```

Install all at once: `pip install -r requirements.txt`

TerraMind (separate install): `pip install git+https://github.com/IBM/terratorch.git`

---

*Team NP Complete — Built for the TM2Space Hackathon*  
*Targeting: Telangana groundwater stress early-warning using IBM TerraMind + SWIR physics fusion*