<div align="center">

# 🛰️ GWSat — Groundwater Stress Satellite Classifier

> *"Downlink the answer, not the data."*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![IBM TerraMind](https://img.shields.io/badge/IBM-TerraMind-054ada?logo=ibm)](https://github.com/IBM/terratorch)
[![ONNX](https://img.shields.io/badge/Export-ONNX-005CED?logo=onnx)](https://onnx.ai/)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-FF6B35?logo=gradio)](https://gradio.app/)
[![🏆 2nd Place — TM2Space Hackathon](https://img.shields.io/badge/🏆_TM2Space_Hackathon-2nd_Place-silver)](https://github.com)

**GWSat detects groundwater stress weeks before visible crop damage — using Sentinel-2 satellite imagery, IBM TerraMind embeddings, and SWIR physics fusion.**

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Results](#-results) · [Full Pipeline](#-full-pipeline) · [Deploy](#-edge-deployment) · [Limitations](#-honest-limitations)

---

*Team **NP Complete** · Target Region: **Telangana, India** · Hardware Target: **Jetson Orin NX / MNR CubeSat***

</div>

---

## 🌍 Why This Exists

Traditional NDVI-based monitoring detects drought only after visible canopy damage — at which point crop failure is already likely. GWSat targets the **silent window**: the 2–4 weeks when groundwater is already critically depleted, stomata are starting to close, but fields still look green.

By fusing IBM TerraMind's geospatial embeddings with hand-crafted SWIR physics indices, GWSat classifies Sentinel-2 tiles into three groundwater stress levels with **86% F1 Macro** — compared to 47% for NDVI alone.

| Class | Label | Groundwater Depth | Interpretation |
|:-----:|:-----:|:-----------------:|:--------------|
| 0 | ✅ Stable | < 5 m | Normal range, standard monitoring frequency |
| 1 | ⚠️ Moderate | 5–10 m | Early stress — weeks before visual crop damage |
| 2 | 🚨 Critical | > 10 m | Stomatal closure confirmed — crop failure risk in 2–4 weeks |

The key physical insight: **SWIR bands (B11, B12) detect stomatal closure before NDVI detects canopy damage.** GWSat explicitly models this — NDVI threshold methods cannot.

---

## 🏆 Hackathon Achievement

GWSat was built for the **TM2Space Hackathon**, targeting early drought warning for smallholder farmers in Telangana using real satellite data and IBM TerraMind's geospatial foundation model.

**Result: 🥈 2nd Place**

---

## 📊 Results

All numbers are from real held-out Sentinel-2 patches (never seen during training), evaluated on confirmed real TerraMind (`backend: terratorch`).

### Overall Performance

| Method | F1 Macro | Accuracy | Notes |
|:-------|:--------:|:--------:|:------|
| NDVI Threshold *(baseline)* | 0.4756 | 0.5345 | 0% recall on Moderate class |
| **GWSat [terratorch]** | **0.8602** | **0.8448** | TerraMind + SWIR physics fusion |

**Δ vs NDVI baseline: +0.3846 F1 Macro**

### Per-Class Breakdown

| Class | n | Accuracy | Precision | Recall |
|:------|:-:|:--------:|:---------:|:------:|
| Stable | 22 | 0.8182 | 0.7826 | 0.8182 |
| Moderate | 21 | 0.7619 | 0.8000 | 0.7619 |
| **Critical** | **15** | **1.0000** | **1.0000** | **1.0000** |

> 🚨 **Critical class: zero misses.** The most dangerous groundwater state is never overlooked. The NDVI baseline gets 0% recall on Moderate — classifying every at-risk field as stable. That's exactly the failure mode GWSat fixes.

### Efficiency Metrics

| Metric | Value |
|:-------|:-----:|
| Head checkpoint | 273 KB |
| TerraMind encoder | ~13 MB |
| INT8 quantized ONNX | ~3–5 MB |
| CPU latency (64×64 patch) | **3.8 ms** |
| Alert payload | **64 bytes** |
| Raw tile size | 131 KB |
| Compression ratio | **2,048×** |

---

## 🏗️ Architecture

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

### Key Design Decisions

- **Frozen encoder:** TerraMind weights are frozen — only the fusion head trains. This prevents overfitting on the small (~12 acquisition) dataset while preserving TerraMind's geospatial pre-training.
- **Raw-band physics:** Spectral scalars are extracted from raw bands *before* attention weighting — they measure true physical signal, not attention-modified signals.
- **Tiled scene inference:** Scene-level inference uses a sliding 64×64 window (stride 32) with a conservative alert rule — if ≥15% of vegetated patches are Critical, the scene is escalated regardless of majority vote.

### Spectral Physics Indices

Six physics-derived indices are appended to the TerraMind embedding before classification:

| Index | Formula | Physical Meaning |
|:------|:--------|:----------------|
| NDVI | (B8−B4)/(B8+B4) | Greenness — drops late, fooled by irrigation |
| RedEdge Chl | (B8A/B5)−1 | Chlorophyll — drops early under stress |
| LSWI | (B8−B11)/(B8+B11) | Direct leaf water content |
| NDWI | (B8A−B11)/(B8A+B11) | Open water signal |
| SWIR Ratio | B11/B12 | ABA proxy — stomatal closure signal |
| CWC | (B8−B12)/(B8+B12) | Canopy water content |

> **SWIR Ratio and LSWI are the primary discriminators between Moderate and Critical** — these are the physics that NDVI misses entirely.

---

## ⚡ Quick Start

```bash
# 1. Clone and install
git clone https://github.com/LeafyChan/Blue-Green-Drought.git
cd Blue-Green-Drought
pip install -r requirements.txt

# 2. Install real TerraMind (strongly recommended)
pip install git+https://github.com/IBM/terratorch.git
# ✅ You want to see: "TerraMind loaded via terratorch (embed_dim=192)"
# ⚠️  If you see "DeiT-tiny PROXY" — fix the install before running evals.

# 3. Run demo on synthetic tiles (no data files needed)
python infer.py --demo

# 4. Or launch the interactive Gradio demo
python demo_app.py
# → Open http://localhost:7860
```

---

## 🔁 Full Pipeline

### Step 0 — Install Dependencies

```bash
pip install -r requirements.txt

# TerraMind — required for real benchmark results
pip install git+https://github.com/IBM/terratorch.git
# OR: pip install terratorch
```

> If you see `⚠️ TerraMind unavailable — using DeiT-tiny PROXY`, your results are from DeiT-tiny, not TerraMind. The `backend` field in every output file will tell you exactly what loaded.

---

### Step 1 — Build Dataset

**From synthetic tiles (no data required):**
```bash
python build_real_dataset.py
```

**From real Telangana GeoTIFF files:**
```bash
# Split multi-band or per-band TIFs into processed/
python split_bands.py --input_dir raw/

# Convert to .pt tensors
python tif_to_pt.py --folder ./Telangana_Instant_Data --out scene.pt
```

This produces `data/train.pt`, `data/val.pt`, and `data/test.pt`. The test split uses completely different seeds from training — it is a true holdout.

---

### Step 2 — Train

**Standard training (cross-entropy + AdamW):**
```bash
python train.py --epochs 60 --batch_size 32 --lr 3e-4
```

**If Moderate recall is 0% (common with imbalanced data), use the fixed trainer:**
```bash
python train_moderate_fix.py --epochs 80 --moderate_weight 5.0 --focal_gamma 2.0
# Uses: Focal loss + Mixup augmentation + cosine LR schedule + Moderate weight ×5
```

Only `PhysicsFusionHead` and `SpectralAttention` weights are trained. The TerraMind encoder stays frozen throughout.

Training will print the active backend clearly:
```
*** ACTIVE BACKEND: terratorch ***    ← real TerraMind
*** ACTIVE BACKEND: timm_proxy ***   ← DeiT-tiny proxy
```

---

### Step 3 — Validate

```bash
# Evaluate on held-out test split
python validate.py

# With additional out-of-distribution TIF scene folders
# (folder path must contain 'stable', 'moderate', or 'critical' to auto-infer label)
python validate.py --scene_dirs path/to/extra_stable path/to/extra_critical

# Custom checkpoint
python validate.py --checkpoint checkpoints/best_head.pth
```

Output is written to `validation_results.json`.

---

### Step 4 — Inspect Metrics

```bash
# View all results
python validate.py
cat validation_results.json
cat results_moderate_fixed.json

# Model size
ls -lh checkpoints/best_head.pth
ls -lh checkpoints/gwsat_terratorch.onnx 2>/dev/null || echo "not exported yet"

# CPU latency benchmark
python3 -c "
import torch, time
from model import GWSatModel
model = GWSatModel(device='cpu')
model.load_head('checkpoints/best_head.pth')
model.eval()
x = torch.randn(1, 8, 64, 64)
for _ in range(3): model(x)   # warm-up
t0 = time.perf_counter()
for _ in range(50):
    with torch.no_grad(): model(x)
ms = (time.perf_counter() - t0) / 50 * 1000
print(f'CPU latency: {ms:.1f} ms per patch')
"

# F1 vs NDVI delta summary
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

---

### Step 5 — Export to ONNX

```bash
# Full-precision ONNX (filename includes backend name automatically)
python export.py --verify

# INT8 quantized — recommended for Jetson deployment
python export.py --quantize --verify

# TensorRT engine (compile on Jetson)
trtexec --onnx=checkpoints/gwsat_terratorch.onnx \
        --saveEngine=checkpoints/gwsat.engine \
        --int8 --workspace=256
```

**Output files:**
| File | Description |
|:-----|:-----------|
| `checkpoints/gwsat_terratorch.onnx` | Full-precision, real TerraMind backend |
| `checkpoints/gwsat_timm-proxy.onnx` | Proxy backend, clearly labelled |
| `checkpoints/gwsat_terratorch.meta.json` | Export metadata |

---

### Step 6 — Run the Demo

```bash
# Local demo
python demo_app.py

# Shareable public URL (Gradio tunnelling)
python demo_app.py --share
```

The demo accepts synthetic sample tiles, a folder of TIFs, or a `.pt` tensor file. Output includes backend name, stress class, confidence score, spectral index values, and a spatial heatmap.

---

## 🔍 Inference Modes

```bash
# Synthetic tiles — no data required
python infer.py --demo

# Full Sentinel-2 TIF scene (tiled, no resize aliasing)
python infer.py --scene B4.tif B5.tif B6.tif B7.tif B8.tif B8A.tif B11.tif B12.tif

# Single .pt tile
python infer.py --tile sample_input/sample_class2.pt

# GEE batch tensor
python infer.py --tensor data/inference/telangana_demo.pt
```

> Scene mode tiles into 64×64 patches rather than resizing — this preserves the SWIR stress signal that gets destroyed by global averaging.

**Minimal ONNX inference (for deployment environments):**

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

## 📦 Input / Output Format

**Input:** 8-band Sentinel-2 L2A patch — shape `[8, 64, 64]`, band order `[B4, B5, B6, B7, B8, B8A, B11, B12]`, reflectance values `[0.0, 1.0]` (divide raw DN by 10000).

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

## 🔬 Backend Transparency

GWSat is fully transparent about which encoder loaded. Every results file includes the backend explicitly — there is no silent fallback.

| `backend_name` | Meaning |
|:--------------|:--------|
| `terratorch` | ✅ Real IBM TerraMind via terratorch package |
| `hf_transformers` | ✅ Real IBM TerraMind via HuggingFace |
| `timm_proxy` | ⚠️ DeiT-tiny proxy — NOT TerraMind |
| `edge_cnn` | ❌ CNN fallback — all encoder strategies failed |

```python
from model import GWSatModel
model = GWSatModel(device="cpu")
print(model.backend_name)   # → 'terratorch'
```

Results with `timm_proxy` will look similar on synthetic data (the head learns physics features regardless of backbone) but will underperform on real out-of-distribution imagery where TerraMind's geospatial pre-training matters.

---

## 🛰️ Edge Deployment

GWSat is designed for on-orbit inference on the Jetson Orin NX and MNR CubeSat.

| Metric | Value |
|:-------|:-----:|
| Head checkpoint | 273 KB |
| TerraMind-tiny encoder | ~13 MB |
| INT8 quantized ONNX | ~3–5 MB |
| CPU latency (64×64 patch) | 3.8 ms |
| Estimated Jetson latency (INT8) | ~15–30 ms |
| RAM footprint (inference only) | < 256 MB |
| Alert output payload | 64 bytes |

The TensorRT engine is pre-compiled on-ground and flashed alongside weights. On-orbit inference requires only the ONNX runtime (< 30 MB).

### On-Orbit Continual Learning

`ocl.py` implements memory-bounded continual learning for CubeSat deployment:

- The **production model** serves all inference (frozen).
- A **shadow head** trains on labelled corrections received from ground.
- At orbit boundaries, `MissionSupervisor` gates the swap: requires F1 improvement > 0.02 and no single class dropping > 0.10.
- A `RollbackBank` stores the last 3 checkpoints for safe reversion.

Memory budget: ~18 MB for a 150-tile correction buffer.  
> ⚠️ Not yet tested on physical Jetson hardware.

---

## ⚠️ Honest Limitations

- **Small training set.** 12 Sentinel-2 acquisitions (5 Stable, 2 Moderate, 5 Critical). The Stable/Moderate boundary is partly seasonal. `train_moderate_fix.py` addresses this with Focal loss and Mixup augmentation.
- **No per-patch borewell ground-truth.** Labels are derived from CGWB district records and expert scene annotation — not per-patch measured groundwater depth. The next step is co-registering 50+ patches with open CGWB borewell records.
- **OCL not validated on hardware.** Power and thermal constraints on-orbit may require further tuning of the update budget.
- **ONNX export on terratorch** may require opset≥17 depending on terratorch version. Use `python export.py --verify` to confirm the exported graph runs correctly.

---

## 📁 File Reference

| File | Purpose |
|:-----|:--------|
| `model.py` | Core architecture — TerraMind encoder + Physics Fusion Head |
| `train.py` | Standard training loop (head only, encoder frozen) |
| `train_weighted.py` | Weighted CE to address Moderate class imbalance |
| `train_moderate_fix.py` | Focal loss + Mixup — fixes 0% Moderate recall |
| `validate.py` | Holdout validation + NDVI baseline comparison |
| `infer.py` | All inference modes: demo, scene, tile, tensor |
| `inference.py` | Minimal ONNX runtime example |
| `data_pipeline.py` | Synthetic Sentinel-2 tile generator |
| `build_real_dataset.py` | Builds train/val/test `.pt` tensors from raw data |
| `tif_to_pt.py` | Converts GeoTIFF to PyTorch tensors (auto band detection) |
| `split_bands.py` | Splits multi-band or per-band TIF files into `processed/` |
| `calculate_ndvi.py` | Patch-level spectral analysis and NDVI baseline reporting |
| `demo_app.py` | Gradio web demo with heatmap and spectral charts |
| `export.py` | ONNX export with backend-aware wrapper + INT8 quantization |
| `ocl.py` | On-orbit continual learning (shadow training, rollback bank) |
| `visualize.py` | Heatmap and false-colour composite utilities |
| `run.py` | Convenience entry point for common workflows |
| `requirements.txt` | All dependencies with minimum versions |
| `RUN_GUIDE.md` | Detailed fix log and troubleshooting guide |

---

## 📦 Dependencies

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

<div align="center">

*Team **NP Complete** · Built for the TM2Space Hackathon*  
*Targeting Telangana groundwater stress early-warning using IBM TerraMind + SWIR physics fusion*

</div>
