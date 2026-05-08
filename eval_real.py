"""
eval_real.py — GWSat v3
------------------------
Evaluate the ALREADY-TRAINED model against your real Sentinel-2 TIF files
in gee_raw/. These scenes were NEVER used in training (training used only
synthetic data), so this is your true out-of-distribution test.

Each TIF in gee_raw/ is a multi-band file. This script:
  1. Reads each TIF (multi-band or single-band, auto-detected)
  2. Tiles into 64×64 patches
  3. Runs the trained model on every patch
  4. Reports per-scene and aggregate F1/Acc vs ground-truth labels
     derived from the filename (critical/moderate/stable)

Usage:
    python eval_real.py                                  # uses gee_raw/ by default
    python eval_real.py --tif_dir /path/to/gee_raw      # custom folder
    python eval_real.py --checkpoint checkpoints/best_head.pth
    python eval_real.py --stride 32                     # denser tiling
    python eval_real.py --save_csv real_eval_results.csv
"""

import argparse, sys, json, time, warnings
import numpy as np
import torch
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))


LABEL_NAMES = ["Stable", "Moderate", "Critical"]

# Filename → ground truth label mapping
# Matches any filename containing these substrings (case-insensitive)
LABEL_MAP = {
    "critical": 2,
    "moderate": 1,
    "stable":   0,
}


def infer_label_from_filename(fname: str) -> int:
    """
    Returns ground-truth label from TIF filename.
    critical_2022.tiff → 2
    stable_nizam_2023.tiff → 0
    Returns -1 if unknown.
    """
    fname_lower = fname.lower()
    for key, lbl in LABEL_MAP.items():
        if key in fname_lower:
            return lbl
    return -1


def load_multiband_tif(path: str) -> tuple:
    """
    Load a GEE-exported multi-band TIF.
    GEE typically exports all bands in a single TIF with band descriptions
    like 'B4', 'B8', etc.

    Returns (tensor [8, H, W], band_names_found)
    Missing bands are filled with zeros and flagged.
    """
    try:
        import rasterio
    except ImportError:
        print("ERROR: pip install rasterio"); sys.exit(1)

    BAND_ORDER = ["B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    with rasterio.open(path) as src:
        n_bands = src.count
        # Try to read band descriptions (GEE sets these)
        desc = [src.descriptions[i] or f"band_{i+1}"
                for i in range(n_bands)]

        # Read all bands
        data = src.read().astype(np.float32)  # [n_bands, H, W]
        H, W = src.height, src.width

    # Normalize: GEE S2 exports are often in [0, 10000]
    if data.max() > 2.0:
        data = data / 10000.0
    data = np.clip(data, 0.0, 1.0)

    # Map available bands to GWSat's expected [B4,B5,B6,B7,B8,B8A,B11,B12] order
    # Strategy 1: match by band description string
    band_arrays = {}
    for i, d in enumerate(desc):
        d_upper = d.upper().strip()
        # Handle GEE naming: 'B4', 'B04', 'SR_B4', etc.
        for bname in BAND_ORDER:
            variants = [bname, bname.replace("B", "B0") if len(bname)==2 else bname,
                        f"SR_{bname}", f"SR_B{bname[1:]}"]
            if any(v.upper() in d_upper or d_upper == v.upper() for v in variants):
                if bname not in band_arrays:
                    band_arrays[bname] = data[i]

    # Strategy 2: if descriptions not set, infer from band count
    # GEE S2 SR exports typically: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 (10 bands)
    # or just the 8 we need
    if not band_arrays:
        if n_bands == 8:
            # Assume exactly our 8 bands in order
            for i, bname in enumerate(BAND_ORDER):
                band_arrays[bname] = data[i]
        elif n_bands == 10:
            # GEE S2 SR 10-band: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
            gee_10 = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
            for i, bname in enumerate(gee_10):
                if bname in BAND_ORDER:
                    band_arrays[bname] = data[i]
        elif n_bands == 12:
            # GEE S2 SR 12-band: B1..B8,B8A,B9,B11,B12
            gee_12 = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
            for i, bname in enumerate(gee_12):
                if bname in BAND_ORDER:
                    band_arrays[bname] = data[i]
        else:
            # Last resort: take first 8 bands in order
            print(f"    ⚠️  Unknown band count ({n_bands}), mapping first 8 to GWSat order")
            for i, bname in enumerate(BAND_ORDER[:min(n_bands, 8)]):
                band_arrays[bname] = data[i]

    # Build output tensor — missing bands → zeros
    out = np.zeros((8, H, W), dtype=np.float32)
    found = []
    for i, bname in enumerate(BAND_ORDER):
        if bname in band_arrays:
            out[i] = band_arrays[bname]
            found.append(bname)

    return torch.from_numpy(out), found


def tile_and_filter(scene: torch.Tensor, patch_size: int = 64,
                    stride: int = 64, min_veg: float = 0.10) -> torch.Tensor:
    """
    Tile scene [8,H,W] into [N,8,64,64] patches.
    Filters patches with NDVI < min_veg (bare soil, water, cloud).
    """
    C, H, W = scene.shape
    patches = []

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            p = scene[:, y:y+patch_size, x:x+patch_size]
            b4 = p[0]; b8 = p[4]
            ndvi = ((b8 - b4) / (b8 + b4 + 1e-8)).mean().item()
            if ndvi >= min_veg:
                patches.append(p)

    if not patches:
        return torch.zeros(0, C, patch_size, patch_size)
    return torch.stack(patches)


def run_model_on_patches(model, patches: torch.Tensor,
                          device: str, batch_size: int = 32) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(patches), batch_size):
        xb = patches[i:i+batch_size].to(device)
        with torch.no_grad():
            logits = model(xb)
        preds.extend(logits.argmax(1).cpu().numpy())
    return np.array(preds)


def scene_verdict(preds: np.ndarray) -> tuple:
    """Majority vote + confidence (fraction of patches voting for winner)."""
    if len(preds) == 0:
        return -1, 0.0
    counts = np.bincount(preds, minlength=3)
    cls    = int(counts.argmax())
    conf   = counts[cls] / len(preds)
    return cls, conf


def print_bar(val: float, width: int = 30) -> str:
    return "█" * int(val * width) + "░" * (width - int(val * width))


def main(args):
    from sklearn.metrics import f1_score, classification_report, confusion_matrix

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*66}")
    print("GWSat — Real TIF Evaluation (out-of-distribution test)")
    print(f"{'='*66}")
    print(f"TIF dir:    {args.tif_dir}")
    print(f"Device:     {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    from model import GWSatModel
    model = GWSatModel(device=device)
    ckpt  = Path(args.checkpoint)
    if ckpt.exists():
        model.load_head(str(ckpt))
        print(f"Checkpoint: {ckpt}  (backend={model.backend_name})")
    else:
        print(f"⚠️  No checkpoint at {ckpt}. Using random weights.")
    model.eval()

    # ── Find TIF files ────────────────────────────────────────────────────────
    tif_dir  = Path(args.tif_dir)
    tif_files = sorted(tif_dir.glob("*.tif")) + sorted(tif_dir.glob("*.tiff"))
    if not tif_files:
        print(f"ERROR: No TIF files found in {tif_dir}"); sys.exit(1)

    print(f"\nFound {len(tif_files)} TIF files:\n")

    # ── Per-scene evaluation ──────────────────────────────────────────────────
    scene_results = []
    all_preds     = []
    all_labels    = []

    for tif_path in tif_files:
        true_label = infer_label_from_filename(tif_path.name)
        if true_label == -1:
            print(f"  ⚠️  {tif_path.name}: cannot infer label — skipping")
            continue

        print(f"  {'─'*62}")
        print(f"  📁 {tif_path.name}")
        print(f"     Ground truth: {LABEL_NAMES[true_label]}")

        t0 = time.time()
        try:
            scene, bands_found = load_multiband_tif(str(tif_path))
        except Exception as e:
            print(f"     ❌ Load failed: {e}"); continue

        C, H, W = scene.shape
        print(f"     Scene:        {C} bands × {H}×{W} px  "
              f"≈{H*10/1000:.1f}×{W*10/1000:.1f} km")
        print(f"     Bands found:  {bands_found}")

        if "B4" not in bands_found or "B8" not in bands_found:
            print(f"     ❌ Missing B4 or B8 — cannot compute NDVI for tiling"); continue
        if "B11" not in bands_found:
            print(f"     ⚠️  B11 (SWIR1) missing — LSWI physics will be zero")

        patches = tile_and_filter(scene, patch_size=64,
                                   stride=args.stride, min_veg=args.min_ndvi)
        n_patches = len(patches)
        total_tiles = ((H - 64) // args.stride + 1) * ((W - 64) // args.stride + 1)
        skipped = max(0, total_tiles - n_patches)
        print(f"     Patches:      {n_patches} vegetated / {total_tiles} total "
              f"({skipped} bare-soil/water skipped)")

        if n_patches == 0:
            print(f"     ⚠️  No vegetated patches — try --min_ndvi 0.0"); continue

        preds = run_model_on_patches(model, patches, device)
        pred_cls, pred_conf = scene_verdict(preds)

        patch_dist = {LABEL_NAMES[i]: int((preds==i).sum()) for i in range(3)}
        elapsed    = time.time() - t0

        # Physics baseline (NDVI on patches)
        b4  = patches[:, 0].mean(dim=(1, 2))
        b8  = patches[:, 4].mean(dim=(1, 2))
        ndvi_vals = ((b8 - b4) / (b8 + b4 + 1e-8)).numpy()
        ndvi_preds = np.where(ndvi_vals > 0.5, 0, np.where(ndvi_vals > 0.35, 1, 2))
        ndvi_verdict, _ = scene_verdict(ndvi_preds)

        correct_model = (pred_cls == true_label)
        correct_ndvi  = (ndvi_verdict == true_label)

        mark = "✅" if correct_model else "❌"
        ndvi_mark = "✅" if correct_ndvi else "❌"
        print(f"     Prediction:   {mark} {LABEL_NAMES[pred_cls]}  "
              f"(conf={pred_conf:.1%}, {elapsed:.1f}s)")
        print(f"     NDVI base:    {ndvi_mark} {LABEL_NAMES[ndvi_verdict]}")
        print(f"     Patch dist:   "
              f"Stable:{patch_dist['Stable']} "
              f"Moderate:{patch_dist['Moderate']} "
              f"Critical:{patch_dist['Critical']}")

        # Patch-level metrics (each patch labelled with scene ground truth)
        patch_labels = np.full(n_patches, true_label)
        patch_f1  = f1_score(patch_labels, preds, average="macro", zero_division=0)
        patch_acc = (preds == true_label).mean()
        print(f"     Patch F1:     {patch_f1:.4f}  "
              f"Patch Acc (all→true): {patch_acc:.4f}")

        scene_results.append({
            "file":         tif_path.name,
            "true_label":   LABEL_NAMES[true_label],
            "pred_label":   LABEL_NAMES[pred_cls],
            "correct":      correct_model,
            "confidence":   round(float(pred_conf), 4),
            "n_patches":    n_patches,
            "patch_dist":   patch_dist,
            "patch_f1":     round(float(patch_f1), 4),
            "patch_acc":    round(float(patch_acc), 4),
            "ndvi_verdict": LABEL_NAMES[ndvi_verdict],
            "ndvi_correct": correct_ndvi,
            "elapsed_s":    round(elapsed, 2),
            "bands_found":  bands_found,
        })

        all_preds.extend(preds.tolist())
        all_labels.extend([true_label] * n_patches)

    # ── Aggregate results ─────────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print("AGGREGATE RESULTS (all real scenes)")
    print(f"{'='*66}")

    if not scene_results:
        print("No scenes evaluated."); return

    # Scene-level summary
    n_scenes  = len(scene_results)
    n_correct = sum(1 for r in scene_results if r["correct"])
    n_correct_ndvi = sum(1 for r in scene_results if r["ndvi_correct"])

    print(f"\nScene-level accuracy:")
    print(f"  GWSat:  {n_correct}/{n_scenes} correct "
          f"({100*n_correct/n_scenes:.0f}%)")
    print(f"  NDVI:   {n_correct_ndvi}/{n_scenes} correct "
          f"({100*n_correct_ndvi/n_scenes:.0f}%)")

    print(f"\nPer-scene breakdown:")
    print(f"  {'File':<32} {'True':<12} {'Pred':<12} {'Conf':>6} {'Patches':>8}")
    print(f"  {'─'*70}")
    for r in scene_results:
        mark = "✅" if r["correct"] else "❌"
        print(f"  {mark} {r['file']:<30} {r['true_label']:<12} "
              f"{r['pred_label']:<12} {r['confidence']:>5.1%} {r['n_patches']:>8}")

    # Patch-level aggregate
    if all_labels:
        all_preds_np  = np.array(all_preds)
        all_labels_np = np.array(all_labels)

        agg_f1  = f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
        agg_acc = (all_preds_np == all_labels_np).mean()
        cm      = confusion_matrix(all_labels_np, all_preds_np, labels=[0,1,2])

        # NDVI aggregate
        b4_all  = torch.tensor(all_preds_np)   # we don't have raw band values here
        # Recompute NDVI baseline patch predictions across all scenes
        ndvi_all_preds = []
        ndvi_all_labels = []
        for r in scene_results:
            tif_path = tif_dir / r["file"]
            scene, _ = load_multiband_tif(str(tif_path))
            patches = tile_and_filter(scene, 64, args.stride, args.min_ndvi)
            if len(patches) == 0: continue
            b4 = patches[:, 0].mean(dim=(1,2))
            b8 = patches[:, 4].mean(dim=(1,2))
            ndvi_v = ((b8 - b4)/(b8 + b4 + 1e-8)).numpy()
            ndvi_p = np.where(ndvi_v > 0.5, 0, np.where(ndvi_v > 0.35, 1, 2))
            true_lbl = LABEL_MAP[r["true_label"].lower()]
            ndvi_all_preds.extend(ndvi_p.tolist())
            ndvi_all_labels.extend([true_lbl] * len(ndvi_p))

        ndvi_f1  = f1_score(ndvi_all_labels, ndvi_all_preds, average="macro", zero_division=0)
        ndvi_acc = (np.array(ndvi_all_preds) == np.array(ndvi_all_labels)).mean()

        print(f"\nPatch-level aggregate ({len(all_labels):,} patches total):")
        print(f"  {'Method':<30} {'F1 (macro)':>12} {'Accuracy':>10}")
        print(f"  {'─'*54}")
        print(f"  {'NDVI threshold':<30} {ndvi_f1:>12.4f} {ndvi_acc:>10.4f}")
        print(f"  {'GWSat ['+model.backend_name+']':<30} {agg_f1:>12.4f} {agg_acc:>10.4f}")
        print(f"  {'Δ (GWSat - NDVI)':<30} {agg_f1-ndvi_f1:>+12.4f} {agg_acc-ndvi_acc:>+10.4f}")

        print(f"\nClassification report (patch level):")
        print(classification_report(all_labels_np, all_preds_np,
              target_names=LABEL_NAMES, zero_division=0))

        print(f"Confusion matrix (rows=true, cols=pred):")
        print(f"             {'Stable':>8} {'Mod':>8} {'Crit':>8}")
        for i, row in enumerate(cm.tolist()):
            print(f"  {LABEL_NAMES[i]:<12}" + "".join(f"{v:>8}" for v in row))

        # Save JSON results
        out = {
            "backend":     model.backend_name,
            "checkpoint":  str(ckpt),
            "tif_dir":     str(tif_dir),
            "n_scenes":    n_scenes,
            "scene_accuracy": {
                "gwsat": round(n_correct/n_scenes, 4),
                "ndvi":  round(n_correct_ndvi/n_scenes, 4),
            },
            "patch_aggregate": {
                "n_patches":  len(all_labels),
                "gwsat_f1":   round(float(agg_f1), 4),
                "gwsat_acc":  round(float(agg_acc), 4),
                "ndvi_f1":    round(float(ndvi_f1), 4),
                "ndvi_acc":   round(float(ndvi_acc), 4),
                "delta_f1":   round(float(agg_f1-ndvi_f1), 4),
                "confusion_matrix": cm.tolist(),
            },
            "per_scene": scene_results,
        }

        with open("real_eval_results.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n✅ Full results saved: real_eval_results.json")

        if args.save_csv:
            import csv
            with open(args.save_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=scene_results[0].keys())
                w.writeheader(); w.writerows(scene_results)
            print(f"✅ CSV saved: {args.save_csv}")

    print(f"\n{'='*66}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate GWSat on real GEE TIF files (no retraining needed)")
    p.add_argument("--tif_dir",    default="gee_raw",
                   help="Folder containing .tif/.tiff files")
    p.add_argument("--checkpoint", default="checkpoints/best_head.pth")
    p.add_argument("--stride",     type=int, default=64,
                   help="Patch stride in pixels (64=no overlap, 32=50%%)")
    p.add_argument("--min_ndvi",   type=float, default=0.05,
                   help="Minimum patch NDVI to include (0.0 = include all)")
    p.add_argument("--save_csv",   default=None,
                   help="Optional: save per-scene CSV")
    main(p.parse_args())