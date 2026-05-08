"""
validate.py — GWSat v3  (REAL DATA ONLY)
-----------------------------------------
Evaluates the trained model against real Sentinel-2 data only.
No synthetic tiles are generated or used anywhere in this script.

What this tests:
  1. Held-out test split (data/test.pt) — produced by build_real_dataset.py,
     never seen during training (different patches from different scenes).
  2. Additional real scenes passed via --scene_dirs (TIF folders not in
     your processed/ training data — true out-of-distribution test).
  3. NDVI-only baseline on the SAME patches for an honest delta.

Usage:
    # Fast: just evaluate on data/test.pt
    python validate.py

    # With additional out-of-distribution TIF scene folders
    python validate.py --scene_dirs path/to/extra_stable path/to/extra_critical

    # Custom checkpoint
    python validate.py --checkpoint checkpoints/best_head.pth

Output: validation_results.json
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

LABEL_NAMES = ["Stable", "Moderate", "Critical"]
LABEL_MAP = {"stable": 0, "moderate": 1, "critical": 2}


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import (
        f1_score, accuracy_score, confusion_matrix, classification_report,
        precision_score, recall_score,
    )

    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        mask = y_true == i
        if mask.sum() == 0:
            per_class[name] = {"n": 0, "acc": None, "precision": None, "recall": None}
            continue
        per_class[name] = {
            "n":         int(mask.sum()),
            "acc":       round(float((y_pred[mask] == i).mean()), 4),
            "precision": round(float(precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_pred,    labels=[i], average="macro", zero_division=0)), 4),
        }

    report = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, labels=[0, 1, 2], zero_division=0
    )
    return {
        "f1_macro":              round(float(f1), 4),
        "accuracy":              round(float(acc), 4),
        "per_class":             per_class,
        "confusion_matrix":      cm.tolist(),
        "classification_report": report,
    }


# ── NDVI baseline ─────────────────────────────────────────────────────────

def ndvi_baseline_preds(X: torch.Tensor) -> np.ndarray:
    """
    Pure NDVI threshold classifier.
    Thresholds match semi-arid Telangana literature (Reddy et al. 2021).
    This is what NDMA's traditional system sees.
    """
    b4   = X[:, 0].mean(dim=(1, 2))
    b8   = X[:, 4].mean(dim=(1, 2))
    ndvi = ((b8 - b4) / (b8 + b4 + 1e-8)).numpy()
    return np.where(ndvi > 0.45, 0, np.where(ndvi > 0.25, 1, 2))


# ── Model batch inference ─────────────────────────────────────────────────

def run_model_on(model, X: torch.Tensor, device: str,
                  batch_size: int = 32) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        with torch.no_grad():
            logits = model(xb)
        preds.extend(logits.argmax(1).cpu().numpy())
    return np.array(preds)


# ── Evaluate test.pt ──────────────────────────────────────────────────────

def evaluate_test_split(model, device: str) -> dict:
    test_path = Path("data/test.pt")
    if not test_path.exists():
        print("  ⚠️  data/test.pt not found. Run build_real_dataset.py first.")
        return {}

    data = torch.load(str(test_path), map_location="cpu", weights_only=False)
    X = data["X"].float()
    y = data["y"].numpy()

    counts = np.bincount(y, minlength=3)
    print(f"  test.pt: {len(X)} patches  "
          f"S:{counts[0]} M:{counts[1]} C:{counts[2]}")

    t0 = time.time()
    gwsat_preds = run_model_on(model, X, device)
    elapsed = time.time() - t0

    ndvi_preds  = ndvi_baseline_preds(X)
    gwsat_m     = compute_metrics(y, gwsat_preds)
    ndvi_m      = compute_metrics(y, ndvi_preds)

    print(f"\n  GWSat  F1={gwsat_m['f1_macro']:.4f}  Acc={gwsat_m['accuracy']:.4f}  ({elapsed:.1f}s)")
    print(f"  NDVI   F1={ndvi_m['f1_macro']:.4f}  Acc={ndvi_m['accuracy']:.4f}")
    print(f"  Δ F1 (GWSat - NDVI): {gwsat_m['f1_macro'] - ndvi_m['f1_macro']:+.4f}")
    print()
    print(gwsat_m["classification_report"])

    print("  Per-class breakdown:")
    for name, stats in gwsat_m["per_class"].items():
        if stats["n"] == 0:
            continue
        print(f"    {name:<12}  n={stats['n']:>4}  acc={stats['acc']:.4f}  "
              f"prec={stats['precision']:.4f}  rec={stats['recall']:.4f}")

    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    header = "             " + "".join(f"{l:>10}" for l in LABEL_NAMES)
    print(header)
    for i, row in enumerate(gwsat_m["confusion_matrix"]):
        print(f"  {LABEL_NAMES[i]:<12}" + "".join(f"{v:>10}" for v in row))

    return {
        "n_patches": len(X),
        "class_counts": counts.tolist(),
        "gwsat": {k: v for k, v in gwsat_m.items() if k != "classification_report"},
        "ndvi":  {k: v for k, v in ndvi_m.items()  if k != "classification_report"},
        "delta_f1_vs_ndvi": round(gwsat_m["f1_macro"] - ndvi_m["f1_macro"], 4),
    }


# ── Evaluate extra TIF scene folders ─────────────────────────────────────

def evaluate_scene_folder(model, folder: str, device: str) -> dict:
    """
    Load a raw TIF scene folder (not in training data), tile it,
    infer the label from the folder path (stable/moderate/critical),
    and run GWSat + NDVI baseline.
    """
    from build_real_dataset import detect_bands, load_band, BAND_ORDER
    try:
        import rasterio
    except ImportError:
        print("  rasterio not installed"); return {}

    # Infer label from path
    folder_lower = folder.lower()
    true_label = -1
    for key, lbl in LABEL_MAP.items():
        if key in folder_lower:
            true_label = lbl
            break

    if true_label == -1:
        print(f"  ⚠️  Cannot infer label from path: {folder}")
        print(f"     Path must contain 'stable', 'moderate', or 'critical'.")
        return {}

    print(f"\n  Scene: {folder}")
    print(f"  Ground truth: {LABEL_NAMES[true_label]}")

    band_paths = detect_bands(folder)
    if "B4" not in band_paths or "B8" not in band_paths:
        print(f"  ⚠️  Missing B4 or B8 — cannot tile this scene")
        return {}

    # Load bands
    b4_arr = load_band(band_paths["B4"])
    ref_shape = b4_arr.shape
    H, W = ref_shape

    arrays = {"B4": b4_arr}
    for band in BAND_ORDER:
        if band == "B4":
            continue
        arrays[band] = load_band(band_paths[band], ref_shape) if band in band_paths else np.zeros(ref_shape, dtype=np.float32)

    # Tile
    patch_size, stride = 64, 64
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = np.stack([arrays[b][y:y+patch_size, x:x+patch_size] for b in BAND_ORDER], axis=0)
            b4m = patch[0].mean(); b8m = patch[4].mean()
            ndvi = (b8m - b4m) / (b8m + b4m + 1e-8)
            if ndvi >= 0.08:
                patches.append(patch)

    if not patches:
        print(f"  ⚠️  No vegetated patches found in {folder}")
        return {}

    X = torch.tensor(np.stack(patches), dtype=torch.float32)
    y_true = np.full(len(X), true_label, dtype=np.int64)

    gwsat_preds = run_model_on(model, X, device)
    ndvi_preds  = ndvi_baseline_preds(X)
    gwsat_m     = compute_metrics(y_true, gwsat_preds)
    ndvi_m      = compute_metrics(y_true, ndvi_preds)

    # Scene-level verdict (conservative escalation rule)
    counts_arr = np.bincount(gwsat_preds, minlength=3)
    n_total = len(gwsat_preds)
    frac_crit = counts_arr[2] / n_total
    frac_mod  = counts_arr[1] / n_total
    raw_winner = int(np.argmax(counts_arr))
    if frac_crit >= 0.15 and raw_winner < 2:
        scene_cls = 2
    elif frac_mod >= 0.25 and raw_winner == 0:
        scene_cls = 1
    else:
        scene_cls = raw_winner

    correct = scene_cls == true_label
    mark = "✅" if correct else "❌"

    print(f"  {len(X)} vegetated patches")
    print(f"  Patch dist: S:{counts_arr[0]} M:{counts_arr[1]} C:{counts_arr[2]}")
    print(f"  Scene verdict: {mark} {LABEL_NAMES[scene_cls]}  (GT: {LABEL_NAMES[true_label]})")
    print(f"  GWSat patch  F1={gwsat_m['f1_macro']:.4f}  Acc={gwsat_m['accuracy']:.4f}")
    print(f"  NDVI  patch  F1={ndvi_m['f1_macro']:.4f}  Acc={ndvi_m['accuracy']:.4f}")
    print(f"  Δ F1: {gwsat_m['f1_macro'] - ndvi_m['f1_macro']:+.4f}")

    return {
        "folder": folder,
        "ground_truth": LABEL_NAMES[true_label],
        "scene_verdict": LABEL_NAMES[scene_cls],
        "correct": correct,
        "n_patches": int(len(X)),
        "patch_dist": {"Stable": int(counts_arr[0]), "Moderate": int(counts_arr[1]), "Critical": int(counts_arr[2])},
        "gwsat_patch_f1":  round(float(gwsat_m["f1_macro"]), 4),
        "gwsat_patch_acc": round(float(gwsat_m["accuracy"]),  4),
        "ndvi_patch_f1":   round(float(ndvi_m["f1_macro"]),  4),
        "ndvi_patch_acc":  round(float(ndvi_m["accuracy"]),  4),
        "delta_f1_vs_ndvi": round(gwsat_m["f1_macro"] - ndvi_m["f1_macro"], 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*64}")
    print("GWSat — Real Data Validation")
    print(f"{'='*64}")
    print(f"Device:     {device}")

    from model import GWSatModel
    model = GWSatModel(device=device, terramind_model=args.terramind_model)

    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        model.load_head(str(ckpt))
        print(f"Checkpoint: {ckpt}")
    else:
        print(f"⚠️  No checkpoint at {ckpt}. Validating with random weights.")

    print(f"Backend:    {model.backend_name}")
    if model.backend_name == "timm_proxy":
        print("  ⚠️  WARNING: DeiT-tiny proxy — NOT real TerraMind.")
        print("     Install terratorch for real results.")

    results = {
        "backend":    model.backend_name,
        "checkpoint": str(ckpt),
    }

    # ── Test split (from build_real_dataset.py) ───────────────────────────
    print(f"\n── Held-out test split (data/test.pt) ─────────────────────────")
    test_results = evaluate_test_split(model, device)
    if test_results:
        results["test_split"] = test_results

    # ── Extra scene folders ────────────────────────────────────────────────
    if args.scene_dirs:
        print(f"\n── Out-of-distribution scene folders ──────────────────────────")
        scene_results = []
        for folder in args.scene_dirs:
            r = evaluate_scene_folder(model, folder, device)
            if r:
                scene_results.append(r)

        if scene_results:
            n_correct = sum(1 for r in scene_results if r["correct"])
            print(f"\n  Scene-level: {n_correct}/{len(scene_results)} correct "
                  f"({100*n_correct/len(scene_results):.0f}%)")
            results["extra_scenes"] = scene_results

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = Path("validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*64}")
    print("SUMMARY")
    print(f"{'='*64}")
    print(f"  Backend:     {model.backend_name}")
    if test_results:
        gf = test_results["gwsat"]["f1_macro"]
        nf = test_results["ndvi"]["f1_macro"]
        ga = test_results["gwsat"]["accuracy"]
        print(f"  GWSat F1:    {gf:.4f}  Acc={ga:.4f}  (held-out test split)")
        print(f"  NDVI  F1:    {nf:.4f}")
        print(f"  Δ vs NDVI:   {gf-nf:+.4f}")
    if args.scene_dirs and "extra_scenes" in results:
        n = len(results["extra_scenes"])
        nc = sum(1 for r in results["extra_scenes"] if r["correct"])
        print(f"  OOD scenes:  {nc}/{n} correct scene-level predictions")
    print(f"\n  Saved: {out_path}")
    print(f"{'='*64}\n")

    if model.backend_name == "timm_proxy":
        print("⚠️  Results used DeiT-tiny proxy, NOT TerraMind.")
        print("   Run: pip install terratorch && retrain for real numbers.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GWSat real-data validation")
    p.add_argument("--checkpoint",      default="checkpoints/best_head.pth")
    p.add_argument("--terramind_model", default="ibm-esa-geospatial/TerraMind-1.0-tiny")
    p.add_argument(
        "--scene_dirs", nargs="*", default=[],
        help=(
            "Optional extra TIF scene folders NOT used in training. "
            "Folder path must contain 'stable', 'moderate', or 'critical' "
            "to auto-infer the ground-truth label."
        ),
    )
    main(p.parse_args())