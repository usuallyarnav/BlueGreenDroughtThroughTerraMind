"""
build_real_dataset.py — GWSat v3  (REAL DATA ONLY)
----------------------------------------------------
Builds train/val/test .pt files from your real Sentinel-2 scenes.

Expected folder layout:
    processed/
        stable/
            scene_name/          ← one subfolder per scene
                *B4*.tif
                *B5*.tif
                *B8*.tif
                ...
        moderate/
            scene_name/
                *.tif
        critical/
            scene_name/
                *.tif

Each scene subfolder must contain individual band TIF files.
Band files are detected by name pattern (B4/B04, B8, B8A, B11, B12, etc.).
The folder name (stable / moderate / critical) is used as the ground-truth
class label. LSWI is computed per-patch and used to verify/filter
physically implausible patches (e.g. cloud, water) but the folder label wins.

Usage:
    python build_real_dataset.py
    python build_real_dataset.py --processed_dir /path/to/processed
    python build_real_dataset.py --stride 32 --min_ndvi 0.08
    python build_real_dataset.py --preview           # stats only, no save
    python build_real_dataset.py --no_balance        # keep raw class counts
"""

import argparse
import glob
import os
import re
import sys
import numpy as np
import torch
from pathlib import Path

try:
    import rasterio
    from rasterio.enums import Resampling
except ImportError:
    print("ERROR: rasterio not installed. Run: pip install rasterio")
    sys.exit(1)

# ── Band order GWSat expects ───────────────────────────────────────────────
BAND_ORDER = ["B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Detection priority: B8A before B8 to avoid false matches
BAND_PATTERNS = {
    "B8A": [r"B8A", r"B08A"],
    "B4":  [r"_B04[_.]", r"_B4[_.]",  r"\bB04\b", r"\bB4\b"],
    "B5":  [r"_B05[_.]", r"_B5[_.]",  r"\bB05\b", r"\bB5\b"],
    "B6":  [r"_B06[_.]", r"_B6[_.]",  r"\bB06\b", r"\bB6\b"],
    "B7":  [r"_B07[_.]", r"_B7[_.]",  r"\bB07\b", r"\bB7\b"],
    "B8":  [r"_B08[_.]", r"_B8[_.]",  r"\bB08\b", r"(?<![A8])B8(?!A)\b"],
    "B11": [r"_B11[_.]", r"\bB11\b"],
    "B12": [r"_B12[_.]", r"\bB12\b"],
}

LABEL_MAP = {"stable": 0, "moderate": 1, "critical": 2}
LABEL_NAMES = ["Stable", "Moderate", "Critical"]


def infer_label_from_name(name: str) -> int:
    """
    Infer stress label from a folder name by substring match.
    Works for both:
      - Flat layout:  may2020critical, nov2021stable, sept2023moderate
      - Nested layout: processed/critical/may2020/, processed/stable/nov2021/
    Returns -1 if no match.
    """
    name_lower = name.lower()
    for key, lbl in LABEL_MAP.items():
        if key in name_lower:
            return lbl
    return -1


# ── Band auto-detection ────────────────────────────────────────────────────

def detect_bands(folder: str) -> dict:
    """
    Scan folder for TIF files and match to the 8 required bands.
    Returns {band_name: path}. Missing optional bands are omitted;
    missing B4/B8 raises an error (NDVI/LSWI cannot be computed).
    """
    tif_files = (
        glob.glob(os.path.join(folder, "*.tif"))
        + glob.glob(os.path.join(folder, "*.TIF"))
        + glob.glob(os.path.join(folder, "*.tiff"))
    )
    if not tif_files:
        return {}

    detected, used = {}, set()
    priority = ["B8A", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]

    for band in priority:
        patterns = BAND_PATTERNS.get(band, [])
        for tif in tif_files:
            if tif in used:
                continue
            fname = os.path.basename(tif)
            for pat in patterns:
                if re.search(pat, fname, re.IGNORECASE):
                    detected[band] = tif
                    used.add(tif)
                    break
            if band in detected:
                break

    return detected


# ── Single-band TIF loader ─────────────────────────────────────────────────

def load_band(path: str, ref_shape=None) -> np.ndarray:
    """
    Load one band TIF as float32 in [0, 1].
    Resamples to ref_shape if needed (handles 20m B11/B12 vs 10m B4/B8).
    """
    with rasterio.open(path) as src:
        if ref_shape and (src.height, src.width) != ref_shape:
            arr = src.read(
                1,
                out_shape=(1, ref_shape[0], ref_shape[1]),
                resampling=Resampling.bilinear,
            ).astype(np.float32)[0]
        else:
            arr = src.read(1).astype(np.float32)

    if arr.max() > 2.0:
        arr /= 10000.0
    return np.clip(arr, 0.0, 1.0)


# ── Scene → patch extraction ───────────────────────────────────────────────

def scene_to_patches(
    folder: str,
    folder_label: int,
    patch_size: int = 64,
    stride: int = 64,
    min_ndvi: float = 0.08,
    verbose: bool = True,
) -> tuple:
    """
    Load all bands from a scene folder, tile into patches, filter
    and label each patch.

    Label assignment (conservative):
      - Folder label is ground truth (set by the researcher who placed
        the scene in stable/ moderate/ critical/).
      - Patches with NDVI < min_ndvi are skipped (cloud, bare soil, water).
      - LSWI is only used to skip obviously invalid patches
        (LSWI < -0.7 → likely cloud/water body), NOT to override the label.

    Returns (patches: list of np.ndarray [8,64,64], labels: list of int)
    """
    band_paths = detect_bands(folder)

    if "B4" not in band_paths:
        if verbose:
            print(f"    SKIP (no B4): {folder}")
        return [], []
    if "B8" not in band_paths:
        if verbose:
            print(f"    SKIP (no B8): {folder}")
        return [], []

    # Load B4 first to get reference shape
    b4_arr = load_band(band_paths["B4"])
    ref_shape = b4_arr.shape
    H, W = ref_shape

    arrays = {"B4": b4_arr}
    for band in BAND_ORDER:
        if band == "B4":
            continue
        if band in band_paths:
            arrays[band] = load_band(band_paths[band], ref_shape)
        else:
            arrays[band] = np.zeros(ref_shape, dtype=np.float32)

    patches, labels = [], []
    n_skipped_veg = 0
    n_skipped_cloud = 0

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = np.stack(
                [arrays[b][y : y + patch_size, x : x + patch_size] for b in BAND_ORDER],
                axis=0,
            )  # [8, patch_size, patch_size]

            b4_mean = patch[0].mean()
            b8_mean = patch[4].mean()
            b11_mean = patch[6].mean()

            ndvi = (b8_mean - b4_mean) / (b8_mean + b4_mean + 1e-8)
            lswi = (b8_mean - b11_mean) / (b8_mean + b11_mean + 1e-8)

            # Skip bare soil / cloud / water
            if ndvi < min_ndvi:
                n_skipped_veg += 1
                continue

            # Skip obviously invalid (cloud shadow, water body → extreme LSWI)
            if lswi < -0.70:
                n_skipped_cloud += 1
                continue

            patches.append(patch)
            labels.append(folder_label)

    if verbose:
        total = ((H - patch_size) // stride + 1) * ((W - patch_size) // stride + 1)
        kept = len(patches)
        skipped = n_skipped_veg + n_skipped_cloud
        print(
            f"    {Path(folder).name:<40}  "
            f"{kept:>4} patches kept  "
            f"({skipped} skipped: {n_skipped_veg} non-veg, "
            f"{n_skipped_cloud} cloud/water)  "
            f"label={LABEL_NAMES[folder_label]}"
        )

    return patches, labels


# ── Band profile stats ─────────────────────────────────────────────────────

def print_band_stats(X: np.ndarray, y: np.ndarray):
    print("\n── Real Band Profile Stats ────────────────────────────────────")
    print(f"  {'Class':<12} {'B4':>6} {'B8':>6} {'B11':>6} {'NDVI':>7} {'LSWI':>7} {'N':>6}")
    print("  " + "─" * 56)
    for cls, name in enumerate(LABEL_NAMES):
        mask = y == cls
        if mask.sum() == 0:
            print(f"  {name:<12}  NO DATA")
            continue
        Xc = X[mask]
        b4 = Xc[:, 0].mean()
        b8 = Xc[:, 4].mean()
        b11 = Xc[:, 6].mean()
        ndvi = (b8 - b4) / (b8 + b4 + 1e-8)
        lswi = (b8 - b11) / (b8 + b11 + 1e-8)
        print(
            f"  {name:<12} {b4:>6.3f} {b8:>6.3f} {b11:>6.3f} "
            f"{ndvi:>+7.3f} {lswi:>+7.3f} {mask.sum():>6}"
        )
    print()
    print("  Physical sanity check:")
    print("    Stable   → LSWI most positive  (B8 >> B11, leaf water present)")
    print("    Moderate → LSWI near zero       (partial stomatal closure)")
    print("    Critical → LSWI most negative  (B11 > B8, severe deficit)")
    print()


# ── Main ───────────────────────────────────────────────────────────────────

def collect_scenes(processed_dir: Path) -> list:
    """
    Collect (scene_dir, label) pairs from processed_dir.

    Supports two layouts automatically:

    Layout A — Flat (your actual structure):
        processed/
            may2020critical/   ← label inferred from folder name
            nov2021stable/
            sept2023moderate/

    Layout B — Nested (label folders contain scene subfolders):
        processed/
            critical/
                may2020/
            stable/
                nov2021/

    Both layouts can coexist in the same processed/ folder.
    """
    scenes = []

    for item in sorted(processed_dir.iterdir()):
        if not item.is_dir():
            continue

        # Check if THIS folder name contains a label keyword (Layout A)
        label = infer_label_from_name(item.name)
        if label != -1:
            scenes.append((item, label))
            continue

        # Check if it's a class-name folder containing scene subfolders (Layout B)
        if item.name.lower() in LABEL_MAP:
            class_label = LABEL_MAP[item.name.lower()]
            sub_scenes = sorted([d for d in item.iterdir() if d.is_dir()])
            if sub_scenes:
                for sub in sub_scenes:
                    scenes.append((sub, class_label))
            else:
                # TIF files directly inside the class folder
                scenes.append((item, class_label))

    return scenes


def main(args):
    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        print(f"ERROR: Processed directory not found: {processed_dir}")
        print("\nExpected layout A (flat — your structure):")
        print("  processed/")
        print("    may2020critical/   ← label from folder name")
        print("    nov2021stable/")
        print("    sept2023moderate/")
        print("\nExpected layout B (nested):")
        print("  processed/")
        print("    critical/  stable/  moderate/")
        sys.exit(1)

    Path("data").mkdir(exist_ok=True)
    all_patches, all_labels = [], []

    print(f"\n── Loading scenes from: {processed_dir} ───────────────────────────")

    scenes = collect_scenes(processed_dir)
    if not scenes:
        print(f"ERROR: No labelled scene folders found in {processed_dir}")
        print("Folder names must contain 'stable', 'moderate', or 'critical'.")
        sys.exit(1)

    # Group by label for display
    by_label = {0: [], 1: [], 2: []}
    for scene_dir, label in scenes:
        by_label[label].append(scene_dir)

    for cls, name in enumerate(LABEL_NAMES):
        print(f"\n  [{name.upper()}]  {len(by_label[cls])} scene(s):")
        for scene_dir in by_label[cls]:
            p, l = scene_to_patches(
                str(scene_dir),
                cls,
                patch_size=args.patch_size,
                stride=args.stride,
                min_ndvi=args.min_ndvi,
                verbose=True,
            )
            all_patches.extend(p)
            all_labels.extend(l)
        class_total = sum(1 for lbl in all_labels if lbl == cls)
        print(f"    → {class_total} total {name} patches so far")

    if not all_patches:
        print("\nERROR: No patches extracted.")
        print("Check that:")
        print("  1. processed/ contains stable/ moderate/ critical/ subfolders")
        print("  2. Each subfolder contains scene subdirs with TIF band files")
        print("  3. TIF filenames contain band identifiers like B4, B8, B11, etc.")
        sys.exit(1)

    X = torch.tensor(np.stack(all_patches), dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.long)

    counts = y.bincount(minlength=3)
    print(f"\n── Raw totals ──────────────────────────────────────────────────")
    print(f"  Total patches : {len(X)}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:<12}: {counts[i].item()}")

    print_band_stats(X.numpy(), y.numpy())

    if args.preview:
        print("Preview mode — nothing saved.")
        return

    if counts.min() == 0:
        print("ERROR: At least one class has zero patches.")
        print("  Check your processed/ folder structure and TIF band naming.")
        sys.exit(1)

    # ── Class balancing ────────────────────────────────────────────────────
    if not args.no_balance:
        min_count = counts.min().item()
        cap = min_count  # hard cap at minority class count
        print(f"── Balancing classes (cap per class: {cap}) ────────────────────")

        balanced_X, balanced_y = [], []
        for cls in range(3):
            mask = (y == cls).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(mask))[:cap]
            sel = mask[perm]
            balanced_X.append(X[sel])
            balanced_y.append(y[sel])
            print(f"  {LABEL_NAMES[cls]:<12}: {len(mask)} → {len(sel)}")

        X = torch.cat(balanced_X)
        y = torch.cat(balanced_y)
    else:
        print("── Skipping balancing (--no_balance set) ───────────────────────")

    # Shuffle
    idx = torch.randperm(len(X))
    X, y = X[idx], y[idx]

    # Warn if any class is very small
    final_counts = y.bincount(minlength=3)
    if final_counts.min() < 50:
        print(f"\n  WARNING: smallest class has only {final_counts.min().item()} patches.")
        print("  Consider collecting more data or using --no_balance.")

    # ── Split 70 / 15 / 15 ────────────────────────────────────────────────
    n = len(X)
    t = int(0.70 * n)
    v = int(0.85 * n)

    torch.save({"X": X[:t],  "y": y[:t]},  "data/train.pt")
    torch.save({"X": X[t:v], "y": y[t:v]}, "data/val.pt")
    torch.save({"X": X[v:],  "y": y[v:]},  "data/test.pt")

    print(f"\n── Saved ───────────────────────────────────────────────────────")
    print(f"  data/train.pt  {t} patches  {y[:t].bincount(minlength=3).tolist()}")
    print(f"  data/val.pt    {v-t} patches  {y[t:v].bincount(minlength=3).tolist()}")
    print(f"  data/test.pt   {n-v} patches  {y[v:].bincount(minlength=3).tolist()}")
    print(f"\n  [Stable, Moderate, Critical]")
    print(f"\nNext step:  python train_moderate_fix.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build GWSat dataset from real Sentinel-2 TIF scenes"
    )
    p.add_argument(
        "--processed_dir", default="processed",
        help="Root folder containing stable/ moderate/ critical/ subfolders (default: processed/)",
    )
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride",     type=int, default=64,
                   help="Patch stride in pixels. 64=no overlap, 32=50%% overlap (more patches)")
    p.add_argument("--min_ndvi",   type=float, default=0.08,
                   help="Minimum patch NDVI to keep (filters bare soil, clouds, water)")
    p.add_argument("--preview",    action="store_true",
                   help="Print stats only, do not save .pt files")
    p.add_argument("--no_balance", action="store_true",
                   help="Skip class balancing (keep raw counts)")
    main(p.parse_args())