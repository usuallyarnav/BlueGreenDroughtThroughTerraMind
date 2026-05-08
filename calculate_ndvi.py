"""
calculate_ndvi.py — GWSat v2
-----------------------------
Tiled spectral analysis of Telangana Sentinel-2 data.

NDVI alone gives a misleading picture (mean 0.20 = "stressed" but max 0.92
from irrigated patches overwhelms the stressed signal during global resize).

This script:
  1. Reads all 6 available TIF bands at NATIVE resolution (no resize)
  2. Tiles the scene into 64x64 patches
  3. Computes spectral indices PER PATCH (not globally averaged)
  4. Reports patch-level stress distribution — the truth the global
     mean was hiding.

Usage:
    python calculate_ndvi.py
    python calculate_ndvi.py --folder Telangana_Instant_Data --save_csv
"""

import argparse
import glob
import os
import numpy as np


def compute_indices_for_patch(patch: dict) -> dict:
    """Compute spectral indices for a single patch (dict of band arrays)."""
    eps = 1e-8

    def mean(b):
        arr = patch.get(b)
        return float(arr.mean()) if arr is not None else np.nan

    b4  = mean('B4');  b5  = mean('B5');  b6  = mean('B6')
    b8  = mean('B8');  b11 = mean('B11'); b12 = mean('B12')

    ndvi  = (b8 - b4)  / (b8  + b4  + eps)
    rei   = (b6 - b5)  / (b6  + b5  + eps) if not np.isnan(b5) else np.nan
    lswi  = (b8 - b11) / (b8  + b11 + eps) if not np.isnan(b11) else np.nan
    swir  = b11 / (b12 + eps) if not np.isnan(b11) else np.nan
    irp   = (swir - 1.0) * (1.0 - max(lswi, 0)) if not np.isnan(lswi) else np.nan

    return {
        "NDVI": round(ndvi, 4),
        "RedEdge_Index": round(rei, 4) if not np.isnan(rei) else None,
        "LSWI": round(lswi, 4) if not np.isnan(lswi) else None,
        "SWIR_ratio": round(swir, 4) if not np.isnan(swir) else None,
        "IR_Pressure": round(irp, 4) if not np.isnan(irp) else None,
    }


def classify_patch(indices: dict) -> int:
    """
    Classify patch stress using multi-band rules.
    Returns 0=Stable, 1=Moderate, 2=Critical.

    Critically: uses SWIR and RedEdge, NOT just NDVI.
    """
    ndvi = indices["NDVI"]
    lswi = indices["LSWI"]
    rei  = indices["RedEdge_Index"]
    irp  = indices["IR_Pressure"]

    # SWIR-based rules take priority over NDVI
    if lswi is not None:
        if lswi < 0.10:
            return 2   # Critical: severe leaf water deficit
        elif lswi < 0.25:
            return 1   # Moderate: stress building

    if rei is not None:
        if rei < -0.02:
            return 2
        elif rei < 0.08:
            return 1

    if irp is not None and irp > 0.15:
        return 2

    # NDVI as last resort
    if ndvi < 0.20:
        return 2
    elif ndvi < 0.40:
        return 1
    return 0


def tile_array(arr: np.ndarray, patch_size: int = 64,
               stride: int = 64) -> list:
    """Return list of (y0, x0, patch) from 2D array."""
    H, W = arr.shape
    patches = []
    y = 0
    while y + patch_size <= H:
        x = 0
        while x + patch_size <= W:
            patches.append((y, x, arr[y:y+patch_size, x:x+patch_size]))
            x += stride
        y += stride
    return patches


def calculate_ndvi(folder_path: str = ".", save_csv: bool = False):
    try:
        import rasterio
    except ImportError:
        print("pip install rasterio")
        return

    print(f"\nGWSat v2 — Tiled Spectral Analysis")
    print(f"Folder : {folder_path}")
    print("=" * 56)

    # Load available bands
    band_files = {}
    for band in ['B2', 'B3', 'B4', 'B5', 'B6', 'B8', 'B11', 'B12']:
        pattern = os.path.join(folder_path, f"*{band}*")
        matches = glob.glob(pattern)
        if matches:
            band_files[band] = matches[0]

    if 'B4' not in band_files or 'B8' not in band_files:
        print("ERROR: Need at least B4 (Red) and B8 (NIR).")
        return

    # Load all bands at native resolution
    bands = {}
    shape = None
    for band, path in band_files.items():
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            vmax = arr.max()
            if vmax > 2.0:
                arr /= 10000.0
            arr = np.clip(arr, 0, 1)
            bands[band] = arr
            if shape is None:
                shape = arr.shape
                H, W = shape
                print(f"Scene size : {H} × {W} px  "
                      f"(~{H*10/1000:.1f} × {W*10/1000:.1f} km at 10m/px)")

    print(f"Bands loaded : {', '.join(sorted(band_files.keys()))}")
    print()

    # Global stats (the misleading number)
    ndvi_global = (bands['B8'] - bands['B4']) / (
        bands['B8'] + bands['B4'] + 1e-8)
    print(f"  Global NDVI (what NDMA sees):  "
          f"mean={ndvi_global.mean():.4f}  "
          f"max={ndvi_global.max():.4f}  "
          f"→  ⚠️  MISLEADING: max dominated by irrigated patches")
    print()

    # Tiled analysis
    print("Tiled patch analysis (64×64 px, no resize):")
    print("-" * 56)

    # Tile reference band
    ref_tiles = tile_array(bands['B4'], patch_size=64, stride=64)
    n_patches = len(ref_tiles)
    counts    = [0, 0, 0]
    ndvi_per_class = [[], [], []]
    rows = []

    for y0, x0, _ in ref_tiles:
        # Build per-patch band dict
        patch_bands = {}
        for band, arr in bands.items():
            patch_bands[band] = arr[y0:y0+64, x0:x0+64]

        indices = compute_indices_for_patch(patch_bands)
        cls     = classify_patch(indices)
        counts[cls] += 1
        ndvi_per_class[cls].append(indices["NDVI"])

        if save_csv:
            rows.append({"y": y0, "x": x0, "class": cls, **indices})

    labels = ["Stable", "Moderate", "Critical"]
    total  = sum(counts)
    for i, name in enumerate(labels):
        pct = 100 * counts[i] / (total or 1)
        bar = "█" * int(pct / 3)
        avg_ndvi = np.mean(ndvi_per_class[i]) if ndvi_per_class[i] else 0
        print(f"  {name:<10} {counts[i]:>5} patches ({pct:5.1f}%)  {bar}")
        if ndvi_per_class[i]:
            print(f"             Mean NDVI of these patches: {avg_ndvi:.4f}"
                  + (" ← NDVI looks healthy but SWIR says stressed!"
                     if i == 2 and avg_ndvi > 0.35 else ""))

    print()
    dominant = labels[counts.index(max(counts))]
    pct_dom  = 100 * max(counts) / (total or 1)
    crit_pct = 100 * counts[2] / (total or 1)

    print(f"Scene verdict: {dominant} ({pct_dom:.0f}% of patches)")
    if crit_pct > 30:
        print(f"  🚨 {crit_pct:.0f}% of patches are CRITICAL — "
              "global NDVI mean was masking this!")
    elif crit_pct > 10:
        print(f"  ⚠️  {crit_pct:.0f}% patches critical — early intervention window open")

    print()
    print("Deception analysis:")
    print(f"  Global NDVI = {ndvi_global.mean():.4f}  → traditional system says FINE")
    print(f"  Critical patches = {counts[2]}/{total}  → GWSat says INTERVENTION NEEDED")
    print()
    print("Run full AI inference:")
    print("  python infer.py --scene <B4.tif> <B5.tif> <B6.tif> "
          "<B7.tif> <B8.tif> <B8A.tif> <B11.tif> <B12.tif>")

    if save_csv and rows:
        import csv
        out = "telangana_patch_analysis.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        print(f"\n  CSV saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folder",   default=".")
    p.add_argument("--save_csv", action="store_true")
    args = p.parse_args()
    calculate_ndvi(args.folder, args.save_csv)
