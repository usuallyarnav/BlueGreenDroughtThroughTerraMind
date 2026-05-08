"""
tif_to_pt.py — GWSat v3
------------------------
Converts Sentinel-2 TIF files into .pt tensors ready for GWSat inference
and training. Handles the real-world situation where you have multiple
TIF files (one per band) from the same scene/date/location.

Band order GWSat expects: [B4, B5, B6, B7, B8, B8A, B11, B12]
                           [ 0,  1,  2,  3,  4,   5,   6,   7]

Usage examples:
    # Auto-detect bands in a folder and convert to a single .pt
    python tif_to_pt.py --folder ./Telangana_Instant_Data --out scene.pt

    # Specify band files explicitly (if auto-detect fails)
    python tif_to_pt.py \
        --b4  Telangana_Instant_Data_B4.tif \
        --b5  Telangana_Instant_Data_B5.tif \
        --b6  Telangana_Instant_Data_B6.tif \
        --b7  Telangana_Instant_Data_B7.tif \
        --b8  Telangana_Instant_Data_B8.tif \
        --b8a Telangana_Instant_Data_B8A.tif \
        --b11 Telangana_Instant_Data_B11.tif \
        --b12 Telangana_Instant_Data_B12.tif \
        --out scene.pt

    # Convert + immediately tile into 64x64 patches (for training dataset)
    python tif_to_pt.py --folder ./Telangana_Instant_Data --out scene.pt --tile

    # Convert multiple scene folders at once (batch mode)
    python tif_to_pt.py --batch_folders scene1/ scene2/ scene3/ --out_dir data/scenes/

    # Preview what was loaded (no save)
    python tif_to_pt.py --folder ./Telangana_Instant_Data --preview
"""

import argparse
import glob
import os
import sys
import re
from pathlib import Path

import numpy as np
import torch


# ─── Band detection patterns ──────────────────────────────────────────────────
# These patterns match common naming conventions from:
#   - Copernicus Hub direct download  (e.g. T44NMH_20230101T050901_B04.tif)
#   - GEE exports                     (e.g. Telangana_Instant_Data_B4.tif)
#   - SNAP exports                    (e.g. S2A_MSIL2A_B8A.tif)
#   - Simple naming                   (e.g. B4.tif, B04.tif)

BAND_PATTERNS = {
    # Band key → list of regex patterns that identify it in a filename
    "B4":  [r"_B04[_.]", r"_B4[_.]",  r"\bB04\b", r"\bB4\b"],
    "B5":  [r"_B05[_.]", r"_B5[_.]",  r"\bB05\b", r"\bB5\b"],
    "B6":  [r"_B06[_.]", r"_B6[_.]",  r"\bB06\b", r"\bB6\b"],
    "B7":  [r"_B07[_.]", r"_B7[_.]",  r"\bB07\b", r"\bB7\b"],
    "B8":  [r"_B08[_.]", r"_B8[_.]",  r"\bB08\b", r"(?<![A8])B8(?!A)\b"],
    "B8A": [r"_B8A[_.]", r"\bB8A\b",  r"_B08A[_.]"],
    "B11": [r"_B11[_.]", r"\bB11\b"],
    "B12": [r"_B12[_.]", r"\bB12\b"],
}

BAND_ORDER = ["B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def auto_detect_bands(folder: str) -> dict:
    """
    Scan a folder for TIF files and match them to the 8 required bands.
    Returns dict like {"B4": "path/to/B4.tif", "B8": "path/to/B8.tif", ...}

    Priority: longer, more specific matches win over shorter ones.
    B8A is checked before B8 to avoid B8A files matching the B8 pattern.
    """
    tif_files = (glob.glob(os.path.join(folder, "*.tif")) +
                 glob.glob(os.path.join(folder, "*.TIF")) +
                 glob.glob(os.path.join(folder, "*.tiff")))

    if not tif_files:
        raise FileNotFoundError(f"No TIF files found in: {folder}")

    detected = {}
    used_files = set()

    # Check B8A before B8 to prevent B8A files matching B8 patterns
    priority_order = ["B8A", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]

    for band in priority_order:
        patterns = BAND_PATTERNS[band]
        for tif in tif_files:
            if tif in used_files:
                continue
            fname = os.path.basename(tif)
            for pattern in patterns:
                if re.search(pattern, fname, re.IGNORECASE):
                    detected[band] = tif
                    used_files.add(tif)
                    break
            if band in detected:
                break

    return detected


def load_band_tif(path: str, reference_shape: tuple = None) -> np.ndarray:
    """
    Load a single-band TIF as float32 array in [0, 1].

    Handles:
    - Multi-band TIFs (takes band 1, ignores rest)
    - Integer reflectance (0–10000 range → divide by 10000)
    - Float reflectance (0.0–1.0 range → kept as-is)
    - Optional resampling to reference_shape if bands have different resolutions
      (e.g. B11/B12 are 20m, B8 is 10m — rare but possible in mixed exports)
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
    except ImportError:
        print("ERROR: rasterio not installed. Run: pip install rasterio")
        sys.exit(1)

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)

        # If reference shape given and this band differs, resample
        if reference_shape is not None and arr.shape != reference_shape:
            arr_resampled = src.read(
                1,
                out_shape=(1, reference_shape[0], reference_shape[1]),
                resampling=Resampling.bilinear
            ).astype(np.float32)[0]
            arr = arr_resampled
            print(f"    ↳ Resampled from {src.height}×{src.width} "
                  f"→ {reference_shape[0]}×{reference_shape[1]}")

    # Normalize reflectance to [0, 1]
    # Sentinel-2 L2A surface reflectance is stored as uint16 in [0, 10000]
    # Some exports already normalise to float [0.0, 1.0]
    vmax = arr.max()
    if vmax > 2.0:
        arr = arr / 10000.0

    arr = np.clip(arr, 0.0, 1.0)
    return arr


def tifs_to_tensor(band_paths: dict,
                   verbose: bool = True) -> torch.Tensor:
    """
    Given a dict of {band_name: tif_path}, load all bands and stack into
    [8, H, W] float32 tensor in the order GWSat expects:
    [B4, B5, B6, B7, B8, B8A, B11, B12]

    Missing optional bands (B5, B6, B7, B8A) are filled with zeros.
    B4 and B8 are mandatory — raises if either is missing.
    """
    if "B4" not in band_paths:
        raise ValueError("B4 (Red) is required but not found.")
    if "B8" not in band_paths:
        raise ValueError("B8 (NIR) is required but not found.")

    # Use B4 as the reference shape (10m band, native Sentinel-2 resolution)
    ref_arr  = load_band_tif(band_paths["B4"])
    ref_shape = ref_arr.shape
    H, W = ref_shape
    if verbose:
        print(f"  Reference shape (B4): {H} × {W} px  "
              f"≈ {H*10/1000:.1f} × {W*10/1000:.1f} km at 10m/px")

    bands_loaded = []
    for band in BAND_ORDER:
        if band == "B4":
            arr = ref_arr          # already loaded
        elif band in band_paths:
            if verbose:
                print(f"  Loading {band}: {os.path.basename(band_paths[band])}")
            arr = load_band_tif(band_paths[band], reference_shape=ref_shape)
        else:
            if verbose:
                status = "REQUIRED — using zeros (NDVI/SWIR will be wrong!)" \
                    if band in ("B8",) else f"missing — filling with zeros"
                print(f"  ⚠️  {band}: {status}")
            arr = np.zeros(ref_shape, dtype=np.float32)
        bands_loaded.append(arr)

    stacked = np.stack(bands_loaded, axis=0)     # [8, H, W]
    tensor  = torch.from_numpy(stacked)          # float32
    return tensor


def tile_scene(scene: torch.Tensor,
               patch_size: int = 64,
               stride: int = 64,
               min_ndvi: float = 0.05) -> tuple:
    """
    Cut a full scene [8, H, W] into overlapping 64×64 patches.
    Filters out bare-soil / water patches (NDVI too low).
    Returns (X, positions) where X is [N, 8, 64, 64].
    """
    C, H, W = scene.shape
    patches, positions = [], []

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = scene[:, y:y+patch_size, x:x+patch_size]
            # Quick NDVI check: skip non-vegetation patches
            b4 = patch[0]; b8 = patch[4]
            ndvi = ((b8 - b4) / (b8 + b4 + 1e-8)).mean().item()
            if ndvi >= min_ndvi:
                patches.append(patch)
                positions.append((y, x))

    if not patches:
        print("  ⚠️  No patches passed the NDVI filter. "
              "Lowering min_ndvi or checking band assignment.")
        return torch.zeros(0, C, patch_size, patch_size), []

    X = torch.stack(patches)
    return X, positions


def print_scene_stats(tensor: torch.Tensor, band_paths: dict):
    """Print a quick spectral sanity check on the loaded scene."""
    print("\n  Spectral sanity check:")
    print(f"  {'Band':<6} {'Min':>8} {'Mean':>8} {'Max':>8}  {'Loaded from'}")
    print("  " + "-" * 70)
    for i, band in enumerate(BAND_ORDER):
        arr = tensor[i]
        src = os.path.basename(band_paths.get(band, "zeros"))
        print(f"  {band:<6} {arr.min():>8.4f} {arr.mean():>8.4f} "
              f"{arr.max():>8.4f}  {src}")

    # Quick NDVI
    b4 = tensor[0]; b8 = tensor[4]
    ndvi = ((b8 - b4) / (b8 + b4 + 1e-8))
    print(f"\n  NDVI  min={ndvi.min():.4f}  mean={ndvi.mean():.4f}  "
          f"max={ndvi.max():.4f}")

    # LSWI (needs B11)
    b11 = tensor[6]
    if b11.max() > 0:
        lswi = ((b8 - b11) / (b8 + b11 + 1e-8))
        print(f"  LSWI  min={lswi.min():.4f}  mean={lswi.mean():.4f}  "
              f"max={lswi.max():.4f}  (leaf water)")
    else:
        print("  LSWI  ⚠️  B11 is all zeros — cannot compute leaf water stress")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Convert Sentinel-2 TIF bands → GWSat .pt tensor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--folder",       help="Folder containing TIF files (auto-detects bands)")
    mode.add_argument("--batch_folders",nargs="+", metavar="FOLDER",
                      help="Multiple scene folders (batch mode)")

    # Explicit band paths (override auto-detect)
    for band in ["b4", "b5", "b6", "b7", "b8", "b8a", "b11", "b12"]:
        p.add_argument(f"--{band}", metavar="PATH", help=f"Path to {band.upper()} TIF")

    p.add_argument("--out",      default="scene.pt",
                   help="Output .pt file (single scene mode)")
    p.add_argument("--out_dir",  default="data/scenes",
                   help="Output directory (batch mode)")
    p.add_argument("--tile",     action="store_true",
                   help="Also tile scene into 64×64 patches and save as patches.pt")
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride",     type=int, default=64,
                   help="Stride for tiling (64=no overlap, 32=50%% overlap)")
    p.add_argument("--min_ndvi",   type=float, default=0.05,
                   help="Minimum NDVI to keep a patch (filters bare soil/water)")
    p.add_argument("--preview",  action="store_true",
                   help="Print stats only, do not save")
    p.add_argument("--label",    type=int, default=None, choices=[0, 1, 2],
                   help="Optional ground-truth stress label (0=Stable 1=Moderate 2=Critical)")
    return p


def process_single(band_paths: dict, args, out_path: str):
    """Load bands, optionally tile, save to out_path."""
    print(f"\n{'='*60}")
    print(f"Converting → {out_path}")
    print(f"{'='*60}")
    print(f"  Bands detected/specified:")
    for band in BAND_ORDER:
        status = os.path.basename(band_paths[band]) if band in band_paths else "⚠️  MISSING"
        print(f"    {band:<5} {status}")

    scene = tifs_to_tensor(band_paths, verbose=True)
    print_scene_stats(scene, band_paths)

    if args.preview:
        print("\n  [Preview mode — not saved]")
        return

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "X":            scene,            # [8, H, W] full scene
        "bands":        BAND_ORDER,
        "band_files":   {k: str(v) for k, v in band_paths.items()},
        "scene_shape":  list(scene.shape),
        "gwsat_version": 3,
    }
    if args.label is not None:
        save_dict["label"] = args.label

    torch.save(save_dict, out_path)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"\n  ✅ Scene saved: {out_path}  ({size_mb:.2f} MB)")
    print(f"     Load in Python:  data = torch.load('{out_path}')")
    print(f"                      scene = data['X']  # shape: {list(scene.shape)}")
    print(f"\n  Run inference:")
    print(f"     python infer.py --scene {' '.join(band_paths.get(b, 'MISSING') for b in BAND_ORDER)}")
    print(f"  OR (after saving this .pt):")
    print(f"     python infer.py --tile {out_path}")

    # Tile mode
    if args.tile:
        print(f"\n  Tiling into {args.patch_size}×{args.patch_size} patches "
              f"(stride={args.stride})…")
        X_patches, positions = tile_scene(
            scene,
            patch_size=args.patch_size,
            stride=args.stride,
            min_ndvi=args.min_ndvi
        )
        n = len(X_patches)
        print(f"  Patches extracted: {n}")

        if n > 0:
            patches_path = out_path.replace(".pt", "_patches.pt")
            patch_dict = {
                "X":         X_patches,          # [N, 8, 64, 64]
                "positions": positions,           # list of (y, x) tuples
                "bands":     BAND_ORDER,
                "source":    out_path,
            }
            if args.label is not None:
                # Replicate label for every patch
                patch_dict["y"] = torch.full((n,), args.label, dtype=torch.long)

            torch.save(patch_dict, patches_path)
            size_mb2 = Path(patches_path).stat().st_size / 1e6
            print(f"  ✅ Patches saved: {patches_path}  ({size_mb2:.2f} MB)")
            print(f"     Shape: X={list(X_patches.shape)}")
            print(f"\n  Use for training:")
            print(f"     d = torch.load('{patches_path}')")
            print(f"     X, y = d['X'], d['y']   # y only if --label was set")


def main():
    args = build_parser().parse_args()

    # Build band_paths from explicit flags or auto-detect
    explicit = {
        "B4":  args.b4,  "B5":  args.b5,  "B6":  args.b6,  "B7":  args.b7,
        "B8":  args.b8,  "B8A": args.b8a, "B11": args.b11, "B12": args.b12,
    }
    has_explicit = any(v is not None for v in explicit.values())

    if args.batch_folders:
        # ── Batch mode ────────────────────────────────────────────
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        for folder in args.batch_folders:
            try:
                band_paths = auto_detect_bands(folder)
                out_path   = os.path.join(args.out_dir,
                                          Path(folder).name + ".pt")
                process_single(band_paths, args, out_path)
            except Exception as e:
                print(f"  ❌ {folder}: {e}")

    elif has_explicit:
        # ── Explicit mode (user named every band) ─────────────────
        band_paths = {k: v for k, v in explicit.items() if v is not None}
        process_single(band_paths, args, args.out)

    elif args.folder:
        # ── Auto-detect mode ───────────────────────────────────────
        print(f"Scanning {args.folder} for TIF files…")
        band_paths = auto_detect_bands(args.folder)
        print(f"  Found {len(band_paths)}/8 bands: {', '.join(sorted(band_paths))}")
        missing = [b for b in BAND_ORDER if b not in band_paths]
        if missing:
            print(f"  Missing: {', '.join(missing)}  (will be filled with zeros)")
        process_single(band_paths, args, args.out)

    else:
        build_parser().print_help()
        print("\nERROR: Provide --folder, --batch_folders, or explicit --b4 --b8 etc.")
        sys.exit(1)


if __name__ == "__main__":
    main()