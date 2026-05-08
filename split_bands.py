"""
split_bands.py — GWSat Band Splitter
--------------------------------------
Reads every TIF in raw/ (or a specified input folder).
For each TIF, auto-detects bands B4–B8, B8A, B11, B12 and writes
one GeoTIFF per band into:

    processed/<tiff_stem>/<BAND>/<tiff_stem>_<BAND>.tif

Handles two input layouts:
  A) Single multi-band TIF  — each rasterio band is mapped to a Sentinel-2 band
     by position (band 1→B4, 2→B5, 3→B6, 4→B7, 5→B8, 6→B8A, 7→B11, 8→B12).
  B) Folder of single-band TIFs — each file is matched to a band by filename
     (same regex patterns as tif_to_pt.py).

Usage:
    # Process all TIFs in raw/ (default)
    python split_bands.py

    # Custom input folder
    python split_bands.py --input_dir my_scenes/

    # Process a single multi-band TIF
    python split_bands.py --input_dir raw/ --file scene.tif

    # Custom output root
    python split_bands.py --output_dir my_processed/

    # Preview only (no files written)
    python split_bands.py --preview
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

# ── Sentinel-2 band config ────────────────────────────────────────────────────

BAND_ORDER = ["B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

BAND_PATTERNS = {
    "B4":  [r"_B04[_.]", r"_B4[_.]",  r"\bB04\b", r"\bB4\b"],
    "B5":  [r"_B05[_.]", r"_B5[_.]",  r"\bB05\b", r"\bB5\b"],
    "B6":  [r"_B06[_.]", r"_B6[_.]",  r"\bB06\b", r"\bB6\b"],
    "B7":  [r"_B07[_.]", r"_B7[_.]",  r"\bB07\b", r"\bB7\b"],
    "B8":  [r"_B08[_.]", r"_B8[_.]",  r"\bB08\b", r"(?<![A8])B8(?!A)\b"],
    "B8A": [r"_B8A[_.]", r"\bB8A\b",  r"_B08A[_.]"],
    "B11": [r"_B11[_.]", r"\bB11\b"],
    "B12": [r"_B12[_.]", r"\bB12\b"],
}

# Human-readable band descriptions for logging
BAND_DESC = {
    "B4":  "Red (665 nm)",
    "B5":  "Red-Edge 1 (705 nm)",
    "B6":  "Red-Edge 2 (740 nm)",
    "B7":  "Red-Edge 3 (783 nm)",
    "B8":  "NIR (842 nm)",
    "B8A": "Narrow NIR (865 nm)",
    "B11": "SWIR 1 (1610 nm)",
    "B12": "SWIR 2 (2190 nm)",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def match_band_from_filename(filename: str) -> str | None:
    """Return the Sentinel-2 band key that matches this filename, or None."""
    # Check B8A before B8 to avoid false match
    for band in ["B8A"] + [b for b in BAND_ORDER if b != "B8A"]:
        for pattern in BAND_PATTERNS[band]:
            if re.search(pattern, filename, re.IGNORECASE):
                return band
    return None


def write_band_tif(src_dataset, band_index: int,
                   out_path: Path, preview: bool = False) -> dict:
    """
    Extract one band from an open rasterio dataset and write it as a
    single-band GeoTIFF, preserving CRS, transform, nodata, and dtype.

    Returns metadata dict.
    """
    import rasterio
    from rasterio.enums import Resampling

    arr = src_dataset.read(band_index)   # (H, W) numpy array

    meta = src_dataset.meta.copy()
    meta.update({
        "count":   1,
        "driver":  "GTiff",
        "compress": "deflate",
        "tiled":   True,
        "blockxsize": 256,
        "blockysize": 256,
    })

    info = {
        "shape":   arr.shape,
        "dtype":   str(arr.dtype),
        "min":     float(arr.min()),
        "max":     float(arr.max()),
        "nodata":  src_dataset.nodata,
        "crs":     str(src_dataset.crs) if src_dataset.crs else "None",
        "out_path": str(out_path),
    }

    if not preview:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(arr, 1)

    return info


# ── Core processing functions ─────────────────────────────────────────────────

def split_multiband_tif(tif_path: Path, output_root: Path,
                        preview: bool = False) -> list[dict]:
    """
    Layout A: single TIF with N bands.
    Maps band positions 1–8 → B4, B5, B6, B7, B8, B8A, B11, B12.
    Extra bands (beyond 8) are skipped with a warning.
    """
    import rasterio
    results = []

    with rasterio.open(tif_path) as src:
        n_bands = src.count
        scene_name = tif_path.stem
        print(f"\n  Multi-band TIF: {tif_path.name}  ({n_bands} bands, "
              f"{src.width}×{src.height} px)")

        if n_bands < 8:
            print(f"  ⚠️  Only {n_bands} bands found — "
                  f"expected 8 (B4–B8, B8A, B11, B12).")
            print(f"     Will map what's available by position.")

        for i, band_name in enumerate(BAND_ORDER[:n_bands], start=1):
            out_dir  = output_root / scene_name
            out_file = out_dir / f"{scene_name}_{band_name}.tif"

            info = write_band_tif(src, i, out_file, preview=preview)
            info["band"]   = band_name
            info["source"] = str(tif_path)
            info["band_index"] = i
            results.append(info)

            status = "[preview]" if preview else "✅"
            print(f"    {status} Band {i} → {band_name:<4} "
                  f"({BAND_DESC.get(band_name, '')})"
                  f"  shape={info['shape']}  dtype={info['dtype']}"
                  f"  range=[{info['min']:.1f}, {info['max']:.1f}]"
                  + (f"\n         → {out_file}" if not preview else ""))

        if n_bands > 8:
            print(f"  ⚠️  {n_bands - 8} extra bands ignored "
                  f"(only 8 Sentinel-2 bands expected).")

    return results


def split_singleband_folder(folder: Path, output_root: Path,
                             preview: bool = False) -> list[dict]:
    """
    Layout B: folder containing one TIF per band.
    Each file is matched to a band by filename regex.
    Output goes to processed/<folder_name>/<BAND>/<stem>_<BAND>.tif
    """
    import rasterio

    tif_files = (list(folder.glob("*.tif")) +
                 list(folder.glob("*.TIF")) +
                 list(folder.glob("*.tiff")))

    if not tif_files:
        print(f"  ⚠️  No TIF files found in: {folder}")
        return []

    scene_name = folder.name
    print(f"\n  Single-band folder: {folder}  ({len(tif_files)} TIFs)")

    # Match files to bands
    band_files: dict[str, Path] = {}
    unmatched: list[Path] = []
    used: set[Path] = set()

    for band in ["B8A"] + [b for b in BAND_ORDER if b != "B8A"]:
        for tif in tif_files:
            if tif in used:
                continue
            if match_band_from_filename(tif.name) == band:
                band_files[band] = tif
                used.add(tif)
                break

    unmatched = [t for t in tif_files if t not in used]
    if unmatched:
        print(f"  ⚠️  {len(unmatched)} unmatched TIFs (will be skipped):")
        for u in unmatched:
            print(f"       {u.name}")

    results = []
    for band_name in BAND_ORDER:
        if band_name not in band_files:
            print(f"    ⚠️  {band_name:<4} — not found (skipped)")
            continue

        src_path = band_files[band_name]
        out_dir  = output_root / scene_name
        out_file = out_dir / f"{scene_name}_{band_name}.tif"

        with rasterio.open(src_path) as src:
            info = write_band_tif(src, 1, out_file, preview=preview)

        info["band"]   = band_name
        info["source"] = str(src_path)
        info["band_index"] = 1
        results.append(info)

        status = "[preview]" if preview else "✅"
        print(f"    {status} {band_name:<4} ({BAND_DESC.get(band_name, '')})"
              f"  shape={info['shape']}  dtype={info['dtype']}"
              f"  range=[{info['min']:.1f}, {info['max']:.1f}]"
              f"  ← {src_path.name}"
              + (f"\n         → {out_file}" if not preview else ""))

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Split Sentinel-2 TIF files into per-band GeoTIFFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input_dir",  default="raw",
                   help="Folder to scan for TIF files / scene folders (default: raw/)")
    p.add_argument("--output_dir", default="processed",
                   help="Output root directory (default: processed/)")
    p.add_argument("--file",       default=None,
                   help="Process a single TIF file (overrides --input_dir scan)")
    p.add_argument("--preview",    action="store_true",
                   help="Dry-run: show what would be written, but write nothing")
    args = p.parse_args()

    try:
        import rasterio
    except ImportError:
        print("ERROR: rasterio is required.  Run:  pip install rasterio")
        sys.exit(1)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.preview:
        print("─── PREVIEW MODE — no files will be written ───────────────")

    print(f"Input  : {input_dir.resolve()}")
    print(f"Output : {output_dir.resolve()}")
    print(f"Bands  : {', '.join(BAND_ORDER)}")

    all_results: list[dict] = []

    # ── Single file mode ──────────────────────────────────────────────────────
    if args.file:
        tif_path = Path(args.file)
        if not tif_path.exists():
            print(f"ERROR: File not found: {tif_path}")
            sys.exit(1)
        results = split_multiband_tif(tif_path, output_dir, preview=args.preview)
        all_results.extend(results)

    else:
        # ── Folder scan mode ──────────────────────────────────────────────────
        if not input_dir.exists():
            print(f"ERROR: Input directory not found: {input_dir}")
            print(f"       Create it and place your TIF files inside, or use --input_dir.")
            sys.exit(1)

        # Find items in input_dir:
        #   - Direct TIF files → Layout A (multi-band)
        #   - Sub-folders containing TIFs → Layout B (per-band folder)

        direct_tifs = (list(input_dir.glob("*.tif")) +
                       list(input_dir.glob("*.TIF")) +
                       list(input_dir.glob("*.tiff")))

        sub_folders = [d for d in input_dir.iterdir()
                       if d.is_dir() and not d.name.startswith(".")]

        if not direct_tifs and not sub_folders:
            print(f"ERROR: No TIF files or sub-folders found in {input_dir}")
            sys.exit(1)

        # Process direct multi-band TIFs
        for tif_path in sorted(direct_tifs):
            results = split_multiband_tif(tif_path, output_dir, preview=args.preview)
            all_results.extend(results)

        # Process sub-folders (each treated as one scene with per-band TIFs)
        for folder in sorted(sub_folders):
            folder_tifs = (list(folder.glob("*.tif")) +
                           list(folder.glob("*.TIF")) +
                           list(folder.glob("*.tiff")))
            if folder_tifs:
                results = split_singleband_folder(folder, output_dir, preview=args.preview)
                all_results.extend(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Summary: {len(all_results)} band file(s) processed")

    if not args.preview and all_results:
        # Count unique scene dirs written
        scenes = set()
        for r in all_results:
            out_p = Path(r["out_path"])
            # processed/<scene>/<band>/<file>.tif → <scene> is parts[-3]
            if len(out_p.parts) >= 3:
                scenes.add(out_p.parts[-3])

        print(f"  Scenes  : {len(scenes)}")
        print(f"  Output  : {output_dir.resolve()}")
        print()
        print("  Output layout:")
        print("    processed/")
        print("    └── <scene_name>/")
        for b in BAND_ORDER:
            print(f"        ├── <scene_name>_{b}.tif")
        print()
        print("  Use in GWSat:")
        print("    python tif_to_pt.py \\")
        print("        --b4  processed/<scene>/B4/<scene>_B4.tif \\")
        print("        --b8  processed/<scene>/B8/<scene>_B8.tif \\")
        print("        ... (etc) --out scene.pt")

    elif args.preview:
        print("  (Preview only — run without --preview to write files)")


if __name__ == "__main__":
    main()