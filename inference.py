"""
inference.py — GWSat ONNX Inference
=====================================
Replaces the old inference.py and run_onnx.py.

How to run:
    # Test the pipeline works (random data, no TIF files needed):
    python inference.py --dummy

    # On your real Telangana TIF folder:
    python inference.py --folder ./Telangana_Instant_Data

    # On a saved .pt scene file:
    python inference.py --pt scene.pt

    # On Jetson (GPU/TensorRT):
    python inference.py --folder ./Telangana_Instant_Data --jetson

Install:
    pip install onnxruntime numpy
    pip install rasterio       # only needed for --folder
"""

import argparse
import os
import sys
import numpy as np


# ─── find the onnx file automatically ────────────────────────────────────────
# Looks in checkpoints/ for any .onnx file.
# Prefers gwsat_terratorch.onnx, then any other gwsat_*.onnx, then any .onnx.

def find_onnx() -> str:
    folder = "checkpoints"
    if not os.path.isdir(folder):
        sys.exit("No checkpoints/ folder found. Run export.py first.")

    candidates = [f for f in os.listdir(folder) if f.endswith(".onnx")]
    if not candidates:
        sys.exit("No .onnx file found in checkpoints/. Run: python export.py --verify")

    # priority order
    for preferred in ["gwsat_terratorch.onnx", "gwsat_hf-transformers.onnx",
                      "gwsat_timm-proxy.onnx", "gwsat_edge-cnn.onnx"]:
        if preferred in candidates:
            return os.path.join(folder, preferred)

    # fall back to the first gwsat_*.onnx found
    gwsat = [f for f in candidates if f.startswith("gwsat_")]
    if gwsat:
        return os.path.join(folder, gwsat[0])

    # fall back to whatever .onnx exists
    return os.path.join(folder, candidates[0])


# ─── step 1: load session ─────────────────────────────────────────────────────

def load_session(onnx_path: str, jetson: bool = False):
    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("Missing library. Run: pip install onnxruntime")

    if not os.path.isfile(onnx_path):
        sys.exit(f"ONNX file not found: {onnx_path}\nRun: python export.py --verify")

    providers = (
        ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        if jetson else
        ["CPUExecutionProvider"]
    )

    print(f"Loading : {onnx_path}  ({os.path.getsize(onnx_path)//1024} KB)")
    session = ort.InferenceSession(onnx_path, providers=providers)

    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"Input   : name='{inp.name}'  shape={inp.shape}  dtype={inp.type}")
    print(f"Output  : name='{out.name}'  shape={out.shape}")
    print()

    return session


# ─── step 2: get input data ───────────────────────────────────────────────────

def make_dummy_patch() -> np.ndarray:
    """Random values in [0,1] — use this just to check the pipeline runs."""
    print("Mode: dummy (random data — prediction is meaningless)")
    print()
    patch = np.random.rand(1, 8, 64, 64).astype(np.float32)
    return patch


def load_patch_from_folder(folder: str) -> np.ndarray:
    """
    Auto-detects band TIF files in the folder, loads the full scene,
    then cuts the centre 64x64 patch.
    """
    try:
        sys.path.insert(0, ".")
        from tif_to_pt import auto_detect_bands, tifs_to_tensor
    except ImportError:
        sys.exit("Cannot find tif_to_pt.py — run this script from your project folder.")

    print(f"Mode: folder  →  {folder}")
    band_paths = auto_detect_bands(folder)
    missing = [b for b in ["B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
               if b not in band_paths]
    print(f"  Bands found   : {', '.join(sorted(band_paths.keys()))}")
    if missing:
        print(f"  Bands missing : {', '.join(missing)}  (filled with zeros)")

    scene = tifs_to_tensor(band_paths, verbose=False)
    C, H, W = scene.shape
    print(f"  Scene shape   : {C} bands x {H} x {W} px")

    cy, cx = H // 2, W // 2
    patch = scene[:, cy-32:cy+32, cx-32:cx+32].numpy().astype(np.float32)
    patch = patch[np.newaxis]   # add batch dimension → (1, 8, 64, 64)

    print(f"  Patch shape   : {patch.shape}")
    print(f"  Value range   : {patch.min():.4f} – {patch.max():.4f}")
    print()
    return patch


def load_patch_from_pt(pt_path: str) -> np.ndarray:
    """Load a patch from a .pt scene file created by tif_to_pt.py."""
    try:
        import torch
    except ImportError:
        sys.exit("Missing library. Run: pip install torch")

    if not os.path.isfile(pt_path):
        sys.exit(f"File not found: {pt_path}")

    print(f"Mode: .pt file  →  {pt_path}")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    scene = data["X"] if isinstance(data, dict) and "X" in data else data

    if scene.ndim == 4:
        # already a batch of patches — take the first one
        patch = scene[0:1].numpy().astype(np.float32)
    else:
        # full scene — cut the centre patch
        C, H, W = scene.shape
        cy, cx = H // 2, W // 2
        patch = scene[:, cy-32:cy+32, cx-32:cx+32].numpy().astype(np.float32)
        patch = patch[np.newaxis]

    print(f"  Patch shape : {patch.shape}")
    print()
    return patch


# ─── step 3: run inference ────────────────────────────────────────────────────

def run_inference(session, patch: np.ndarray) -> dict:
    import time

    # feed the patch into the model
    # "s2_patch" is the input name baked into the .onnx file by export.py
    t0 = time.perf_counter()
    outputs = session.run(None, {"s2_patch": patch})
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # outputs[0] has shape (1, 3) — one row, three class scores
    logits = outputs[0][0]   # drop batch dimension → shape (3,)

    # softmax: convert raw scores into probabilities that sum to 1.0
    # subtract max first to prevent overflow (doesn't change the result)
    exp_l = np.exp(logits - logits.max())
    probs = exp_l / exp_l.sum()

    labels = ["Stable", "Moderate", "Critical"]
    idx = int(np.argmax(probs))

    return {
        "class":         labels[idx],
        "confidence":    float(probs[idx]),
        "prob_stable":   float(probs[0]),
        "prob_moderate": float(probs[1]),
        "prob_critical": float(probs[2]),
        "logits":        logits.tolist(),
        "latency_ms":    latency_ms,
    }


# ─── step 4: print result ─────────────────────────────────────────────────────

def print_result(r: dict):
    icons   = {"Stable": "OK   ", "Moderate": "WARN ", "Critical": "ALERT"}
    actions = {
        "Stable":   "No action needed. Standard monitoring.",
        "Moderate": "Early stress detected. Field inspection within 14 days.",
        "Critical": "Stomatal closure confirmed. Alert NDMA. 2–4 week window.",
    }

    cls  = r["class"]
    conf = r["confidence"]

    print("=" * 54)
    print(f"  [{icons[cls]}]  {cls.upper()}")
    print(f"  Confidence  : {conf:.1%}")
    print(f"  Latency     : {r['latency_ms']} ms")
    print()

    for label, prob in [
        ("Stable",   r["prob_stable"]),
        ("Moderate", r["prob_moderate"]),
        ("Critical", r["prob_critical"]),
    ]:
        bar = "#" * int(prob * 32)
        print(f"  {label:<10} {prob:>5.1%}  {bar}")

    print()
    print(f"  Action : {actions[cls]}")
    print("=" * 54)
    print()
    print(f"  Raw logits → Stable={r['logits'][0]:.3f}  "
          f"Moderate={r['logits'][1]:.3f}  "
          f"Critical={r['logits'][2]:.3f}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GWSat ONNX inference")
    parser.add_argument("--onnx",   default=None,
                        help="Path to .onnx file (auto-detected if not given)")
    parser.add_argument("--folder", help="Folder containing TIF band files")
    parser.add_argument("--pt",     help="Path to a saved .pt scene file")
    parser.add_argument("--dummy",  action="store_true",
                        help="Run on random dummy data to test the pipeline")
    parser.add_argument("--jetson", action="store_true",
                        help="Use TensorRT/GPU providers for Jetson deployment")
    args = parser.parse_args()

    if not args.folder and not args.pt and not args.dummy:
        parser.print_help()
        print()
        print("Examples:")
        print("  python inference.py --dummy")
        print("  python inference.py --folder ./Telangana_Instant_Data")
        print("  python inference.py --pt scene.pt")
        sys.exit(1)

    # find the onnx file
    onnx_path = args.onnx if args.onnx else find_onnx()

    # step 1
    session = load_session(onnx_path, jetson=args.jetson)

    # step 2
    if args.dummy:
        patch = make_dummy_patch()
    elif args.folder:
        patch = load_patch_from_folder(args.folder)
    else:
        patch = load_patch_from_pt(args.pt)

    # step 3
    result = run_inference(session, patch)

    # step 4
    print_result(result)


if __name__ == "__main__":
    main()