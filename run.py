"""
run.py  --  GWSat single-command pipeline
------------------------------------------
One command. One folder. One verdict.

Usage:
    python run.py --folder ./Telangana_20231101

    # If the model is wrong, correct it immediately:
    python run.py --folder ./Telangana_20231101 --correct 1

    # Web UI (share with judges):
    python run.py --folder ./Telangana_20231101 --ui

Arguments:
    --folder    Folder containing TIF files for one scene/date (required)
    --correct   Override label if model was wrong: 0=Stable 1=Moderate 2=Critical
    --ui        Launch Gradio browser UI instead of CLI output
    --checkpoint Path to head weights (default: checkpoints/best_head.pth)
    --no_train  Skip quick-adapt even if corrections exist
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

BAND_ORDER = ["B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
LABELS = ["Stable", "Moderate", "Critical"]
CORRECTIONS_FILE = "corrections.jsonl"
CHECKPOINT = "checkpoints/best_head.pth"

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


# ---------------------------------------------------------------------------
# Step 1: TIF loading
# ---------------------------------------------------------------------------

def detect_bands(folder: str) -> dict:
    tifs = (glob.glob(os.path.join(folder, "*.tif")) +
            glob.glob(os.path.join(folder, "*.TIF")) +
            glob.glob(os.path.join(folder, "*.tiff")))
    if not tifs:
        raise FileNotFoundError(f"No TIF files found in: {folder}")

    found, used = {}, set()
    for band in BAND_ORDER:
        for tif in tifs:
            if tif in used:
                continue
            fname = os.path.basename(tif).upper()
            for pat in BAND_PATTERNS.get(band, []):
                if re.search(pat.upper(), fname):
                    found[band] = tif
                    used.add(tif)
                    break
            if band in found:
                break
    return found


def load_band(path: str, ref_shape=None) -> np.ndarray:
    try:
        import rasterio
        from rasterio.enums import Resampling
    except ImportError:
        sys.exit("Run:  pip install rasterio")

    with rasterio.open(path) as src:
        if ref_shape and (src.height, src.width) != ref_shape:
            arr = src.read(1, out_shape=(1, ref_shape[0], ref_shape[1]),
                           resampling=Resampling.bilinear).astype(np.float32)[0]
        else:
            arr = src.read(1).astype(np.float32)

    if arr.max() > 2.0:
        arr /= 10000.0
    return np.clip(arr, 0.0, 1.0)


def folder_to_tensor(folder: str) -> tuple:
    """Returns (tensor [8,H,W], band_paths dict, missing list)."""
    band_paths = detect_bands(folder)
    missing = [b for b in BAND_ORDER if b not in band_paths]

    ref_arr = load_band(band_paths["B4"]) if "B4" in band_paths else None
    if ref_arr is None:
        raise ValueError("B4 (Red) not found -- cannot proceed.")

    ref_shape = ref_arr.shape
    bands = []
    for band in BAND_ORDER:
        if band == "B4":
            bands.append(ref_arr)
        elif band in band_paths:
            bands.append(load_band(band_paths[band], ref_shape))
        else:
            bands.append(np.zeros(ref_shape, dtype=np.float32))

    tensor = torch.from_numpy(np.stack(bands, axis=0))
    return tensor, band_paths, missing


# ---------------------------------------------------------------------------
# Step 2: Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint: str):
    from model import GWSatModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GWSatModel(device=device)
    if Path(checkpoint).exists():
        model.load_head(checkpoint)
        print(f"Checkpoint loaded: {checkpoint}")
    else:
        print(f"No checkpoint at {checkpoint}. Run train.py first.")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Step 3: Inference
# ---------------------------------------------------------------------------

def run_inference(model, scene: torch.Tensor) -> dict:
    t0 = time.perf_counter()
    result = model.predict_scene(scene, patch_size=64, stride=32)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


# ---------------------------------------------------------------------------
# Step 4: Correction + quick adaptation
# ---------------------------------------------------------------------------

def save_correction(folder: str, tensor: torch.Tensor, true_label: int):
    """Save a correction record so future runs can adapt."""
    record = {
        "folder": folder,
        "true_label": true_label,
        "timestamp": time.time(),
    }
    with open(CORRECTIONS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Save the tensor patch for replay
    Path("corrections").mkdir(exist_ok=True)
    patch_path = f"corrections/{Path(folder).name}_{true_label}.pt"
    # Use mean tile from scene as the correction sample
    C, H, W = tensor.shape
    ps = 64
    patches = []
    for y in range(0, H - ps + 1, ps):
        for x in range(0, W - ps + 1, ps):
            p = tensor[:, y:y+ps, x:x+ps]
            b4, b8 = p[0], p[4]
            ndvi = ((b8 - b4) / (b8 + b4 + 1e-8)).mean().item()
            if ndvi > 0.05:
                patches.append(p)
    if patches:
        X = torch.stack(patches[:20])   # max 20 patches per correction
        y = torch.full((len(X),), true_label, dtype=torch.long)
        torch.save({"X": X, "y": y, "folder": folder}, patch_path)
        print(f"Correction saved: {patch_path}  ({len(X)} patches, label={LABELS[true_label]})")
    return patch_path


def quick_adapt(model, checkpoint: str):
    """
    Fine-tune only the classification head on all saved corrections.
    Runs in < 30 seconds on CPU. Saves updated checkpoint.
    Strategy: 10 epochs, low LR, only head weights.
    """
    corr_files = list(Path("corrections").glob("*.pt"))
    if not corr_files:
        print("No correction data found.")
        return

    print(f"\nAdapting model on {len(corr_files)} correction file(s)...")

    all_X, all_y = [], []
    for f in corr_files:
        d = torch.load(str(f), weights_only=False, map_location="cpu")
        all_X.append(d["X"])
        all_y.append(d["y"])
    X = torch.cat(all_X).to(model.device)
    y = torch.cat(all_y).to(model.device)

    # Snapshot current head before adapting (rollback safety)
    import copy
    old_head = copy.deepcopy(model.head.state_dict())

    model.eval()
    model.head.train()
    opt = torch.optim.Adam(model.head.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    EPOCHS = 15
    BATCH = min(16, len(X))
    for ep in range(1, EPOCHS + 1):
        idx = torch.randperm(len(X))
        total_loss = 0
        for i in range(0, len(X), BATCH):
            xb = X[idx[i:i+BATCH]]
            yb = y[idx[i:i+BATCH]]
            with torch.no_grad():
                feats = model._forward_features(xb)
                phys  = model._physics_scalars(xb)
            logits = model.head(feats, phys)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        if ep % 5 == 0:
            with torch.no_grad():
                feats = model._forward_features(X)
                phys  = model._physics_scalars(X)
                preds = model.head(feats, phys).argmax(1)
            acc = (preds == y).float().mean().item()
            print(f"  Epoch {ep:2d}/{EPOCHS}  loss={total_loss:.4f}  correction_acc={acc:.2%}")
            # Rollback if accuracy collapses
            if acc < 0.30:
                print("  Accuracy collapsed -- rolling back.")
                model.head.load_state_dict(old_head)
                return

    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    model.save_head(checkpoint)
    print(f"Updated checkpoint saved: {checkpoint}")


# ---------------------------------------------------------------------------
# Step 5: Print result
# ---------------------------------------------------------------------------

def print_verdict(result: dict, folder: str, missing: list):
    cls   = result["stress_class"]
    conf  = result["confidence"]
    probs = result["probabilities"]
    si    = result["spectral_indices"]
    lat   = result.get("latency_ms", 0)
    pd    = result.get("patch_distribution", {})

    line = "-" * 56
    print(f"\n{line}")
    print(f"  GWSat  --  {Path(folder).name}")
    print(line)
    print(f"  Verdict    : {LABELS[cls].upper()}")
    print(f"  Confidence : {conf:.1%}   Latency: {lat} ms")
    if pd:
        print(f"  Patches    : Stable={pd.get('Stable',0)}  "
              f"Moderate={pd.get('Moderate',0)}  "
              f"Critical={pd.get('Critical',0)}")
    if missing:
        print(f"  Missing    : {', '.join(missing)} (filled with zeros)")
    print()
    print(f"  {'Class':<12} {'Prob':>6}")
    for lbl, p in probs.items():
        bar = "#" * int(p * 30)
        print(f"  {lbl:<12} {p:>5.1%}  {bar}")
    print()
    print(f"  NDVI  {si['NDVI']:+.4f}  "
          f"({'misleading' if si['NDVI'] > 0.40 and cls > 0 else 'consistent'})")
    print(f"  LSWI  {si['LeafWaterStress_LSWI']:+.4f}  "
          f"({'water deficit' if si['LeafWaterStress_LSWI'] < 0.20 else 'adequate'})")
    print(f"  IRP   {si['IR_Pressure_Index']:+.4f}  "
          f"({'stomata closing' if si['IR_Pressure_Index'] > 0.10 else 'normal'})")
    print(line)

    if cls == 2:
        print("  ALERT: Critical pre-drought signal.")
        print("  Recommend: notify NDMA, check crop insurance. 2-4 week window.")
    elif cls == 1:
        print("  WARN:  Moderate stress. Field inspection within 14 days.")
    else:
        print("  OK:    Stable. Standard monitoring.")
    print()
    print(f"  To correct this result:")
    print(f"  python run.py --folder {folder} --correct <0|1|2>")
    print(line)


# ---------------------------------------------------------------------------
# Step 6: Gradio UI (optional)
# ---------------------------------------------------------------------------

def launch_ui(args):
    try:
        import gradio as gr
    except ImportError:
        sys.exit("Run:  pip install gradio>=4.0")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = load_model(args.checkpoint)

    def process(folder_path, correct_label, do_correct):
        folder_path = folder_path.strip()
        if not folder_path:
            return "Provide a folder path.", None, None

        try:
            scene, band_paths, missing = folder_to_tensor(folder_path)
        except Exception as e:
            return f"Error loading TIFs: {e}", None, None

        if do_correct and correct_label is not None:
            true_lbl = int(correct_label.split(":")[0])
            save_correction(folder_path, scene, true_lbl)
            quick_adapt(model, args.checkpoint)
            return f"Correction applied. Label set to {LABELS[true_lbl]}. Model updated.", None, None

        result = run_inference(model, scene)
        cls    = result["stress_class"]
        probs  = result["probabilities"]
        si     = result["spectral_indices"]
        conf   = result["confidence"]

        status = (
            f"## {LABELS[cls].upper()}  ({conf:.1%} confidence)\n\n"
            f"Latency: {result.get('latency_ms',0)} ms\n\n"
        )
        if missing:
            status += f"Missing bands (zeros): {', '.join(missing)}\n\n"

        alerts = {
            0: "Stable. Standard monitoring.",
            1: "Moderate hidden stress. NDVI may look healthy. Inspect within 14 days.",
            2: "CRITICAL. Stomatal closure signal detected 2-4 weeks before crop failure. Notify NDMA.",
        }
        status += alerts[cls]

        # Probability chart
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.barh(list(probs.keys()), list(probs.values()),
                color=["#2ecc71", "#f39c12", "#e74c3c"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        plt.tight_layout()

        # Spectral table
        spec = (
            f"| Index | Value | Signal |\n"
            f"|---|---|---|\n"
            f"| NDVI | {si['NDVI']:.4f} | {'misleading' if si['NDVI']>0.4 and cls>0 else 'consistent'} |\n"
            f"| LSWI | {si['LeafWaterStress_LSWI']:.4f} | {'water deficit' if si['LeafWaterStress_LSWI']<0.20 else 'adequate'} |\n"
            f"| IRP  | {si['IR_Pressure_Index']:.4f} | {'stomata closing' if si['IR_Pressure_Index']>0.10 else 'normal'} |\n"
        )

        return status, fig, spec

    with gr.Blocks(title="GWSat") as demo:
        gr.Markdown("# GWSat -- Pre-Drought Stress Monitor\nPoint at a folder of TIF files, get a verdict.")

        with gr.Row():
            folder_in = gr.Textbox(label="TIF folder path", placeholder="/path/to/Telangana_20231101")
            with gr.Column():
                correct_dd = gr.Dropdown(
                    ["0: Stable", "1: Moderate", "2: Critical"],
                    label="Correction (if model was wrong)", value=None
                )
                correct_btn = gr.Button("Apply correction + retrain", variant="secondary")

        run_btn = gr.Button("Run", variant="primary", size="lg")

        status_out = gr.Markdown()
        prob_plot  = gr.Plot(label="Probabilities")
        spec_out   = gr.Markdown(label="Spectral indices")

        run_btn.click(
            fn=lambda f, c: process(f, c, False),
            inputs=[folder_in, correct_dd],
            outputs=[status_out, prob_plot, spec_out]
        )
        correct_btn.click(
            fn=lambda f, c: process(f, c, True),
            inputs=[folder_in, correct_dd],
            outputs=[status_out, prob_plot, spec_out]
        )

    demo.launch(share=args.share, server_port=getattr(args, "port", 7860))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="GWSat single-command pipeline")
    p.add_argument("--folder",     required=False, help="Folder of TIF files")
    p.add_argument("--correct",    type=int, choices=[0, 1, 2],
                   help="Correct the model: 0=Stable 1=Moderate 2=Critical")
    p.add_argument("--ui",         action="store_true", help="Launch Gradio UI")
    p.add_argument("--share",      action="store_true", help="Public Gradio URL")
    p.add_argument("--port",       type=int, default=7860)
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--no_train",   action="store_true",
                   help="Skip quick-adapt even if corrections exist")
    args = p.parse_args()

    if args.ui:
        launch_ui(args)
        return

    if not args.folder:
        p.print_help()
        sys.exit(1)

    print(f"\nLoading TIFs from: {args.folder}")
    scene, band_paths, missing = folder_to_tensor(args.folder)
    C, H, W = scene.shape
    print(f"Scene: {C} bands x {H} x {W} px")
    if missing:
        print(f"Missing bands (zeros): {', '.join(missing)}")

    # If --correct, save correction and adapt, then re-run
    if args.correct is not None:
        model = load_model(args.checkpoint)
        save_correction(args.folder, scene, args.correct)
        if not args.no_train:
            quick_adapt(model, args.checkpoint)
        result = run_inference(model, scene)
        print_verdict(result, args.folder, missing)
        return

    # Normal inference, with optional auto-adapt if corrections exist
    model = load_model(args.checkpoint)
    if not args.no_train and Path(CORRECTIONS_FILE).exists():
        print("Correction data found -- adapting before inference...")
        quick_adapt(model, args.checkpoint)

    result = run_inference(model, scene)
    print_verdict(result, args.folder, missing)


if __name__ == "__main__":
    main()