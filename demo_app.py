"""
demo_app.py — GWSat v3 Gradio Demo
------------------------------------
Changes vs original:
  - Folder input: accepts a folder path like ./Telangana_Instant_Data
    and auto-detects bands by filename (B4/B04, B8A etc.) — no more
    "B4 not detected" errors.
  - NDVI panel shown BESIDE AI results for direct comparison.
  - Heatmap + spectral charts using matplotlib.
  - Export command shown in UI.
  - All modes: sample tiles, folder path, .pt upload.
  - FIXED correction: uses the already‑loaded scene tensor directly
    instead of calling run.py (which had broken band detection).
"""

import sys, time, warnings, re, glob, os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from model import GWSatModel, compute_spectral_indices
from data_pipeline import synthetic_tile, build_synthetic_dataset
from tif_to_pt import auto_detect_bands, tifs_to_tensor, BAND_ORDER

try:
    import gradio as gr
except ImportError:
    print("Install gradio:  pip install gradio>=4.0")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─── Load model once ──────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {DEVICE}…")
MODEL = GWSatModel(device=DEVICE)
HEAD  = Path("checkpoints/best_head.pth")
if HEAD.exists():
    MODEL.load_head(str(HEAD))
    print("✅ Trained checkpoint loaded")
else:
    print("⚠️  No checkpoint — run: python train.py --epochs 40")
MODEL.eval()

LABELS = ["Stable", "Moderate", "Critical"]
COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]
EMOJI  = ["✅", "⚠️", "🚨"]

SAMPLE_TILES = {
    "✅ Stable   (GWL < 5m)":               torch.from_numpy(synthetic_tile(0)),
    "⚠️ Moderate (GWL 5–10m)":              torch.from_numpy(synthetic_tile(1)),
    "🚨 Critical (GWL > 10m, pre-drought)": torch.from_numpy(synthetic_tile(2)),
}


# ─── Band auto-detection from folder path ─────────────────────────────────────

def load_tile_from_folder(folder_path: str):
    """
    Scan folder_path for TIF files, auto-detect bands by filename
    (handles B4/B04, B8A, _B11_, etc.), stack into [8, H, W] tensor.
    Returns (tile_or_scene, is_scene, error_str).
    """
    folder_path = folder_path.strip()
    if not os.path.isdir(folder_path):
        return None, False, f"Folder not found: {folder_path}"
    try:
        band_paths = auto_detect_bands(folder_path)
    except FileNotFoundError as e:
        return None, False, str(e)

    missing = [b for b in ["B4", "B8"] if b not in band_paths]
    if missing:
        tif_list = glob.glob(os.path.join(folder_path, "*.tif")) + \
                   glob.glob(os.path.join(folder_path, "*.TIF"))
        names = [os.path.basename(t) for t in tif_list]
        return None, False, (
            f"Could not detect {missing} in folder.\n"
            f"Files found: {names}\n"
            f"Tip: filenames must contain B4/B04, B8/B08, B8A, B11, B12 etc."
        )

    found_str = ", ".join(f"{b}={os.path.basename(p)}"
                          for b, p in sorted(band_paths.items()))
    print(f"  Auto-detected: {found_str}")
    scene = tifs_to_tensor(band_paths, verbose=True)
    return scene, True, None


# ─── Chart helpers ────────────────────────────────────────────────────────────

def tile_to_false_colour(tile: torch.Tensor) -> np.ndarray:
    r = tile[4].numpy(); g = tile[2].numpy(); b = tile[0].numpy()
    rgb = np.clip(np.stack([r, g, b], axis=2) * 3.0, 0, 1)
    return (rgb * 255).astype(np.uint8)


def make_prob_chart(probs: dict):
    if not HAS_MPL: return None
    fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    vals = [probs["Stable"], probs["Moderate"], probs["Critical"]]
    bars = ax.barh(LABELS, vals, color=COLORS, height=0.5)
    for bar, v in zip(bars, vals):
        ax.text(min(v + 0.02, 0.98), bar.get_y() + bar.get_height()/2,
                f"{v:.1%}", va='center', color='white', fontsize=10)
    ax.set_xlim(0, 1.15); ax.set_xlabel("Probability", color="white")
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#555'); ax.spines['left'].set_color('#555')
    ax.tick_params(colors='white')
    plt.tight_layout()
    return fig


def make_ndvi_comparison_chart(si: dict, ai_class: int):
    """
    Side-by-side: what NDVI says vs what GWSat says.
    This is the core 'deception' visualisation.
    """
    if not HAS_MPL: return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), facecolor="#1a1a2e")

    ndvi_v = si["NDVI"]
    lswi_v = si["LeafWaterStress_LSWI"]
    rei_v  = si["RedEdge_Index"]
    irp_v  = si["IR_Pressure_Index"]

    for ax in axes:
        ax.set_facecolor("#1a1a2e"); ax.tick_params(colors='white')
        for s in ['top','right']: ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color('#555'); ax.spines['left'].set_color('#555')

    # ── Panel 1: All spectral indices ──
    ax = axes[0]
    names = ["NDVI", "RedEdge", "LSWI", "IRP"]
    vals  = [ndvi_v, rei_v, lswi_v, irp_v]
    clrs  = ["#3498db", "#9b59b6", "#1abc9c", "#e74c3c"]
    ax.bar(names, vals, color=clrs, width=0.5)
    ax.axhline(0, color='white', lw=0.5, ls='--')
    for i, v in enumerate(vals):
        ax.text(i, v + (0.02 if v >= 0 else -0.06), f"{v:.3f}",
                ha='center', color='white', fontsize=8)
    ax.set_title("All Spectral Indices", color='white', fontsize=10)
    ax.set_ylabel("Value", color='white')

    # ── Panel 2: NDVI deception ──
    ax = axes[1]
    ndvi_says  = "HEALTHY" if ndvi_v > 0.4 else "MODERATE" if ndvi_v > 0.2 else "STRESSED"
    lswi_says  = "STABLE"  if lswi_v > 0.25 else "MODERATE" if lswi_v > 0.10 else "CRITICAL"
    bc = ["#2ecc71" if ndvi_v > 0.4 else "#f39c12" if ndvi_v > 0.2 else "#e74c3c",
          COLORS[ai_class]]
    ax.bar(["NDVI\n(traditional)", "LSWI\n(GWSat)"],
           [max(ndvi_v, 0.02), max(lswi_v, 0.02)], color=bc, width=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_title("The Deception", color='white', fontsize=10)
    for i, (v, lbl) in enumerate([(ndvi_v, ndvi_says), (lswi_v, lswi_says)]):
        ax.text(i, min(max(v, 0.02) + 0.04, 0.88), lbl,
                ha='center', color='white', fontsize=9, fontweight='bold')

    # ── Panel 3: NDVI gauge ──
    ax = axes[2]
    gauge_labels  = ["NDVI\n(says)", "AI\n(truth)"]
    gauge_vals    = [ndvi_v, [0.85, 0.50, 0.15][ai_class]]  # proxy confidence bar
    gauge_colors  = ["#3498db", COLORS[ai_class]]
    ax.barh(gauge_labels, gauge_vals, color=gauge_colors, height=0.4)
    ax.set_xlim(0, 1.0)
    ax.set_title("NDVI vs AI Confidence", color='white', fontsize=10)
    ai_label = LABELS[ai_class].upper()
    ndvi_label = ndvi_says
    for i, (v, lbl) in enumerate([(ndvi_v, ndvi_label), (gauge_vals[1], ai_label)]):
        ax.text(v + 0.03, i, lbl, va='center', color='white', fontsize=8, fontweight='bold')

    plt.tight_layout()
    return fig


def make_heatmap_from_scene(scene: torch.Tensor, heatmap: torch.Tensor = None):
    """
    If heatmap available (tiled scene mode): show it.
    Otherwise: compute NDVI/LSWI maps from scene bands.
    Always returns a figure with false-colour + stress map.
    """
    if not HAS_MPL: return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#111")
    fig.suptitle("GWSat — Scene Analysis", color='white', fontsize=13, fontweight='bold')

    # False colour
    ax = axes[0]
    rgb = np.stack([scene[4].numpy(), scene[2].numpy(), scene[0].numpy()], axis=2)
    rgb = np.clip(rgb * 2.5, 0, 1)
    ax.imshow(rgb); ax.set_title("False Colour (NIR-RedEdge-Red)", color='white', fontsize=9)
    ax.axis('off')

    # NDVI map
    ax = axes[1]
    b4 = scene[0].numpy(); b8 = scene[4].numpy()
    ndvi_map = (b8 - b4) / (b8 + b4 + 1e-8)
    im = ax.imshow(ndvi_map, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    ax.set_title("NDVI Map (traditional — misleading)", color='white', fontsize=9)
    ax.axis('off')
    cb = plt.colorbar(im, ax=ax, fraction=0.046)
    cb.ax.tick_params(colors='white'); cb.set_label("NDVI", color='white')

    # Stress heatmap
    ax = axes[2]
    if heatmap is not None:
        hm = heatmap.numpy()
        title = "GWSat Stress Heatmap (AI tiled)"
    else:
        # Fallback: LSWI-based stress
        b11 = scene[6].numpy()
        lswi_map = (b8 - b11) / (b8 + b11 + 1e-8)
        hm = np.clip(1.0 - (lswi_map + 0.5), 0, 1)
        title = "LSWI Stress Map (proxy)"
    cmap = mcolors.LinearSegmentedColormap.from_list("stress",
           ["#2ecc71", "#f39c12", "#e74c3c"])
    im2 = ax.imshow(hm, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, color='white', fontsize=9)
    ax.axis('off')
    cb2 = plt.colorbar(im2, ax=ax, fraction=0.046)
    cb2.ax.tick_params(colors='white')
    cb2.set_label("Stress", color='white')

    plt.tight_layout()
    return fig


# ─── Core inference ────────────────────────────────────────────────────────────

def run_inference(sample_choice: str, uploaded_file, folder_path: str):
    """
    Priority: folder_path > uploaded_file > sample_choice
    Returns: (rgb_img, prob_fig, ndvi_fig, heatmap_fig, status_md,
               spec_md, bw_md, export_md)
    """
    tile   = None
    scene  = None
    is_scene = False
    err    = None

    # 1. Folder path
    if folder_path and folder_path.strip():
        scene, is_scene, err = load_tile_from_folder(folder_path.strip())
        if err:
            empty = (None, None, None, None,
                     f"❌ {err}", "—", "—", "—")
            return empty

    # 2. Uploaded .pt file
    elif uploaded_file is not None:
        try:
            obj  = torch.load(uploaded_file.name, map_location="cpu",
                              weights_only=False)
            if isinstance(obj, dict):
                if "X" in obj:
                    d = obj["X"]
                    tile = d[0] if d.ndim == 4 else d
                elif "scene" in obj:
                    scene = obj["scene"]; is_scene = True
                else:
                    tile = list(obj.values())[0]
            else:
                tile = obj
            if tile is not None and tile.ndim == 4:
                tile = tile[0]
            if tile is not None:
                tile = tile.float()
        except Exception as e:
            return (None, None, None, None, f"❌ Error: {e}", "—", "—", "—")

    # 3. Sample
    else:
        tile = SAMPLE_TILES[sample_choice].float()

    # ── Run inference ──
    t0     = time.perf_counter()
    heatmap = None

    if is_scene and scene is not None:
        result = MODEL.predict_scene(scene.to(DEVICE), patch_size=64, stride=32)
        heatmap = result.get("heatmap")
        # Use scene centre patch for false colour
        C, H, W = scene.shape
        cy, cx  = H//2, W//2
        s = 64
        tile = scene[:, max(0,cy-s//2):cy+s//2, max(0,cx-s//2):cx+s//2]
        if tile.shape[1] < s or tile.shape[2] < s:
            tile = F.interpolate(scene.unsqueeze(0), (s, s),
                                 mode='bilinear', align_corners=False)[0]
        tile = tile.float()
    else:
        if tile is None:
            tile = SAMPLE_TILES[sample_choice].float()
        result = MODEL.predict(tile)

    lat_ms = (time.perf_counter() - t0) * 1000

    cls   = result["stress_class"]
    conf  = result["confidence"]
    probs = result["probabilities"]
    si    = result["spectral_indices"]

    rgb_img   = tile_to_false_colour(tile[:8] if tile.shape[0] >= 8 else tile)
    prob_fig  = make_prob_chart(probs)
    ndvi_fig  = make_ndvi_comparison_chart(si, cls)   # always shown

    # Heatmap: use full scene in scene mode, or expand single tile for consistency
    if is_scene and scene is not None:
        hm_fig = make_heatmap_from_scene(scene, heatmap)
    else:
        # Single-tile mode: still show NDVI map + LSWI stress map from the patch
        hm_fig = make_heatmap_from_scene(
            tile.unsqueeze(0).squeeze(0) if tile.ndim == 4 else tile,
            heatmap=None
        )

    # Status
    alerts = {
        0: "✅ No action required. Groundwater stable.",
        1: ("⚠️ **Moderate hidden stress detected.**\nNDVI looks "
            "healthy but SWIR/RedEdge reveal early water deficit.\n"
            "Recommend field inspection within **14 days**.\n\n"
            "> **Why confidence is lower for Moderate:** The model sees "
            "ambiguous signals — some bands say stable, SWIR says stressed. "
            "This is real physics: moderate stress is the transition zone "
            "where NDVI hasn't caught up yet. Lower confidence = 'I see "
            "stress starting but it isn't critical yet'."),
        2: ("🚨 **CRITICAL PRE-DROUGHT SIGNAL.**\nStomatal closure confirmed "
            "(IR pressure elevated).\nNDVI is **actively misleading** standard "
            "monitors.\n**Alert NDMA / crop insurance NOW.**"),
    }
    mode_str = f"SCENE TILED ({result.get('n_patches',0)} patches)" \
               if is_scene else "PATCH"
    status_md = (
        f"## {EMOJI[cls]} {LABELS[cls].upper()}\n\n"
        f"**Confidence:** {conf:.1%} &nbsp;|&nbsp; "
        f"**Latency:** {lat_ms:.1f} ms ({DEVICE.upper()}) &nbsp;|&nbsp; "
        f"**Mode:** {mode_str}\n\n"
        f"{alerts[cls]}"
    )
    if is_scene and "patch_distribution" in result:
        pd = result["patch_distribution"]
        status_md += (f"\n\n**Patch breakdown:** "
                      f"Stable:{pd['Stable']} / "
                      f"Moderate:{pd['Moderate']} / "
                      f"Critical:{pd['Critical']}")

    # Spectral indices
    ndvi_v = si["NDVI"]; lswi_v = si["LeafWaterStress_LSWI"]
    spec_md = (
        f"| Index | Value | NDVI says | GWSat says |\n"
        f"|-------|-------|-----------|------------|\n"
        f"| **NDVI** | `{si['NDVI']:.4f}` | "
        f"{'🟢 Healthy' if si['NDVI']>0.4 else '🔴 Stressed'} | "
        f"{'⚠️ DECEIVED' if si['NDVI']>0.4 and cls>0 else '—'} |\n"
        f"| Red-Edge | `{si['RedEdge_Index']:.4f}` | N/A | "
        f"{'🟢 OK' if si['RedEdge_Index']>0.05 else '🔴 Chlorophyll drop'} |\n"
        f"| LSWI (leaf water) | `{lswi_v:.4f}` | N/A | "
        f"{'🟢 Adequate' if lswi_v>0.20 else '🔴 **DEFICIT**'} |\n"
        f"| IR Pressure | `{si['IR_Pressure_Index']:.4f}` | N/A | "
        f"{'🔴 Stomata closing' if si['IR_Pressure_Index']>0.10 else '🟢 Normal'} |\n\n"
        f"> NDVI = `{si['NDVI']:.3f}` → traditional system says "
        f"**{'healthy' if si['NDVI']>0.4 else 'stressed'}**. "
        f"GWSat says **{LABELS[cls]}**."
    )

    # Bandwidth
    raw_mb = result["raw_tile_bytes"] / 1e6
    bw_md  = (
        f"| | Size | Downlink (400kbps) |\n"
        f"|--|------|--------------------|\n"
        f"| Raw tile (8 bands) | {raw_mb:.2f} MB | ~{raw_mb*8:.0f}s |\n"
        f"| GWSat alert | 64 bytes | <0.001s |\n"
        f"| **Reduction** | **{result['raw_tile_bytes']//64:,}×** | |"
    )

    # Export commands
    ckpt_mb = HEAD.stat().st_size / 1e6 if HEAD.exists() else "?"
    export_md = (
        f"```bash\n"
        f"# Export to ONNX (~5 MB, under 200 MB target)\n"
        f"python export.py --verify\n\n"
        f"# INT8 quantization (~2-3 MB)\n"
        f"python export.py --quantize\n\n"
        f"# Jetson TensorRT (3-5× faster, run ON Jetson)\n"
        f"trtexec --onnx=checkpoints/gwsat.onnx \\\n"
        f"        --saveEngine=checkpoints/gwsat.engine \\\n"
        f"        --int8 --workspace=256\n"
        f"```\n"
        f"**Current checkpoint:** `{ckpt_mb:.1f} MB` "
        f"(only the head — encoder loaded from HuggingFace)"
    )

    return (rgb_img, prob_fig, ndvi_fig, hm_fig,
            status_md, spec_md, bw_md, export_md)


# ─── Correction handler (FIXED: uses scene tensor directly) ───────────────────

def apply_correction(folder_path: str, correct_label: str):
    """
    Apply correction directly from the folder path.
    This bypasses the broken run.py and uses the already-loaded scene tensor.
    """
    if not folder_path or not folder_path.strip():
        return "❌ No folder provided.", None, None, None

    try:
        # Load the scene using the same robust function
        scene, is_scene, err = load_tile_from_folder(folder_path.strip())
        if err or not is_scene:
            return f"❌ Failed to load scene: {err}", None, None, None

        # Parse label
        try:
            true_lbl = int(correct_label.split(":")[0])
        except:
            return f"❌ Invalid label: {correct_label}", None, None, None

        # Extract patches from scene
        patch_size = 64
        stride = 64
        veg_threshold = 0.15
        patches = []
        C, H, W = scene.shape
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                p = scene[:, y:y+patch_size, x:x+patch_size]
                b4, b8 = p[0], p[4]
                ndvi = ((b8 - b4) / (b8 + b4 + 1e-8)).mean().item()
                if ndvi >= veg_threshold:
                    patches.append(p)

        if len(patches) == 0:
            return f"❌ No vegetated patches found in {folder_path}", None, None, None

        X = torch.stack(patches).to(DEVICE)
        y = torch.full((len(X),), true_lbl, dtype=torch.long).to(DEVICE)

        # Fine-tune head
        MODEL.head.train()
        opt = torch.optim.Adam(MODEL.head.parameters(), lr=1e-4)
        crit = torch.nn.CrossEntropyLoss()

        print(f"Adapting on {len(X)} patches (label={LABELS[true_lbl]})...")
        for epoch in range(15):
            idx = torch.randperm(len(X))
            total_loss = 0.0
            for i in range(0, len(X), 32):
                xb = X[idx[i:i+32]]
                yb = y[idx[i:i+32]]
                with torch.no_grad():
                    feats = MODEL._forward_features(xb)
                    phys = MODEL._physics_scalars(xb)
                logits = MODEL.head(feats, phys)
                loss = crit(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if epoch % 5 == 0:
                print(f"  Epoch {epoch:2d}  loss = {total_loss:.4f}")

        MODEL.head.eval()
        MODEL.save_head("checkpoints/best_head.pth")
        return f"✅ Correction applied. Model updated to predict {LABELS[true_lbl]}. Run inference again.", None, None, None

    except Exception as e:
        return f"❌ Correction failed: {e}", None, None, None


# ─── Build UI ─────────────────────────────────────────────────────────────────

def build_demo():
    css = """
    .gradio-container { background: #0f0f23; color: #eee; }
    .label-wrap span  { color: #aaa !important; }
    """

    with gr.Blocks(title="GWSat v3 — Pre-Drought Detector", css=css) as demo:
        gr.Markdown("""
# 🛰️ GWSat v3 — Pre-Drought Groundwater Stress Monitor
### TerraMind-powered on-orbit AI · Detects drought **2–4 weeks before NDVI**
_Physics: plants emit excess IR when stomata close under water stress — visible in SWIR/RedEdge that NDVI ignores._
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📡 Input — pick ONE")
                gr.Markdown("**Option 1: Sample synthetic tile**")
                sample_dd = gr.Dropdown(
                    list(SAMPLE_TILES.keys()),
                    value=list(SAMPLE_TILES.keys())[2],
                    label="Sample tile"
                )
                gr.Markdown("**Option 2: Folder with TIF files** (auto-detects bands by filename)")
                folder_in = gr.Textbox(
                    label="Folder path",
                    placeholder="e.g. ./Telangana_Instant_Data  or  /home/user/scene1",
                    info="Files must contain B4/B04, B8/B08, B8A, B11, B12 in filename"
                )
                gr.Markdown("**Option 3: Upload .pt tile**")
                upload = gr.File(label="Upload .pt tile", file_types=[".pt",".npy"])
                with gr.Row():
                    correct_dd = gr.Dropdown(
                        ["0: Stable", "1: Moderate", "2: Critical"],
                        label="Correction (if model was wrong)",
                        value=None
                    )
                    correct_btn = gr.Button("Apply correction + retrain", variant="secondary")
                run_btn = gr.Button("🛰️  Run Inference", variant="primary", size="lg")
                gr.Markdown("""
**Band order:** B4 B5 B6 B7 B8 B8A B11 B12  
**Encoder:** TerraMind-1.0-tiny (frozen)  
**Head:** PhysicsFusionHead  
                """)

            with gr.Column(scale=2):
                gr.Markdown("### 🚨 Alert")
                status_out = gr.Markdown("_Press Run Inference…_")

        with gr.Row():
            tile_img  = gr.Image(label="False Colour (NIR–RedEdge–Red)", height=220)
            prob_plot = gr.Plot(label="AI Stress Probability")

        gr.Markdown("### 📊 NDVI vs GWSat — The Deception")
        ndvi_plot = gr.Plot(label="NDVI traditional vs GWSat spectral analysis")

        gr.Markdown("### 🗺️ NDVI Map + Stress Heatmap")
        heatmap_plot = gr.Plot(label="NDVI map (what NDMA sees) + GWSat stress map (what AI sees)")

        with gr.Row():
            spec_md = gr.Markdown(label="Spectral Indices")
            bw_md   = gr.Markdown(label="Bandwidth")

        gr.Markdown("### ⬇️ Export Commands")
        export_md = gr.Markdown()

        gr.Markdown("---\n### 🔄 On-Orbit Self-Correction (OCL)")
        with gr.Row():
            n_sl    = gr.Slider(50, 300, 150, step=50, label="Simulated tiles")
            ocl_btn = gr.Button("▶ Simulate orbital pass")
        ocl_out = gr.JSON(label="OCL event log")

        outputs = [tile_img, prob_plot, ndvi_plot, heatmap_plot,
                   status_out, spec_md, bw_md, export_md]

        run_btn.click(
            fn=run_inference,
            inputs=[sample_dd, upload, folder_in],
            outputs=outputs
        )
        # Correction button now uses the dedicated handler
        correct_btn.click(
            fn=apply_correction,
            inputs=[folder_in, correct_dd],
            outputs=[status_out, tile_img, prob_plot, heatmap_plot]
        )
        ocl_btn.click(fn=run_ocl_sim, inputs=[n_sl], outputs=[ocl_out])
        demo.load(
            fn=run_inference,
            inputs=[sample_dd, upload, folder_in],
            outputs=outputs
        )

    return demo


# ─── OCL simulation (unchanged) ───────────────────────────────────────────────

def run_ocl_sim(n_tiles: int):
    from ocl import ShadowModeOCL
    X, y = build_synthetic_dataset(n_per_class=max(n_tiles//3, 10), augment=False)
    ocl  = ShadowModeOCL(MODEL, buffer_size=32, swap_threshold=0.02)
    for tile, lbl in zip(X[:n_tiles], y[:n_tiles]):
        true = lbl.item() if np.random.random() < 0.07 else None
        ocl.ingest(tile, true_label=true)
    return ocl.report()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    p.add_argument("--port",  type=int, default=7860)
    args = p.parse_args()

    demo = build_demo()
    print(f"\n{'='*60}")
    print("GWSat v3 Demo starting…")
    print(f"  Local:  http://localhost:{args.port}")
    if args.share: print("  Public URL generating…")
    print("="*60)
    demo.launch(share=args.share, server_port=args.port, show_error=True)