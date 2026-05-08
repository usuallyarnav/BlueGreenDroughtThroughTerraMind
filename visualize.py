"""
visualize.py — GWSat v3
------------------------
Standalone visualization script. Run AFTER tif_to_pt.py to produce
publication-quality charts from your Telangana scene.

Outputs (all saved in ./plots/):
  1. false_colour.png         — NIR-RedEdge-Red composite
  2. ndvi_map.png             — traditional NDVI (the misleading one)
  3. lswi_heatmap.png         — leaf water stress index
  4. swir_ratio.png           — SWIR ratio (stomata proxy)
  5. stress_heatmap.png       — AI model stress score (if checkpoint exists)
  6. spectral_profile.png     — mean band reflectance profile
  7. index_comparison.png     — NDVI vs LSWI vs RedEdge side-by-side
  8. patch_distribution.png   — patch-level stress class distribution

Usage:
    # Convert TIFs first (once):
    python tif_to_pt.py --folder ./Telangana_Instant_Data --out scene.pt

    # Then visualize:
    python visualize.py --pt scene.pt
    python visualize.py --folder ./Telangana_Instant_Data   # skips tif_to_pt step
    python visualize.py --pt scene.pt --no_ai              # skip model inference
"""

import argparse
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    print("ERROR: pip install matplotlib")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))

PLOT_DIR = Path("plots")
LABELS   = ["Stable", "Moderate", "Critical"]
COLORS   = ["#2ecc71", "#f39c12", "#e74c3c"]
BAND_NAMES = ["B4\n(Red)", "B5\n(RedEdge1)", "B6\n(RedEdge2)",
              "B7\n(RedEdge3)", "B8\n(NIR)", "B8A\n(NIR2)",
              "B11\n(SWIR1)", "B12\n(SWIR2)"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    PLOT_DIR.mkdir(exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {path}")


def _dark_ax(ax):
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#555')
    ax.spines['left'].set_color('#555')


def _colorbar(ax, im, label=""):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb  = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(colors='white', labelsize=7)
    cb.set_label(label, color='white', fontsize=8)
    cax.yaxis.set_tick_params(color='white')


# ─── Individual plots ─────────────────────────────────────────────────────────

def plot_false_colour(scene: torch.Tensor):
    """NIR(B8)→R, RedEdge(B6)→G, Red(B4)→B"""
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#111")
    rgb = np.stack([scene[4].numpy(), scene[2].numpy(), scene[0].numpy()], axis=2)
    rgb = np.clip(rgb * 2.5, 0, 1)
    ax.imshow(rgb)
    ax.set_title("False Colour (NIR-RedEdge-Red)\nHealthy veg=red · stressed=orange/yellow",
                 color='white', fontsize=11)
    ax.axis('off')
    _save(fig, "false_colour.png")


def plot_ndvi_map(scene: torch.Tensor):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111")
    fig.suptitle("NDVI — Traditional Greenness Index", color='white', fontsize=12)

    b4 = scene[0].numpy(); b8 = scene[4].numpy()
    ndvi = (b8 - b4) / (b8 + b4 + 1e-8)

    ax = axes[0]
    im = ax.imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    ax.set_title("NDVI Map", color='white', fontsize=10); ax.axis('off')
    _colorbar(ax, im, "NDVI")

    ax = axes[1]
    ax.set_facecolor("#1a1a2e")
    ax.hist(ndvi.flatten(), bins=80, color="#3498db", edgecolor='none', alpha=0.8)
    ax.axvline(ndvi.mean(), color='white', ls='--', lw=1.5,
               label=f"Mean={ndvi.mean():.3f}")
    ax.axvline(0.4, color='#e74c3c', ls=':', lw=1.5,
               label="0.4 = 'healthy' threshold")
    ax.axvline(0.2, color='#f39c12', ls=':', lw=1.5,
               label="0.2 = 'stressed' threshold")
    ax.set_xlabel("NDVI value", color='white'); ax.set_ylabel("Pixel count", color='white')
    ax.set_title("NDVI Distribution", color='white', fontsize=10)
    ax.tick_params(colors='white')
    leg = ax.legend(facecolor='#333', edgecolor='none', labelcolor='white', fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#555'); ax.spines['left'].set_color('#555')
    _save(fig, "ndvi_map.png")


def plot_lswi_heatmap(scene: torch.Tensor):
    b8  = scene[4].numpy()
    b11 = scene[6].numpy()
    b12 = scene[7].numpy()
    lswi = (b8 - b11) / (b8 + b11 + 1e-8)
    swir = b11 / (b12 + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111")
    fig.suptitle("Leaf Water & SWIR — The Bands NDVI Ignores",
                 color='white', fontsize=12, fontweight='bold')

    ax = axes[0]
    cmap_lswi = mcolors.LinearSegmentedColormap.from_list(
        "lswi", ["#e74c3c", "#f39c12", "#2ecc71"])
    im = ax.imshow(lswi, cmap=cmap_lswi, vmin=-0.2, vmax=0.6)
    ax.set_title("LSWI — Leaf Water Stress\nRed=water deficit · Green=adequate",
                 color='white', fontsize=9); ax.axis('off')
    _colorbar(ax, im, "LSWI")

    ax = axes[1]
    cmap_swir = mcolors.LinearSegmentedColormap.from_list(
        "swir", ["#2ecc71", "#f39c12", "#e74c3c"])
    im2 = ax.imshow(swir, cmap=cmap_swir, vmin=0.5, vmax=2.5)
    ax.set_title("SWIR Ratio (B11/B12) — Stomata Proxy\nHigh=stomata closing",
                 color='white', fontsize=9); ax.axis('off')
    _colorbar(ax, im2, "SWIR ratio")
    _save(fig, "lswi_heatmap.png")


def plot_spectral_profile(scene: torch.Tensor):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1a1a2e")
    _dark_ax(ax)

    wavelengths = [665, 705, 740, 783, 842, 865, 1610, 2190]  # nm
    means = [scene[i].mean().item() for i in range(8)]
    stds  = [scene[i].std().item()  for i in range(8)]

    ax.plot(wavelengths, means, 'o-', color='#3498db', lw=2, ms=8, label='Mean reflectance')
    ax.fill_between(wavelengths,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.25, color='#3498db', label='±1 std')

    # Highlight diagnostic bands
    for wl, label, color in [
        (842, "B8 NIR\n(NDVI)", "#2ecc71"),
        (1610, "B11 SWIR1\n(LSWI)", "#e74c3c"),
        (2190, "B12 SWIR2\n(stomata)", "#e74c3c"),
        (740,  "B6 RedEdge\n(chlorophyll)", "#9b59b6"),
    ]:
        ax.axvline(wl, color=color, ls='--', lw=1.2, alpha=0.7)
        ax.text(wl, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 0.05,
                label, color=color, fontsize=7, ha='center', va='top')

    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Reflectance [0–1]")
    ax.set_title("Mean Spectral Profile — Scene Average", fontsize=11)
    leg = ax.legend(facecolor='#333', edgecolor='none', labelcolor='white')
    xticks = wavelengths
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(w) for w in xticks], rotation=45, fontsize=8)
    _save(fig, "spectral_profile.png")


def plot_index_comparison(scene: torch.Tensor):
    """Show 4 indices together so you can see the divergence."""
    b4  = scene[0]; b5 = scene[1]; b6 = scene[2]
    b8  = scene[4]; b11 = scene[6]; b12 = scene[7]
    eps = 1e-8

    ndvi = ((b8 - b4)  / (b8  + b4  + eps)).numpy()
    rei  = ((b6 - b5)  / (b6  + b5  + eps)).numpy()
    lswi = ((b8 - b11) / (b8  + b11 + eps)).numpy()
    irp  = ((b11 / (b12 + eps)) - 1.0) * (1.0 - np.clip(lswi, 0, 1))

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), facecolor="#111")
    fig.suptitle("Spectral Index Comparison — Why NDVI Is Misleading",
                 color='white', fontsize=13, fontweight='bold')

    panels = [
        (ndvi, "RdYlGn", -0.2, 0.8, "NDVI\n(traditional greenness)", None),
        (rei,  "PuOr",   -0.2, 0.2, "Red-Edge Index\n(chlorophyll degradation)", None),
        (lswi, "RdYlGn", -0.3, 0.6, "LSWI\n(leaf water direct)", None),
        (irp,  "hot_r",   0.0, 0.5, "IR Pressure Index\n(stomata closing)", None),
    ]

    for col, (arr, cmap, vmin, vmax, title, _) in enumerate(panels):
        ax = axes[0, col]
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, color='white', fontsize=8, pad=4)
        ax.axis('off')
        _colorbar(ax, im, "")

        ax2 = axes[1, col]
        ax2.set_facecolor("#1a1a2e")
        ax2.hist(arr.flatten(), bins=60, color=['#3498db','#9b59b6','#1abc9c','#e74c3c'][col],
                 edgecolor='none', alpha=0.85, density=True)
        ax2.axvline(arr.mean(), color='white', ls='--', lw=1.5,
                    label=f"μ={arr.mean():.3f}")
        ax2.set_xlabel("Value", color='white', fontsize=7)
        ax2.set_ylabel("Density", color='white', fontsize=7)
        ax2.tick_params(colors='white', labelsize=6)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_color('#555'); ax2.spines['left'].set_color('#555')
        leg = ax2.legend(facecolor='#333', edgecolor='none', labelcolor='white', fontsize=7)

    plt.tight_layout()
    _save(fig, "index_comparison.png")


def plot_ai_stress_heatmap(scene: torch.Tensor, model):
    """Run tiled inference and show model stress heatmap vs NDVI."""
    print("  Running tiled AI inference for heatmap…")
    scene_d = scene.to(model.device)
    result  = model.predict_scene(scene_d, patch_size=64, stride=32)
    heatmap = result.get("heatmap")
    cls     = result["stress_class"]
    conf    = result["confidence"]
    pd_dist = result.get("patch_distribution", {})

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#111")
    fig.suptitle(
        f"GWSat AI Analysis — {LABELS[cls].upper()} ({conf:.1%} confidence)",
        color=COLORS[cls], fontsize=13, fontweight='bold')

    # False colour
    ax = axes[0]
    rgb = np.stack([scene[4].numpy(), scene[2].numpy(), scene[0].numpy()], axis=2)
    ax.imshow(np.clip(rgb * 2.5, 0, 1))
    ax.set_title("False Colour", color='white', fontsize=10); ax.axis('off')

    # NDVI (the misleading one)
    ax = axes[1]
    b4 = scene[0].numpy(); b8 = scene[4].numpy()
    ndvi = (b8 - b4) / (b8 + b4 + 1e-8)
    im1 = ax.imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    ax.set_title(f"NDVI (traditional)\nmean={ndvi.mean():.3f} → looks {'OK' if ndvi.mean()>0.3 else 'stressed'}",
                 color='white', fontsize=9); ax.axis('off')
    _colorbar(ax, im1, "NDVI")

    # AI stress heatmap
    ax = axes[2]
    if heatmap is not None:
        hm = heatmap.numpy()
        cmap_s = mcolors.LinearSegmentedColormap.from_list(
            "stress", ["#2ecc71", "#f39c12", "#e74c3c"])
        im2 = ax.imshow(hm, cmap=cmap_s, vmin=0, vmax=1)
        ax.set_title(
            f"GWSat Stress Heatmap\n"
            f"S:{pd_dist.get('Stable',0)} M:{pd_dist.get('Moderate',0)} C:{pd_dist.get('Critical',0)} patches",
            color='white', fontsize=9)
        ax.axis('off')
        _colorbar(ax, im2, "Stress score")

    plt.tight_layout()
    _save(fig, "stress_heatmap.png")
    print(f"  AI verdict: {LABELS[cls]} ({conf:.1%})")
    return result


def plot_patch_distribution(scene: torch.Tensor, model):
    """Run tiled inference, show per-patch class distribution."""
    scene_d = scene.to(model.device)
    result  = model.predict_scene(scene_d, patch_size=64, stride=64)
    pd_dist = result.get("patch_distribution", {})
    probs   = result["probabilities"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1a1a2e")
    fig.suptitle("Patch-Level Stress Distribution", color='white', fontsize=12)

    # Pie chart
    sizes = [pd_dist.get(l, 0) for l in LABELS]
    total = sum(sizes) or 1
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=LABELS, colors=COLORS, autopct='%1.0f%%',
        startangle=90, textprops={'color': 'white', 'fontsize': 10}
    )
    for at in autotexts: at.set_color('white')
    ax1.set_facecolor("#1a1a2e")
    ax1.set_title(f"Patch classes\n(total: {sum(sizes)} patches)", color='white')

    # Probability bars
    _dark_ax(ax2)
    vals = list(probs.values())
    bars = ax2.barh(LABELS, vals, color=COLORS, height=0.5)
    for bar, v in zip(bars, vals):
        ax2.text(min(v + 0.02, 0.98), bar.get_y() + bar.get_height()/2,
                 f"{v:.1%}", va='center', color='white', fontsize=10)
    ax2.set_xlim(0, 1.15)
    ax2.set_xlabel("Mean probability")
    ax2.set_title("Scene-averaged probabilities", fontsize=10)

    plt.tight_layout()
    _save(fig, "patch_distribution.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pt",     help=".pt file from tif_to_pt.py")
    grp.add_argument("--folder", help="Folder with TIF files (auto-converts)")
    p.add_argument("--head",   default="checkpoints/best_head.pth")
    p.add_argument("--no_ai",  action="store_true",
                   help="Skip AI inference (just spectral charts)")
    args = p.parse_args()

    # ── Load scene ──
    if args.folder:
        from tif_to_pt import auto_detect_bands, tifs_to_tensor
        print(f"Auto-detecting bands in {args.folder}…")
        band_paths = auto_detect_bands(args.folder)
        scene = tifs_to_tensor(band_paths, verbose=True)
    else:
        data  = torch.load(args.pt, weights_only=False)
        scene = data["X"] if isinstance(data, dict) and "X" in data else data
        if scene.ndim == 4: scene = scene[0]

    print(f"\nScene shape: {list(scene.shape)}")
    print(f"Saving plots to ./{PLOT_DIR}/\n")

    # ── Static spectral charts (no model needed) ──
    print("Generating spectral charts…")
    plot_false_colour(scene)
    plot_ndvi_map(scene)
    plot_lswi_heatmap(scene)
    plot_spectral_profile(scene)
    plot_index_comparison(scene)

    # ── AI charts ──
    if not args.no_ai:
        if not Path(args.head).exists():
            print(f"\n⚠️  No checkpoint at {args.head}.")
            print("   Run: python train.py --epochs 40")
            print("   Then re-run visualize.py")
        else:
            from model import GWSatModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model  = GWSatModel(device=device)
            model.load_head(args.head)
            model.eval()
            print("\nGenerating AI heatmap…")
            plot_ai_stress_heatmap(scene, model)
            plot_patch_distribution(scene, model)

    print(f"\n✅ All plots saved to ./{PLOT_DIR}/")
    print(f"   ls {PLOT_DIR}/")


if __name__ == "__main__":
    main()