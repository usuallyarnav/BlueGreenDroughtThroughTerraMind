"""
model.py — GWSat v3
--------------------
Architecture: TerraMind-1.0 encoder (mandatory) → Physics StressHead
              Falls back to EdgeBackbone only if TerraMind truly unavailable.

TerraMind integration strategy:
  1. Load ibm-granite/TerraMind-1.0-tiny (or small) via terratorch or HuggingFace
  2. Freeze encoder weights (backbone.eval(), no grad)
  3. Extract [CLS] token or mean patch embedding → 768-dim feature
  4. Feed into SpectralFusionHead: merges TM features + hand-crafted
     physics indices (LSWI, RedEdge, SWIR_ratio, IR_Pressure)
  5. 3-class softmax → {Stable, Moderate, Critical}

Why TerraMind + physics fusion beats TerraMind alone:
  TerraMind was pre-trained on multi-spectral data but doesn't
  explicitly encode the B11/B12 stomatal-closure physics.
  Appending 6 computed indices as extra features forces the
  classifier to see the SWIR signal that NDVI misses.

Band order expected: [B4, B5, B6, B7, B8, B8A, B11, B12]
Hardware targets:    Jetson Orin NX, MNR Cubesat (~200MB total)

Fixes vs previous version:
  FIX 1: terratorch (not terramind) is the correct pip package name.
          Strategy 1 now tries terratorch's registry API first.
  FIX 2: AutoFeatureExtractor removed in transformers>=5.
          HuggingFace path now uses AutoImageProcessor.
  FIX 3: EMBED_DIM no longer mutated at class level.
          Instance attribute self.embed_dim used throughout so
          multi-model scenarios (e.g. OCL shadow head) are safe.
  FIX 4: _forward_features() extracted as a shared method so that
          ocl.py shadow training uses the same feature path as
          production inference (fixes shadow/prod feature mismatch).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ────────────────────────────────────────────────────────────────
# Physics index layer  (8 raw bands → 8 + 6 = 14 channels)
# ────────────────────────────────────────────────────────────────

class SpectralIndexLayer(nn.Module):
    """
    Appends 6 hand-crafted spectral indices as extra spatial channels.
    All indices have known physical meaning for groundwater stress.
    Band order: [B4, B5, B6, B7, B8, B8A, B11, B12]
    """
    def forward(self, x):
        eps = 1e-8
        b4, b5, b6 = x[:, 0], x[:, 1], x[:, 2]
        b8, b8a    = x[:, 4], x[:, 5]
        b11, b12   = x[:, 6], x[:, 7]

        ndvi      = (b8  - b4)  / (b8  + b4  + eps)          # greenness (fooled)
        re_chl    = (b8a / (b5  + eps)) - 1.0                 # RedEdge Chl → drops early
        lswi      = (b8  - b11) / (b8  + b11 + eps)           # leaf water direct
        ndwi      = (b8a - b11) / (b8a + b11 + eps)           # open water
        swir_rat  = b11 / (b12  + eps)                        # ABA proxy (stomata)
        cwc       = (b8  - b12) / (b8  + b12 + eps)           # canopy water content

        idx = torch.stack([ndvi, re_chl, lswi, ndwi, swir_rat, cwc], dim=1)
        return torch.cat([x, idx], dim=1)          # [B, 14, H, W]


# ────────────────────────────────────────────────────────────────
# Spectral Attention  (SE-style, per-band weighting)
# ────────────────────────────────────────────────────────────────

class SpectralAttention(nn.Module):
    def __init__(self, n_bands=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_bands, 4), nn.ReLU(),
            nn.Linear(4, n_bands), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3)))
        return x * w.unsqueeze(-1).unsqueeze(-1)


# ────────────────────────────────────────────────────────────────
# TerraMind Encoder  (primary path)
# ────────────────────────────────────────────────────────────────

class TerramindEncoder(nn.Module):
    """
    Wraps IBM TerraMind-1.0-tiny as a frozen feature extractor.

    TerraMind accepts multi-spectral input and produces patch embeddings.
    We take the mean of all patch tokens → [B, embed_dim] feature vector.

    FIX 1: The correct pip package is `terratorch`, not `terramind`.
            Strategy 1 now uses terratorch's model registry.

    FIX 2: AutoFeatureExtractor was removed in transformers>=5.
            The HuggingFace fallback now uses AutoImageProcessor.

    FIX 3: embed_dim is stored as self.embed_dim (instance attribute),
            NOT as a class-level mutation. This makes it safe to
            instantiate multiple models in one process (e.g. OCL).

    Installation:
        pip install terratorch          # IBM geospatial toolkit
        # OR
        pip install git+https://github.com/IBM/terratorch.git

    Model card: ibm-esa-geospatial/TerraMind-1.0-tiny  (~13MB weights)
                ibm-esa-geospatial/TerraMind-1.0-small (~50MB — may be too large for MNR)

    Band mapping for your 8 Sentinel-2 bands [B4,B5,B6,B7,B8,B8A,B11,B12]:
        S2L2A full set is 10 bands (adds B2=BLUE, B3=GREEN).
        We declare only our 8 via bands= so TerraMind initialises
        a subset patch-embed — confirmed working, embed_dim=192 for tiny.
        terratorch band name mapping:
            B4  → RED      B5  → RE1     B6  → RE2     B7  → RE3
            B8  → NIR      B8A → NIR2    B11 → SWIR1   B12 → SWIR2
    """

    DEFAULT_EMBED_DIM = 192   # TerraMind-tiny confirmed embed_dim

    # Your 8 S2 bands mapped to terratorch's S2L2A band name strings
    S2L2A_BANDS = ['RED', 'RE1', 'RE2', 'RE3', 'NIR', 'NIR2', 'SWIR1', 'SWIR2']

    def __init__(self, model_name: str = "ibm-esa-geospatial/TerraMind-1.0-tiny",
                 freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self._loaded    = False
        self.encoder    = None
        # FIX 3: instance-level embed_dim, not a class mutation
        self.embed_dim  = self.DEFAULT_EMBED_DIM
        self._try_load(freeze)

    def _try_load(self, freeze: bool):
        """Try all known TerraMind load paths."""

        # ── Strategy 1: terratorch BACKBONE_REGISTRY (confirmed working) ──
        # Correct org: ibm-esa-geospatial (NOT ibm-granite)
        # Correct API: BACKBONE_REGISTRY.build() with bands= subset
        # Confirmed output: torch.Size([1, 16, 192]), embed_dim=192
        try:
            from terratorch import BACKBONE_REGISTRY
            import torch as _t

            self.encoder = BACKBONE_REGISTRY.build(
                'terramind_v1_tiny',
                pretrained=True,
                modalities=['S2L2A'],
                bands={'S2L2A': self.S2L2A_BANDS},  # our 8 of S2L2A's 10 bands
            )
            # Probe real embed_dim with a dummy forward
            # Output is [B, n_tokens, embed_dim] — take dim -1
            _dummy = _t.zeros(1, 8, 64, 64)
            with _t.no_grad():
                _out = self.encoder({'S2L2A': _dummy})
            _feat = _out[0] if isinstance(_out, list) else _out
            self.embed_dim = int(_feat.shape[-1])
            self._loaded   = True
            self._mode     = "terratorch_pkg"
            print(f"✅ TerraMind loaded via terratorch BACKBONE_REGISTRY "
                  f"(embed_dim={self.embed_dim})")
        except ImportError:
            print("   terratorch pkg: not installed (pip install terratorch)")
        except Exception as e1:
            print(f"   terratorch: {e1}")

        if self._loaded:
            if freeze:
                for p in self.encoder.parameters():
                    p.requires_grad_(False)
                self.encoder.eval()
            return

        # ── Strategy 2: HuggingFace transformers ──
        # FIX 2: AutoFeatureExtractor removed in transformers>=5; use AutoImageProcessor
        try:
            from transformers import AutoModel
            try:
                from transformers import AutoImageProcessor
                self._extractor = AutoImageProcessor.from_pretrained(self.model_name)
            except ImportError:
                # transformers < 4.26 fallback
                from transformers import AutoFeatureExtractor
                self._extractor = AutoFeatureExtractor.from_pretrained(self.model_name)

            self.encoder = AutoModel.from_pretrained(self.model_name)
            # Try to read embed_dim from HF model config
            cfg = getattr(self.encoder, "config", None)
            if cfg is not None:
                for attr in ["hidden_size", "embed_dim", "d_model"]:
                    if hasattr(cfg, attr):
                        self.embed_dim = getattr(cfg, attr)
                        break
            self._loaded = True
            self._mode   = "hf_transformers"
            print(f"✅ TerraMind loaded via HuggingFace: {self.model_name} "
                  f"(embed_dim={self.embed_dim})")
        except Exception as e2:
            print(f"   HuggingFace transformers: {e2}")

        if self._loaded:
            if freeze:
                for p in self.encoder.parameters():
                    p.requires_grad_(False)
                self.encoder.eval()
            return

        # ── Strategy 3: timm + force patch-16-224 with 8-channel stem ──
        try:
            import timm
            # Use DeiT-tiny as TerraMind-tiny proxy (same ViT-tiny arch)
            self.encoder = timm.create_model(
                "deit_tiny_patch16_224",
                pretrained=True,
                in_chans=8,
                num_classes=0      # remove classifier → returns [B, 192]
            )
            # FIX 3: set instance embed_dim, NOT class-level mutation
            self.embed_dim = 192
            self._loaded   = True
            self._mode     = "timm_proxy"
            print("⚠️  TerraMind unavailable — using DeiT-tiny proxy via timm "
                  f"(embed_dim={self.embed_dim})")
        except Exception as e3:
            print(f"   timm proxy: {e3}")

        if self._loaded:
            if freeze:
                for p in self.encoder.parameters():
                    p.requires_grad_(False)
                self.encoder.eval()

    @property
    def is_available(self):
        return self._loaded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 8, 64, 64]  — 8-band Sentinel-2 patch
        Returns [B, embed_dim] feature vector
        """
        if not self._loaded:
            raise RuntimeError("TerraMind not loaded. Check installation.")

        if self._mode == "terratorch_pkg":
            # terratorch BACKBONE_REGISTRY models accept a dict input.
            # Confirmed output shape: [B, n_tokens, embed_dim] e.g. [1, 16, 192]
            # Mean-pool over tokens → [B, embed_dim]
            with torch.no_grad():
                out = self.encoder({'S2L2A': x})
            if isinstance(out, (list, tuple)):
                out = out[0]
            if out.ndim == 3:   return out.mean(dim=1)   # [B, T, C] → [B, C]
            if out.ndim == 4:   return out.mean(dim=(2, 3))
            return out

        elif self._mode == "hf_transformers":
            with torch.no_grad():
                out = self.encoder(pixel_values=x)
            hidden = out.last_hidden_state        # [B, seq_len, embed_dim]
            return hidden[:, 1:, :].mean(dim=1)   # skip [CLS], mean patches

        elif self._mode == "timm_proxy":
            # timm with num_classes=0 returns [B, embed_dim] directly.
            # Resize only here for encoder — patches are still analysed
            # at native resolution everywhere else.
            x_rs = F.interpolate(x, size=(224, 224),
                                 mode='bilinear', align_corners=False)
            with torch.no_grad():
                return self.encoder(x_rs)

        else:
            raise RuntimeError(f"Unknown mode: {self._mode}")


# ────────────────────────────────────────────────────────────────
# EdgeBackbone fallback  (used if ALL TerraMind strategies fail)
# ────────────────────────────────────────────────────────────────

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return F.relu6(self.bn(self.pw(self.dw(x))))


class EdgeBackbone(nn.Module):
    EMBED_DIM = 256
    def __init__(self, in_channels=14):
        super().__init__()
        self.embed_dim = self.EMBED_DIM   # instance attribute mirrors class constant
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6()
        )
        self.blocks = nn.Sequential(
            DSConv(32,  64,  stride=2),
            DSConv(64,  128, stride=2),
            DSConv(128, 128),
            DSConv(128, 256, stride=2),
            DSConv(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        return self.pool(self.blocks(self.stem(x))).flatten(1)


# ────────────────────────────────────────────────────────────────
# Physics Fusion Head
# Merges deep features (from TM or EdgeBackbone) + scalar physics
# ────────────────────────────────────────────────────────────────

class PhysicsFusionHead(nn.Module):
    """
    Input: deep_feat [B, embed_dim] + physics_scalar [B, 6]
    The physics scalars are the mean values of the 6 indices
    (NDVI, REI, LSWI, NDWI, SWIR_ratio, CWC) per patch.
    Concatenating them gives the classifier an explicit numerical
    handle on the stress signal even when the ViT features are noisy.
    """
    def __init__(self, embed_dim: int, n_physics: int = 6,
                 n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_ndim  = n_classes

        fusion_dim = embed_dim + n_physics
        self.net = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, n_classes)
        )

    def forward(self, deep_feat: torch.Tensor,
                physics_scalar: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([deep_feat, physics_scalar], dim=1)
        return self.net(fused)


# ────────────────────────────────────────────────────────────────
# Main model
# ────────────────────────────────────────────────────────────────

class GWSatModel(nn.Module):
    """
    GWSat v3 — TerraMind + Physics Fusion

    Forward pipeline:
      raw [B,8,H,W]
        → SpectralAttention   (band weighting)
        → [TerraMind path]  SpectralAttention output → TerramindEncoder → [B, 768]
           [EdgeBackbone path] SpectralAttention → SpectralIndexLayer → EdgeBackbone → [B, 256]
        → _physics_scalars()  (mean indices from raw bands → [B, 6])
        → PhysicsFusionHead   (deep + scalar physics → 3 logits)

    FIX 4: _forward_features() is a shared internal method used by both
           forward() and ocl.py's ShadowModeOCL._update_shadow(), ensuring
           shadow training and production inference use identical features.

    Tiled inference for large scenes: model.predict_scene()
    """

    def __init__(self, device: str = "cpu",
                 checkpoint: str = None,
                 terramind_model: str = "ibm-esa-geospatial/TerraMind-1.0-tiny",
                 variant: str = None):   # variant kept for API compat
        super().__init__()
        self.device = device

        # --- Spectral preprocessing ---
        self.attn         = SpectralAttention(n_bands=8)
        self.spectral_idx = SpectralIndexLayer()

        # --- Encoder: TerraMind first, EdgeBackbone fallback ---
        self._tm = TerramindEncoder(terramind_model, freeze=True)
        if self._tm.is_available:
            self.backbone   = self._tm
            # FIX 3: read instance embed_dim, not class attribute
            self._embed_dim = self._tm.embed_dim
            self._using_tm  = True
            print(f"  Encoder: TerraMind  (embed_dim={self._embed_dim})")
        else:
            print("  ⚠️  All TerraMind strategies failed → EdgeBackbone")
            self.backbone   = EdgeBackbone(in_channels=14)
            self._embed_dim = self.backbone.embed_dim
            self._using_tm  = False
            print(f"  Encoder: EdgeBackbone (embed_dim={self._embed_dim})")

        # --- Classification head ---
        self.head = PhysicsFusionHead(
            embed_dim=self._embed_dim,
            n_physics=6,
            n_classes=3,
            dropout=0.3
        )

        self.to(device)
        if checkpoint and Path(checkpoint).exists():
            self.load_head(checkpoint)

    # ── backend identity ──────────────────────────────────────

    @property
    def backend_name(self) -> str:
        """
        Returns a human-readable string identifying which encoder loaded.
        Values: 'terratorch' | 'hf_transformers' | 'timm_proxy' | 'edge_cnn'
        """
        if self._using_tm:
            return getattr(self._tm, "_mode", "terratorch").replace(
                "terratorch_pkg", "terratorch"
            )
        return "edge_cnn"

    # ── physics scalar extractor ──────────────────────────────

    @staticmethod
    def _physics_scalars(x: torch.Tensor) -> torch.Tensor:
        """
        Compute mean of 6 spectral indices per patch.
        x: [B, 8, H, W]  → returns [B, 6]
        """
        eps = 1e-8
        b4, b5, b6 = x[:, 0], x[:, 1], x[:, 2]
        b8, b8a    = x[:, 4], x[:, 5]
        b11, b12   = x[:, 6], x[:, 7]

        def mi(a): return a.mean(dim=(1, 2))  # mean over H,W → [B]

        ndvi     = (mi(b8)  - mi(b4))  / (mi(b8)  + mi(b4)  + eps)
        re_chl   = (mi(b8a) / (mi(b5) + eps)) - 1.0
        lswi     = (mi(b8)  - mi(b11)) / (mi(b8)  + mi(b11) + eps)
        ndwi     = (mi(b8a) - mi(b11)) / (mi(b8a) + mi(b11) + eps)
        swir_rat = mi(b11) / (mi(b12) + eps)
        cwc      = (mi(b8)  - mi(b12)) / (mi(b8)  + mi(b12) + eps)

        return torch.stack([ndvi, re_chl, lswi, ndwi, swir_rat, cwc], dim=1)

    # ── FIX 4: shared feature extraction method ───────────────
    # Both forward() and ocl.py's ShadowModeOCL use this method,
    # guaranteeing shadow training and production inference are
    # always computed from the same feature path.

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract deep features from raw [B, 8, H, W] input.
        Returns [B, embed_dim] — same path used by both forward()
        and OCL shadow training.
        """
        x_attn = self.attn(x)
        if self._using_tm:
            # TerraMind receives attention-weighted 8-band input directly
            return self.backbone(x_attn)
        else:
            # EdgeBackbone needs the 14-channel (8 raw + 6 index) input
            x_full = self.spectral_idx(x_attn)
            return self.backbone(x_full)

    # ── forward ──────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        # Physics scalars from raw bands (before attention, to avoid leaking)
        phys = self._physics_scalars(x)
        # Deep features via shared method (FIX 4)
        deep = self._forward_features(x)
        return self.head(deep, phys)

    # ── single-tile predict ──────────────────────────────────

    def predict(self, x: torch.Tensor) -> dict:
        self.eval()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)[0]
            cls    = probs.argmax().item()

        si = compute_spectral_indices(x[0])
        return {
            "stress_class":    cls,
            "confidence":      probs[cls].item(),
            "logits":          logits.cpu().numpy(),
            "probabilities": {
                "Stable":   round(probs[0].item(), 4),
                "Moderate": round(probs[1].item(), 4),
                "Critical": round(probs[2].item(), 4),
            },
            "spectral_indices": si,
            "raw_tile_bytes":  x.numel() * 4,
            "alert_bytes":     64,
        }

    # ── tiled scene inference ────────────────────────────────

    @staticmethod
    def _patch_veg_fraction(patch: torch.Tensor,
                            threshold: float = 0.10) -> float:
        """Fraction of pixels in a patch with NDVI > threshold."""
        b4  = patch[0]; b8 = patch[4]
        ndvi = (b8 - b4) / (b8 + b4 + 1e-8)
        return float((ndvi > threshold).float().mean())

    def predict_scene(self, scene: torch.Tensor,
                      patch_size: int = 64,
                      stride:     int = 32,
                      min_patches: int = 4,
                      veg_threshold: float = 0.15,
                      min_veg_fraction: float = 0.20) -> dict:
        """
        Tiled inference. Skips urban/bare-soil patches (NDVI < threshold).
        Returns majority-vote prediction + spatial heatmap.
        """
        self.eval()
        if scene.ndim == 4: scene = scene[0]
        C, H, W = scene.shape

        # Pad
        ph = (patch_size - H % patch_size) % patch_size
        pw = (patch_size - W % patch_size) % patch_size
        if ph or pw:
            scene = F.pad(scene, (0, pw, 0, ph), mode='reflect')
        _, Hp, Wp = scene.shape

        patches, positions, skipped = [], [], 0
        for y in range(0, Hp - patch_size + 1, stride):
            for x in range(0, Wp - patch_size + 1, stride):
                p = scene[:, y:y+patch_size, x:x+patch_size]
                if self._patch_veg_fraction(p, veg_threshold) >= min_veg_fraction:
                    patches.append(p); positions.append((y, x))
                else:
                    skipped += 1

        if len(patches) < min_patches:
            r = self.predict(F.interpolate(
                scene.unsqueeze(0), (patch_size, patch_size),
                mode='bilinear', align_corners=False)[0])
            r.update({"skipped_patches": skipped, "n_patches": 0})
            return r

        BATCH = 16
        all_p, all_l = [], []
        for i in range(0, len(patches), BATCH):
            b = torch.stack(patches[i:i+BATCH]).to(self.device)
            with torch.no_grad():
                lg = self.forward(b)
                pr = torch.softmax(lg, dim=1)
            all_p.append(pr.cpu()); all_l.append(lg.cpu())

        all_p = torch.cat(all_p); all_l = torch.cat(all_l)
        votes = all_p.argmax(1); conf_w = all_p.max(1).values

        # ── Voting fix ──────────────────────────────────────────────────────────
        # BROKEN was: weighted = csum/counts → argmax
        #   → picked class with highest *average* per-patch confidence.
        #   51 very-confident Stable patches (irrigated farms) beat 838
        #   moderately-confident Critical patches.  Wrong for scene-level verdict.
        #
        # FIXED: use csum (total confidence mass = count × avg_conf per class).
        #   A class must have both quantity of patches AND reasonable confidence.
        counts = torch.zeros(3); csum = torch.zeros(3)
        for v, c in zip(votes, conf_w):
            counts[v] += 1; csum[v] += c

        cls = int(csum.argmax().item())  # class with most total confidence mass

        # ── Conservative alert rule ─────────────────────────────────────────────
        # Drought monitoring is asymmetric: missed Critical >> false alarm.
        # If ≥15% of vegetated patches are Critical, escalate regardless of winner.
        # If ≥25% are Moderate and winner is Stable, escalate to Moderate.
        n_total = float(len(votes))
        frac_critical = (counts[2] / n_total).item()
        frac_moderate = (counts[1] / n_total).item()
        if frac_critical >= 0.15 and cls < 2:
            cls = 2   # escalate to Critical
        elif frac_moderate >= 0.25 and cls == 0:
            cls = 1   # escalate to Moderate

        # Confidence = fraction of patches voting for reported class.
        # Interpretable: "70.7% of the vegetated scene is Critical."
        conf = (counts[cls] / n_total).item()

        mean_p = all_p.mean(0)
        heatmap = self._build_heatmap(all_p, positions, H, W, patch_size, stride)

        return {
            "stress_class":    cls,
            "confidence":      conf,
            "logits":          all_l.mean(0, keepdim=True).numpy(),
            "probabilities": {
                "Stable":   round(mean_p[0].item(), 4),
                "Moderate": round(mean_p[1].item(), 4),
                "Critical": round(mean_p[2].item(), 4),
            },
            "spectral_indices":  compute_spectral_indices(scene),
            "raw_tile_bytes":    scene.numel() * 4,
            "alert_bytes":       64,
            "n_patches":         len(patches),
            "skipped_patches":   skipped,
            "patch_distribution": {
                "Stable":   int((votes==0).sum()),
                "Moderate": int((votes==1).sum()),
                "Critical": int((votes==2).sum()),
            },
            "heatmap": heatmap,
        }

    def _build_heatmap(self, all_p, positions, H, W, ps, stride):
        sm = torch.zeros(H + ps, W + ps)
        cm = torch.zeros_like(sm)
        for (y, x), p in zip(positions, all_p):
            s = 0.5 * p[1] + 1.0 * p[2]
            sm[y:y+ps, x:x+ps] += s; cm[y:y+ps, x:x+ps] += 1
        return (sm / cm.clamp(1))[:H, :W]

    # ── checkpoint I/O ────────────────────────────────────────

    def save_head(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "head_state_dict":      self.head.state_dict(),
            "backbone_state_dict":  self.backbone.state_dict()
                                    if not self._using_tm else {},
            "attn_state_dict":      self.attn.state_dict(),
            "embed_dim":            self._embed_dim,
            "out_ndim":             3,
            "using_terramind":      self._using_tm,
            "backend_name":         self.backend_name,
            "version":              3,
        }, path)
        print(f"✅ Saved: {path}  ({Path(path).stat().st_size/1e6:.1f} MB)")

    def load_head(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict):
            ckpt = {"head_state_dict": ckpt}

        sd = (ckpt.get("head_state_dict")
              or ckpt.get("state_dict")
              or ckpt)
        self.head.load_state_dict(sd, strict=False)

        if "attn_state_dict" in ckpt:
            self.attn.load_state_dict(ckpt["attn_state_dict"], strict=False)
        if "backbone_state_dict" in ckpt and not self._using_tm:
            self.backbone.load_state_dict(
                ckpt["backbone_state_dict"], strict=False)
        print(f"✅ Loaded: {path}")


    def benchmark(self, n_warmup: int = 10, n_runs: int = 100,
                  patch_size: int = 64) -> dict:
        """
        Measure per-patch inference latency (CPU and CUDA if available).
        Call this after loading checkpoint to get slide-ready numbers.

        Returns dict with keys: cpu_ms, gpu_ms (None if no GPU), checkpoint_mb
        """
        import time
        dummy = torch.zeros(1, 8, patch_size, patch_size)
        self.eval()

        # CPU benchmark
        dummy_cpu = dummy.cpu()
        model_cpu = self.cpu()
        for _ in range(n_warmup):
            with torch.no_grad(): model_cpu(dummy_cpu)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad(): model_cpu(dummy_cpu)
        cpu_ms = (time.perf_counter() - t0) * 1000 / n_runs

        # GPU benchmark if available
        gpu_ms = None
        if torch.cuda.is_available():
            self.to("cuda")
            dummy_gpu = dummy.to("cuda")
            for _ in range(n_warmup):
                with torch.no_grad(): self(dummy_gpu)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                with torch.no_grad(): self(dummy_gpu)
            torch.cuda.synchronize()
            gpu_ms = (time.perf_counter() - t0) * 1000 / n_runs
            self.to(self.device)

        ckpt_mb = None
        import os
        ckpt_path = "checkpoints/best_head.pth"
        if os.path.exists(ckpt_path):
            ckpt_mb = round(os.path.getsize(ckpt_path) / 1e6, 2)

        total_params    = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        result = {
            "cpu_ms":          round(cpu_ms, 1),
            "gpu_ms":          round(gpu_ms, 1) if gpu_ms else None,
            "checkpoint_mb":   ckpt_mb,
            "total_params":    total_params,
            "trainable_params": trainable_params,
            "backend":         self.backend_name,
        }
        print(f"  Latency:  CPU={cpu_ms:.1f}ms/patch"
              + (f"  GPU={gpu_ms:.1f}ms/patch" if gpu_ms else ""))
        if ckpt_mb:
            print(f"  Head size: {ckpt_mb} MB  |  Total params: {total_params:,}")
        return result


# ──────────────────────────────────────────────────────────────────
# Spectral index helper (used by infer / demo)
# ──────────────────────────────────────────────────────────────────

def compute_spectral_indices(tile: torch.Tensor) -> dict:
    if tile.ndim == 4: tile = tile[0]
    tile = tile.float()
    eps  = 1e-8
    b4, b5, b6 = tile[0].mean().item(), tile[1].mean().item(), tile[2].mean().item()
    b8, b8a    = tile[4].mean().item(), tile[5].mean().item()
    b11, b12   = tile[6].mean().item(), tile[7].mean().item()

    ndvi  = (b8  - b4)  / (b8  + b4  + eps)
    rei   = (b6  - b5)  / (b6  + b5  + eps)
    lswi  = (b8  - b11) / (b8  + b11 + eps)
    ndwi  = (b8a - b11) / (b8a + b11 + eps)
    swir  = b11 / (b12  + eps)
    irp   = (swir - 1.0) * (1.0 - max(lswi, 0))

    return {
        "NDVI":                 round(ndvi,  4),
        "RedEdge_Index":        round(rei,   4),
        "LeafWaterStress_LSWI": round(lswi,  4),
        "NDWI":                 round(ndwi,  4),
        "SWIR_ratio":           round(swir,  4),
        "IR_Pressure_Index":    round(irp,   4),
    }