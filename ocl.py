"""
ocl.py — GWSat v2 On-Orbit Continual Learning
-----------------------------------------------
Safe, memory-bounded continual learning for edge/satellite deployment.

Design constraints for Jetson Nano / MNR:
  - No optimizer state persisted to disk between orbits
  - Hard buffer limit (RAM is scarce on cubesat)
  - Rollback bank stores only state_dicts (not full models)
  - MissionSupervisor enforces monotonic accuracy improvement

FIX 4 (from model.py review):
  ShadowModeOCL._update_shadow() and evaluate_and_swap() previously
  reconstructed the feature path manually, which diverged from
  GWSatModel.forward() when _using_tm=True (TerraMind path skips
  SpectralIndexLayer but the old OCL code still called it).

  Fixed by calling model._forward_features(x) — the single shared
  method that always mirrors whatever forward() does.
"""

import copy, time
import torch
import torch.nn as nn
from collections import deque
from pathlib import Path


class RollbackBank:
    """Stores N previous head state_dicts for safe rollback."""
    def __init__(self, max_checkpoints: int = 3):
        self.bank = deque(maxlen=max_checkpoints)
        self.acc_history = deque(maxlen=max_checkpoints)

    def snapshot(self, model_head, accuracy: float):
        self.bank.append(copy.deepcopy(model_head.state_dict()))
        self.acc_history.append(accuracy)
        print(f"  📸 Snapshot stored (acc={accuracy:.3f}, "
              f"bank depth={len(self.bank)})")

    def rollback(self, model):
        if self.bank:
            model.head.load_state_dict(self.bank.pop())
            acc = self.acc_history.pop() if self.acc_history else "?"
            print(f"  🚨 ROLLBACK executed → restored acc≈{acc}")
            return True
        print("  ⚠️  Rollback bank empty.")
        return False


class MissionSupervisor:
    """
    Gate that decides whether shadow model is flight-ready.
    Only approves swap if improvement > threshold AND
    no catastrophic forgetting on any single class.
    """
    def __init__(self, swap_threshold: float = 0.02,
                 max_class_drop: float = 0.10):
        self.swap_threshold = swap_threshold
        self.max_class_drop = max_class_drop

    def is_safe_to_swap(self, shadow_metrics: dict,
                        prod_metrics: dict) -> tuple[bool, str]:
        delta_f1 = shadow_metrics["f1"] - prod_metrics["f1"]

        if delta_f1 < self.swap_threshold:
            return False, f"Δf1={delta_f1:.4f} < threshold {self.swap_threshold}"

        # Check for catastrophic forgetting per class
        for cls in range(3):
            s_acc = shadow_metrics.get(f"class_{cls}_acc", 1.0)
            p_acc = prod_metrics.get(f"class_{cls}_acc",  1.0)
            if p_acc - s_acc > self.max_class_drop:
                return False, (f"Class {cls} dropped "
                               f"{p_acc-s_acc:.3f} > {self.max_class_drop}")

        return True, f"Approved — Δf1={delta_f1:+.4f}"


class ShadowModeOCL:
    """
    On-orbit shadow training:
      1. Production model serves all inference (frozen).
      2. Shadow head trains on labelled corrections (rare ground-truth).
      3. At orbit boundary, supervisor decides whether to swap.
      4. On degradation, rollback to last known-good checkpoint.

    Memory budget: buffer_size × 8 × 64 × 64 × 4 bytes
                 = 150 × 131072 bytes ≈ 18 MB at buffer_size=150
    """

    def __init__(self, model, buffer_size: int = 64,
                 swap_threshold: float = 0.02):
        self.prod_model  = model
        self.shadow_head = copy.deepcopy(model.head)
        self.supervisor  = MissionSupervisor(swap_threshold=swap_threshold)
        self.bank        = RollbackBank(max_checkpoints=3)
        self.buffer      = deque(maxlen=buffer_size)
        self.optimizer   = torch.optim.Adam(
            self.shadow_head.parameters(), lr=5e-5)
        self.event_log   = []
        self._prod_acc   = 0.0
        self._n_ingested = 0
        self._n_corrected = 0

    def ingest(self, tile: torch.Tensor, true_label: int = None):
        """
        Called once per tile during an orbital pass.
        Returns production-model prediction (shadow trains in background).
        """
        with torch.no_grad():
            result = self.prod_model.predict(tile)

        self._n_ingested += 1

        if true_label is not None:
            self._n_corrected += 1
            self.buffer.append((tile.clone(), torch.tensor(true_label)))
            self._update_shadow()

        return result

    def _update_shadow(self):
        """Fine-tune shadow head on correction buffer (memory-efficient).

        FIX 4: Uses model._forward_features() instead of manually
        reconstructing the backbone path. This guarantees shadow training
        always mirrors production inference exactly, regardless of whether
        TerraMind or EdgeBackbone is active.
        """
        if len(self.buffer) < 4:
            return   # need minimum batch to avoid noisy updates

        self.shadow_head.train()
        # Keep backbone frozen — only shadow_head is updated
        self.prod_model.eval()

        # Sample from buffer
        batch_size = min(len(self.buffer), 16)
        indices    = torch.randperm(len(self.buffer))[:batch_size]
        buf_list   = list(self.buffer)

        tiles  = torch.stack([buf_list[i][0] for i in indices])
        labels = torch.stack([buf_list[i][1] for i in indices])

        tiles  = tiles.to(self.prod_model.device)
        labels = labels.to(self.prod_model.device)

        self.optimizer.zero_grad()

        # FIX 4: Use shared _forward_features() — correct path for both
        # TerraMind (8-band → ViT) and EdgeBackbone (14-band → CNN).
        # The old code called spectral_idx even in TerraMind mode, which
        # produced 14-channel input to a backbone expecting 8 channels.
        with torch.no_grad():
            feats  = self.prod_model._forward_features(tiles)
            phys   = self.prod_model._physics_scalars(tiles)

        logits = self.shadow_head(feats, phys)
        loss   = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(self.shadow_head.parameters(), 1.0)
        self.optimizer.step()

    def evaluate_and_swap(self, val_tiles: torch.Tensor,
                          val_labels: torch.Tensor,
                          save_path: str = None):
        """
        Compare shadow vs production on validation set.
        Swaps if MissionSupervisor approves.

        FIX 4: Uses _forward_features() + _physics_scalars() so that
        eval features are computed identically to training features.
        """
        from sklearn.metrics import f1_score

        self.prod_model.eval()
        self.shadow_head.eval()

        val_tiles = val_tiles.to(self.prod_model.device)

        # Get features once (same frozen backbone for both heads)
        with torch.no_grad():
            feats = self.prod_model._forward_features(val_tiles)
            phys  = self.prod_model._physics_scalars(val_tiles)

            prod_logits   = self.prod_model.head(feats, phys)
            shadow_logits = self.shadow_head(feats, phys)

        prod_preds   = prod_logits.argmax(1).cpu().numpy()
        shadow_preds = shadow_logits.argmax(1).cpu().numpy()
        true_np      = val_labels.numpy()

        def metrics(preds):
            f1 = f1_score(true_np, preds, average="macro", zero_division=0)
            per_class = {}
            for c in range(3):
                mask = true_np == c
                if mask.sum() > 0:
                    per_class[f"class_{c}_acc"] = (preds[mask] == c).mean()
            return {"f1": f1, **per_class}

        prod_m   = metrics(prod_preds)
        shadow_m = metrics(shadow_preds)

        safe, reason = self.supervisor.is_safe_to_swap(shadow_m, prod_m)

        event = {
            "timestamp":    time.time(),
            "prod_f1":      round(prod_m["f1"],   4),
            "shadow_f1":    round(shadow_m["f1"], 4),
            "swap_approved": safe,
            "reason":       reason,
            "tiles_seen":   self._n_ingested,
            "corrections":  self._n_corrected,
        }
        self.event_log.append(event)

        if safe:
            self.bank.snapshot(self.prod_model.head, prod_m["f1"])
            self.prod_model.head.load_state_dict(
                self.shadow_head.state_dict())
            print(f"  ✅ SWAP approved: {reason}")
            if save_path:
                self.prod_model.save_head(save_path)
        else:
            print(f"  ⏸️  Swap rejected: {reason}")

        return event

    def report(self) -> dict:
        return {
            "tiles_ingested": self._n_ingested,
            "corrections":    self._n_corrected,
            "swap_events":    len([e for e in self.event_log
                                   if e["swap_approved"]]),
            "event_log":      self.event_log[-10:],  # last 10 events
            "bank_depth":     len(self.bank.bank),
        }