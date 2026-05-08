"""
train_moderate_fix.py — fixes 0% Moderate recall
--------------------------------------------------
Run this instead of train.py. Key changes:
  - Moderate class weight raised to 5.0 (was 3.0 in train_weighted.py)
  - Focal loss option to further penalise easy Stable/Critical predictions
  - Per-class accuracy printed every 10 epochs so you can see Moderate improving
  - Saves to checkpoints/best_head_moderate_fixed.pth

Usage:
    python train_moderate_fix.py
    python train_moderate_fix.py --epochs 80 --moderate_weight 6.0
"""

import os, sys, json, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from model import GWSatModel


# ── Focal loss — down-weights easy examples, forces model to learn Moderate ──

class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al. 2017) with per-class weights.
    gamma=2 means examples the model is already confident about
    contribute ~4x less to the gradient than hard examples.
    Combined with class weights, this is the strongest tool
    against a model that just learns Stable + Critical and ignores Moderate.
    """
    def __init__(self, weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)                          # confidence in correct class
        loss = ((1 - pt) ** self.gamma) * ce           # down-weight easy examples
        return loss.mean()


# ── Mixup augmentation — creates synthetic Moderate-like blends ──

def mixup_batch(X: torch.Tensor, y: torch.Tensor,
                alpha: float = 0.4) -> tuple:
    """
    Standard mixup. Alpha=0.4 creates strong blends between class pairs.
    Particularly useful here because Moderate IS a blend of Stable + Critical
    physics — mixup literally creates Moderate-like examples from the other two.
    """
    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(len(X))
    X_mix = lam * X + (1 - lam) * X[idx]
    y_a, y_b = y, y[idx]
    return X_mix, y_a, y_b, lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ── Training ──

def train_one_epoch(model, loader, optimizer, criterion,
                    device, use_mixup: bool = True):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        if use_mixup and np.random.random() < 0.5:
            X_m, y_a, y_b, lam = mixup_batch(X, y)
            logits = model(X_m)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            logits = model(X)
            loss   = criterion(logits, y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    from sklearn.metrics import f1_score, classification_report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Per-class accuracy — the number that was 0% for Moderate
    per_class_acc = {}
    for cls, name in enumerate(["Stable", "Moderate", "Critical"]):
        mask = all_labels == cls
        if mask.sum() > 0:
            per_class_acc[name] = (all_preds[mask] == cls).mean()
        else:
            per_class_acc[name] = 0.0

    return acc, f1, per_class_acc, all_preds, all_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",          type=int,   default=80)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=2e-4)
    p.add_argument("--moderate_weight", type=float, default=5.0,
                   help="Loss weight for Moderate class (was 3.0 → set to 5.0 or higher)")
    p.add_argument("--focal_gamma",     type=float, default=2.0,
                   help="Focal loss gamma. 0=standard CE, 2=standard focal")
    p.add_argument("--no_mixup",        action="store_true")
    p.add_argument("--checkpoint_out",  default="checkpoints/best_head_moderate_fixed.pth")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    print(f"Moderate class weight: {args.moderate_weight}")
    print(f"Focal gamma: {args.focal_gamma}")
    print(f"Mixup: {'off' if args.no_mixup else 'on'}")

    if not os.path.exists("data/train.pt"):
        print("data/train.pt not found. Run build_real_dataset.py first.")
        sys.exit(1)

    train_data = torch.load("data/train.pt", weights_only=False)
    val_data   = torch.load("data/val.pt",   weights_only=False)

    counts = torch.bincount(train_data["y"])
    print(f"\nClass distribution: Stable={counts[0]}  Moderate={counts[1]}  Critical={counts[2]}")
    print(f"Class weights:      Stable=1.0  Moderate={args.moderate_weight}  Critical=1.0\n")

    train_loader = DataLoader(
        TensorDataset(train_data["X"], train_data["y"]),
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(val_data["X"], val_data["y"]),
        batch_size=args.batch_size
    )

    model = GWSatModel(device=device)
    print(f"Backend: {model.backend_name}")
    print(f"*** ACTIVE BACKEND: {model.backend_name.upper()} ***\n")

    weights   = torch.tensor([1.0, args.moderate_weight, 1.0]).to(device)
    criterion = FocalLoss(weight=weights, gamma=args.focal_gamma)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # Cosine annealing — helps escape the Stable/Critical local minimum
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    best_moderate_recall = 0.0
    best_f1              = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    print(f"{'Epoch':>6}  {'Loss':>8}  {'F1':>6}  {'Acc':>6}  "
          f"{'Stable%':>8}  {'Moderate%':>10}  {'Critical%':>10}")
    print("-" * 68)

    for epoch in range(args.epochs):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_mixup=not args.no_mixup
        )
        acc, f1, per_cls, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        mod_recall = per_cls["Moderate"]

        # Save on best F1 that also has Moderate recall > 0.2
        # This prevents saving a checkpoint that ignores Moderate entirely
        is_best = (f1 > best_f1 and mod_recall > 0.20) or \
                  (mod_recall > best_moderate_recall and f1 > 0.55)

        if is_best:
            best_f1              = f1
            best_moderate_recall = mod_recall
            model.save_head(args.checkpoint_out)
            marker = " ← saved"
        else:
            marker = ""

        if epoch % 5 == 0 or epoch == args.epochs - 1 or marker:
            print(f"{epoch:>6}  {loss:>8.4f}  {f1:>6.4f}  {acc:>6.4f}  "
                  f"{per_cls['Stable']:>8.1%}  {per_cls['Moderate']:>10.1%}  "
                  f"{per_cls['Critical']:>10.1%}{marker}")

    print(f"\nTraining complete.")
    print(f"Best F1: {best_f1:.4f}  Best Moderate recall: {best_moderate_recall:.1%}")
    print(f"Checkpoint: {args.checkpoint_out}")

    # Final test evaluation
    if os.path.exists("data/test.pt"):
        from sklearn.metrics import classification_report, confusion_matrix
        test_data   = torch.load("data/test.pt", weights_only=False)
        test_loader = DataLoader(
            TensorDataset(test_data["X"], test_data["y"]),
            batch_size=args.batch_size
        )
        model.load_head(args.checkpoint_out)
        test_acc, test_f1, test_per_cls, test_preds, test_labels = evaluate(
            model, test_loader, device)

        print(f"\nTest set results:")
        print(f"  F1 (macro): {test_f1:.4f}   Accuracy: {test_acc:.4f}")
        print(f"  Per-class:  Stable={test_per_cls['Stable']:.1%}  "
              f"Moderate={test_per_cls['Moderate']:.1%}  "
              f"Critical={test_per_cls['Critical']:.1%}")
        print()
        print(classification_report(test_labels, test_preds,
              target_names=["Stable", "Moderate", "Critical"], zero_division=0))

        cm = confusion_matrix(test_labels, test_preds, labels=[0, 1, 2])
        print("Confusion matrix [rows=true, cols=pred]:")
        for i, row in enumerate(cm.tolist()):
            print(f"  {['Stable','Moderate','Critical'][i]:<10}", row)

        results = {
            "test_f1":           round(float(test_f1),  4),
            "test_acc":          round(float(test_acc), 4),
            "moderate_recall":   round(float(test_per_cls["Moderate"]), 4),
            "moderate_weight":   args.moderate_weight,
            "focal_gamma":       args.focal_gamma,
            "backend":           model.backend_name,
            "checkpoint":        args.checkpoint_out,
        }
        with open("results_moderate_fixed.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: results_moderate_fixed.json")

    # Copy to best_head.pth so validate.py / demo_app.py pick it up
    import shutil
    shutil.copy(args.checkpoint_out, "checkpoints/best_head.pth")
    print(f"Copied to checkpoints/best_head.pth")
    print(f"\nNow run:  python validate.py --no_cv")


if __name__ == "__main__":
    main()