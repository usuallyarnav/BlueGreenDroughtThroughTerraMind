import os, sys, json, time, math, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from model import GWSatModel

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        
        # GWSat expects [B, 8, 64, 64]
        preds = model(X)
        loss = criterion(preds, y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    """Returns (accuracy, f1_macro, all_preds, all_labels)."""
    from sklearn.metrics import f1_score
    model.eval()
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            _, predicted = torch.max(preds.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = correct / total
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1, all_preds, all_labels

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=3e-4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on: {device}")

    # ─── DATA LOADING (UPDATED FOR REAL DATA) ───
    # Ensure you have run 'python build_real_dataset.py' first!
    if not os.path.exists('data/train.pt'):
        print("❌ Error: data/train.pt not found. Run build_real_dataset.py first.")
        sys.exit(1)

    print("📦 Loading real Sentinel-2 tensors...")
    train_data = torch.load('data/train.pt', weights_only=False)
    val_data   = torch.load('data/val.pt', weights_only=False)

    train_loader = DataLoader(
        TensorDataset(train_data['X'], train_data['y']), 
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_data['X'], val_data['y']), 
        batch_size=args.batch_size
    )

    # ─── MODEL SETUP ───
    model = GWSatModel(device=device)
    # Note: Encoder is frozen by default in model.py
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    
    best_f1 = 0.0
    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Training for {args.epochs} epochs (tracking F1 + Accuracy)...")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc, f1, _, _ = evaluate(model, val_loader, device)

        if f1 > best_f1:
            best_f1  = f1
            best_acc = acc
            model.save_head("checkpoints/best_head.pth")

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:03d}/{args.epochs} | "
                  f"Loss: {loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}"
                  + (" ← best" if f1 == best_f1 else ""))

    # ── Final evaluation on test split ─────────────────────────────────────
    print(f"\n✅ Training Complete.")
    print(f"   Best Val F1:  {best_f1:.4f}")
    print(f"   Best Val Acc: {best_acc:.4f}")

    if os.path.exists("data/test.pt"):
        print("\n── Test set evaluation ────────────────────────────────")
        from sklearn.metrics import classification_report, confusion_matrix
        test_data   = torch.load("data/test.pt", weights_only=False)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_data["X"], test_data["y"]),
            batch_size=args.batch_size
        )
        model.load_head("checkpoints/best_head.pth")
        test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, device)
        print(f"  Test Acc: {test_acc:.4f}   Test F1 (macro): {test_f1:.4f}")
        print(classification_report(test_labels, test_preds,
              target_names=["Stable","Moderate","Critical"], zero_division=0))
        cm = confusion_matrix(test_labels, test_preds, labels=[0,1,2])
        print("  Confusion matrix [rows=true, cols=pred]:")
        for i, row in enumerate(cm):
            print(f"    {['Stable','Moderate','Critical'][i]:<10}", row.tolist())

        ckpt_mb = Path("checkpoints/best_head.pth").stat().st_size / 1e6
        results = {
            "val_f1":       round(float(best_f1),  4),
            "val_acc":      round(float(best_acc), 4),
            "test_f1":      round(float(test_f1),  4),
            "test_acc":     round(float(test_acc), 4),
            "checkpoint_mb": round(ckpt_mb, 2),
            "epochs":       args.epochs,
            "backend":      model.backend_name,
        }
        import json
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  results.json saved: {results}")
    else:
        print("  (No data/test.pt found — run build_real_dataset.py first)")

    print("\nHead saved to: checkpoints/best_head.pth")