import os, sys, json, argparse, warnings
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
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
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
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on: {device}")

    # Load data
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

    # CRITICAL FIX: Class weights to force model to learn Moderate
    # Count samples per class
    counts = torch.bincount(train_data['y'])
    print(f"Class distribution: Stable={counts[0]}, Moderate={counts[1]}, Critical={counts[2]}")
    
    # Higher weight for Moderate (class 1)
    weights = torch.tensor([1.0, 3.0, 1.0], dtype=torch.float).to(device)
    print(f"Class weights: Stable=1.0, Moderate=3.0, Critical=1.0")

    model = GWSatModel(device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    
    best_f1 = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc, f1, _, _ = evaluate(model, val_loader, device)

        if f1 > best_f1:
            best_f1 = f1
            model.save_head("checkpoints/best_head_weighted.pth")
            print(f"  ✅ Saved new best (F1={f1:.4f})")

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:03d}/{args.epochs} | Loss: {loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

    print(f"\n✅ Training Complete. Best Val F1: {best_f1:.4f}")
    
    # Test evaluation
    if os.path.exists("data/test.pt"):
        test_data = torch.load("data/test.pt", weights_only=False)
        test_loader = DataLoader(
            TensorDataset(test_data["X"], test_data["y"]),
            batch_size=args.batch_size
        )
        model.load_head("checkpoints/best_head_weighted.pth")
        test_acc, test_f1, _, _ = evaluate(model, test_loader, device)
        print(f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")