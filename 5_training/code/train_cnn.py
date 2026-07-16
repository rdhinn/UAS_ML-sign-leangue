"""
TA: Tahap 5 - Training CNN
===========================
Arsitektur ringan untuk CPU: 3 Conv layers + 2 FC layers.
Input: 48x48 grayscale, Output: 29 kelas ASL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_DIR = Path("D:/TA Mesin")
PROCESSED_DIR = PROJECT_DIR / "data/processed"
OUTPUT_DIR = PROJECT_DIR / "5_training"
MODELS_DIR = OUTPUT_DIR / "models"
IMG_DIR = OUTPUT_DIR / "images"

BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
PATIENCE = 5


class ASL_CNN(nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 48 -> 24
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 24 -> 12
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.conv_layers(x))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        out = model(Xb)
        loss = criterion(out, yb)
        total_loss += loss.item() * Xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return total_loss / total, correct / total


def main():
    print("=" * 60)
    print("TA - Tahap 5: Training CNN")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("\n[STEP] Loading data...")
    X_train = np.load(str(PROCESSED_DIR / "train_X.npy"))
    y_train = np.load(str(PROCESSED_DIR / "train_y.npy"))
    X_val   = np.load(str(PROCESSED_DIR / "val_X.npy"))
    y_val   = np.load(str(PROCESSED_DIR / "val_y.npy"))
    classes = np.load(str(PROCESSED_DIR / "classes.npy"), allow_pickle=True)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Classes: {len(classes)}")

    # --- DataLoaders ---
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long()),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long()),
        batch_size=BATCH_SIZE
    )

    # --- Model ---
    model = ASL_CNN(num_classes=len(classes)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print(f"\n[STEP] Model: {total_params:,} parameters")

    # --- Training ---
    print(f"\n[STEP] Training ({EPOCHS} epochs max, patience={PATIENCE})...")
    best_acc, best_epoch, stall = 0, 0, 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"  Epoch {epoch:2d}/{EPOCHS} | "
              f"train loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
              f"val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
              f"{elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            stall = 0
            torch.save(model.state_dict(), str(MODELS_DIR / "asl_cnn_best.pth"))
        else:
            stall += 1
            if stall >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    total_time = time.time() - t_start
    print(f"\n[OK] Training selesai dalam {total_time/60:.1f} menit")
    print(f"  Best val accuracy: {best_acc:.4f} (epoch {best_epoch})")

    # --- Save final model ---
    torch.save(model.state_dict(), str(MODELS_DIR / "asl_cnn_final.pth"))
    print(f"  Model saved: {MODELS_DIR}")

    # --- Plot ---
    epochs_range = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs_range, val_losses, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs_range, train_accs, "b-", label="Train Accuracy")
    ax2.plot(epochs_range, val_accs, "r-", label="Val Accuracy")
    ax2.axhline(best_acc, color="g", ls="--", label=f"Best Val ({best_acc:.3f})")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    save_path = IMG_DIR / "training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}")

    print(f"\n{'='*60}")
    print("Tahap 5 - Training CNN SELESAI.")
    print("=" * 60)


if __name__ == "__main__":
    main()
