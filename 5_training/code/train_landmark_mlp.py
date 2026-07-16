import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2
import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

PROJECT_DIR = Path("D:/TA Mesin")
TRAIN_DIR = PROJECT_DIR / "data/train"
TEST_DIR = PROJECT_DIR / "data/test"
MODEL_PATH = PROJECT_DIR / "5_training/models/hand_landmarker.task"
OUTPUT_MODEL = PROJECT_DIR / "5_training/models/landmark_mlp.pth"
PROCESSED_DIR = PROJECT_DIR / "data/processed"
OUTPUT_DIR = PROJECT_DIR / "5_training"
IMG_DIR = OUTPUT_DIR / "images"

BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
PATIENCE = 7
N_FEATURES = 63
N_LANDMARKS = 21
IMG_SIZE = (200, 200)
MIN_DETECTION_CONFIDENCE = 0.3


class LandmarkMLP(nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def extract_landmarks_from_image(image_rgb, detector):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)
    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        flat = np.array(
            [(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32
        ).flatten()
        return flat
    return None


def load_and_extract(source_dir, detector, label_map, use_subdirs=True, max_per_class=None):
    X, y = [], []
    start_time = time.time()

    if use_subdirs:
        class_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
        total_files = 0
        for cd in class_dirs:
            total_files += len(list(cd.glob("*.jpg")))

        processed = 0
        detected = 0
        for cls_idx, cls_dir in enumerate(class_dirs):
            cls_name = cls_dir.name
            img_paths = sorted(cls_dir.glob("*.jpg"))
            if max_per_class:
                img_paths = img_paths[:max_per_class]

            for img_path in img_paths:
                gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    processed += 1
                    continue
                gray = cv2.resize(gray, IMG_SIZE)
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

                features = extract_landmarks_from_image(rgb, detector)
                processed += 1
                if features is not None:
                    X.append(features)
                    y.append(cls_idx)
                    detected += 1

                if processed % 2000 == 0:
                    elapsed = time.time() - start_time
                    fps = processed / elapsed if elapsed > 0 else 0
                    eta = (total_files - processed) / fps if fps > 0 else 0
                    print(
                        f"  Progress: {processed}/{total_files} "
                        f"({detected} detected, {detected/max(processed,1)*100:.1f}%) "
                        f"| {fps:.1f} img/s | ETA: {eta/60:.1f}m"
                    )
    else:
        img_paths = sorted(source_dir.glob("*.jpg"))
        for img_path in img_paths:
            cls_name = img_path.stem
            if cls_name not in label_map:
                continue
            cls_idx = label_map.index(cls_name)
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            gray = cv2.resize(gray, IMG_SIZE)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            features = extract_landmarks_from_image(rgb, detector)
            if features is not None:
                X.append(features)
                y.append(cls_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


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
    print("TA - Training Landmark MLP (MediaPipe + MLP)")
    print("=" * 60)

    device = torch.device("cpu")
    print(f"Device: {device}")
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Init MediaPipe ---
    print(f"\n[STEP] Loading MediaPipe HandLandmarker...")
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )
    detector = HandLandmarker.create_from_options(hand_options)
    print("[OK] MediaPipe loaded.")

    # --- Collect classes ---
    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    label_map = classes
    print(f"\n[INFO] {len(classes)} classes: {label_map}")

    # --- Extract landmarks from training data ---
    print(f"\n[STEP] Extracting landmarks from TRAINING set...")
    t0 = time.time()
    X_train, y_train = load_and_extract(
        TRAIN_DIR, detector, label_map, use_subdirs=True
    )
    elapsed = time.time() - t0
    print(f"\n[OK] Training: {len(X_train)} samples detected in {elapsed/60:.1f}m")
    print(f"  Shape: {X_train.shape}")

    # --- Extract landmarks from test data (holdout) ---
    print(f"\n[STEP] Extracting landmarks from TEST set...")
    X_test, y_test = load_and_extract(
        TEST_DIR, detector, label_map, use_subdirs=False
    )
    print(f"[OK] Test holdout: {len(X_test)} samples detected")

    # --- Close detector ---
    detector.close()

    # --- Split train into train/val ---
    print(f"\n[STEP] Splitting data...")
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test holdout: {len(X_test)}")
    print(f"  Classes represented: {len(np.unique(y_train))}")

    # --- DataLoaders ---
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long()),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long()),
        batch_size=BATCH_SIZE,
    )

    # --- Model ---
    model = LandmarkMLP(num_classes=len(classes)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print(f"\n[STEP] Model: {total_params:,} parameters")

    # --- Training ---
    print(f"\n[STEP] Training ({EPOCHS} epochs, patience={PATIENCE})...")
    best_acc, best_epoch, stall = 0, 0, 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(
            f"  Epoch {epoch:2d}/{EPOCHS} | "
            f"train loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
            f"val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            stall = 0
            torch.save(model.state_dict(), str(OUTPUT_MODEL))
        else:
            stall += 1
            if stall >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    total_time = time.time() - t_start
    print(f"\n[OK] Training selesai dalam {total_time/60:.1f} menit")
    print(f"  Best val accuracy: {best_acc:.4f} (epoch {best_epoch})")
    print(f"  Model saved: {OUTPUT_MODEL}")

    # --- Evaluate on holdout test ---
    if len(X_test) > 0:
        print(f"\n[STEP] Evaluating on holdout test set...")
        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_test), torch.from_numpy(y_test).long()
            ),
            batch_size=BATCH_SIZE,
        )
        test_loss, test_acc = eval_epoch(
            model, test_loader, criterion, device
        )
        print(f"  Test holdout accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

        # Per-class breakdown
        model.eval()
        with torch.no_grad():
            all_preds = []
            for Xb, _ in test_loader:
                out = model(Xb.to(device))
                all_preds.append(out.argmax(1).cpu().numpy())
            all_preds = np.concatenate(all_preds)
        for i in range(len(y_test)):
            gt = classes[y_test[i]]
            pred = classes[all_preds[i]]
            mark = "OK" if y_test[i] == all_preds[i] else "XX"
            print(f"  {classes[y_test[i]]:10s} -> {pred:10s} {mark}")

    # --- Plot ---
    epochs_range = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs_range, val_losses, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, train_accs, "b-", label="Train Accuracy")
    ax2.plot(epochs_range, val_accs, "r-", label="Val Accuracy")
    ax2.axhline(best_acc, color="g", ls="--", label=f"Best Val ({best_acc:.3f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path = IMG_DIR / "landmark_training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}")

    print(f"\n{'='*60}")
    print("Landmark MLP Training SELESAI.")
    print("=" * 60)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main()
