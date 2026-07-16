"""
TA: Tahap 6 - Evaluasi Model CNN
=================================
Evaluasi performa model CNN pada test set.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import sys
sys.path.insert(0, str(Path(__file__).parents[2] / "5_training/code"))
import importlib.util
spec = importlib.util.spec_from_file_location("train_cnn", Path(__file__).parents[2] / "5_training/code/train_cnn.py")
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
ASL_CNN = train_mod.ASL_CNN

PROJECT_DIR = Path("D:/TA Mesin")
PROCESSED_DIR = PROJECT_DIR / "data/processed"
MODEL_PATH = PROJECT_DIR / "5_training/models/asl_cnn_best.pth"
OUTPUT_DIR = PROJECT_DIR / "6_evaluasi"
IMG_DIR = OUTPUT_DIR / "images"


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        out = model(Xb)
        all_preds.append(out.argmax(1).cpu().numpy())
        all_labels.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def main():
    print("=" * 60)
    print("TA - Tahap 6: Evaluasi Model CNN")
    print("=" * 60)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # --- Load data ---
    print("\n[STEP] Loading test data...")
    X_test = np.load(str(PROCESSED_DIR / "test_X.npy"))
    y_test = np.load(str(PROCESSED_DIR / "test_y.npy"))
    X_holdout = np.load(str(PROCESSED_DIR / "test_holdout_X.npy"))
    y_holdout = np.load(str(PROCESSED_DIR / "test_holdout_y.npy"))
    classes = np.load(str(PROCESSED_DIR / "classes.npy"), allow_pickle=True)
    print(f"  Test: {X_test.shape}, Holdout: {X_holdout.shape}")

    from torch.utils.data import DataLoader, TensorDataset
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long()),
        batch_size=128
    )
    holdout_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_holdout), torch.from_numpy(y_holdout).long()),
        batch_size=128
    )

    # --- Load model ---
    print("\n[STEP] Loading best model...")
    model = ASL_CNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device, weights_only=True))
    print(f"  Model loaded: {MODEL_PATH}")

    # --- 1. Test set evaluation ---
    print("\n[STEP] Evaluating on test set (13,050 samples)...")
    t0 = time.time()
    y_pred, y_true = predict(model, test_loader, device)
    accuracy = (y_pred == y_true).mean()
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Evaluated in {time.time()-t0:.1f}s")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("\n" + report)

    # Save report
    with open(str(OUTPUT_DIR / "classification_report.txt"), "w") as f:
        f.write(report)

    # --- 2. Confusion Matrix ---
    print("[STEP] Generating confusion matrix...")
    fig, ax = plt.subplots(figsize=(20, 18))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(ax=ax, xticks_rotation=90, values_format="d", cmap="Blues")
    ax.set_title(f"Confusion Matrix (Test Accuracy: {accuracy*100:.2f}%)", fontsize=14)
    plt.tight_layout()
    save_path_cm = IMG_DIR / "confusion_matrix.png"
    plt.savefig(save_path_cm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path_cm}")

    # --- 3. Top misclassified classes ---
    print("\n[STEP] Top misclassified pairs...")
    misclass = {}
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i][j] > 0:
                key = f"{classes[i]} -> {classes[j]}"
                misclass[key] = cm[i][j]
    sorted_mis = sorted(misclass.items(), key=lambda x: -x[1])[:15]
    for k, v in sorted_mis:
        print(f"  {k}: {v} errors")

    # --- 4. Holdout set evaluation (28 images) ---
    print("\n[STEP] Holdout test set (28 images)...")
    y_pred_ho, y_true_ho = predict(model, holdout_loader, device)
    ho_correct = (y_pred_ho == y_true_ho).sum()
    print(f"  Holdout accuracy: {ho_correct}/{len(y_true_ho)} ({ho_correct/len(y_true_ho)*100:.1f}%)")
    for i in range(len(y_true_ho)):
        gt = classes[y_true_ho[i]]
        pred = classes[y_pred_ho[i]]
        mark = "✓" if y_pred_ho[i] == y_true_ho[i] else "✗"
        print(f"  {i+1:2d}. {gt:10s} -> {pred:10s} {mark}")

    # --- 5. Sample predictions (correct + wrong) ---
    print("\n[STEP] Sample predictions visualization...")
    n_samples = 16
    correct_idx = np.where(y_pred == y_true)[0]
    wrong_idx = np.where(y_pred != y_true)[0]

    np.random.seed(42)
    sample_correct = np.random.choice(correct_idx, min(n_samples//2, len(correct_idx)), replace=False)
    sample_wrong = np.random.choice(wrong_idx, min(n_samples//2, len(wrong_idx)), replace=False)

    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    for i, idx in enumerate(sample_correct):
        row, col = i // 8, i % 8
        img = X_test[idx, 0]
        axes[row, col].imshow(img, cmap="gray")
        axes[row, col].set_title(f"GT:{classes[y_true[idx]]}\nPred:{classes[y_pred[idx]]}", fontsize=8, color="green")
        axes[row, col].axis("off")
    for i, idx in enumerate(sample_wrong):
        row, col = (len(sample_correct) + i) // 8, (len(sample_correct) + i) % 8
        img = X_test[idx, 0]
        axes[row, col].imshow(img, cmap="gray")
        axes[row, col].set_title(f"GT:{classes[y_true[idx]]}\nPred:{classes[y_pred[idx]]}", fontsize=8, color="red")
        axes[row, col].axis("off")

    for i in range(n_samples, 32):
        row, col = i // 8, i % 8
        axes[row, col].axis("off")

    plt.tight_layout()
    save_path_sp = IMG_DIR / "sample_predictions.png"
    plt.savefig(save_path_sp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path_sp}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"[SUMMARY] Test Accuracy: {accuracy*100:.2f}%")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Confusion matrix: {save_path_cm}")
    print(f"  Sample predictions: {save_path_sp}")
    print(f"  Classification report: {OUTPUT_DIR / 'classification_report.txt'}")
    print("Tahap 6 - Evaluasi SELESAI.")
    print("=" * 60)


if __name__ == "__main__":
    main()
