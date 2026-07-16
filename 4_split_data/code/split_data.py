"""
TA: Tahap 4 - Split Data CNN
==============================
- Load semua gambar dari data/train/ dan data/test/
- Resize ke 48x48 (ringan buat CPU, cukup buat bentuk tangan)
- Normalisasi pixel ke [0,1]
- Stratified split: 70% train, 15% val, 15% test
- Simpan sebagai .npy untuk training CNN

Input:
  data/train/    - 29 kelas x 3000 gambar = 87.000
  data/test/     - 28 gambar flat (kelas dari nama file)

Output (data/processed/):
  train_X.npy, train_y.npy    (70%)
  val_X.npy,   val_y.npy      (15%)
  test_X.npy,  test_y.npy     (15%)
  test_holdout_X.npy, test_holdout_y.npy  (28 gambar)
  classes.npy
"""

from pathlib import Path
import numpy as np
import cv2
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path("D:/TA Mesin")
DATA_DIR = PROJECT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "4_split_data"
IMG_DIR = OUTPUT_DIR / "images"

IMG_SIZE = (48, 48)

# --- Fungsi ---

def load_images_from_dir(source_dir, label_map, use_subdirs=True):
    """
    Load gambar dari direktori.
    Jika use_subdirs=True: tiap subfolder = 1 kelas (data/train/)
    Jika False: file flat, nama file = kelas (data/test/)
    """
    X, y = [], []
    
    if use_subdirs:
        class_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
        for cls_idx, cls_dir in enumerate(class_dirs):
            cls_name = cls_dir.name
            img_paths = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.jpeg")) + sorted(cls_dir.glob("*.png"))
            for p in img_paths:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                X.append(img)
                y.append(cls_idx)
    else:
        img_paths = sorted(source_dir.glob("*.jpg")) + sorted(source_dir.glob("*.jpeg")) + sorted(source_dir.glob("*.png"))
        for p in img_paths:
            cls_name = p.stem
            if cls_name not in label_map:
                continue
            cls_idx = label_map.index(cls_name)
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(cls_idx)
    
    return np.array(X), np.array(y)


def main():
    print("="*60)
    print("TA - Tahap 4: Split Data CNN")
    print("="*60)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Collect classes ---
    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    label_map = classes
    n_classes = len(label_map)
    print(f"\n[INFO] {n_classes} classes: {label_map[:3]}...{label_map[-3:]}")
    
    # --- 2. Load training images ---
    print(f"\n[STEP] Loading training images...")
    t0 = time.time()
    X, y = load_images_from_dir(TRAIN_DIR, label_map, use_subdirs=True)
    print(f"  Loaded: {len(X)} images in {time.time()-t0:.1f}s")
    print(f"  Shape: {X.shape}")
    
    # --- 3. Split: 70% train, 15% val, 15% test ---
    print(f"\n[STEP] Stratified split...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
    )
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # --- 4. Load hold-out test set (data/test/) ---
    print(f"\n[STEP] Loading hold-out test set...")
    X_holdout, y_holdout = load_images_from_dir(TEST_DIR, label_map, use_subdirs=False)
    print(f"  Holdout test: {len(X_holdout)} samples")
    
    # --- 5. Normalize & reshape ---
    print(f"\n[STEP] Normalizing and reshaping...")
    # Tambah channel dimension: (N, 48, 48) -> (N, 1, 48, 48) untuk PyTorch (NCHW)
    X_train = X_train.astype(np.float32).reshape(-1, 1, 48, 48) / 255.0
    X_val   = X_val.astype(np.float32).reshape(-1, 1, 48, 48) / 255.0
    X_test  = X_test.astype(np.float32).reshape(-1, 1, 48, 48) / 255.0
    X_holdout = X_holdout.astype(np.float32).reshape(-1, 1, 48, 48) / 255.0
    
    # --- 6. Save ---
    print(f"\n[STEP] Saving to {PROCESSED_DIR}...")
    np.save(str(PROCESSED_DIR / "train_X.npy"), X_train)
    np.save(str(PROCESSED_DIR / "train_y.npy"), y_train)
    np.save(str(PROCESSED_DIR / "val_X.npy"), X_val)
    np.save(str(PROCESSED_DIR / "val_y.npy"), y_val)
    np.save(str(PROCESSED_DIR / "test_X.npy"), X_test)
    np.save(str(PROCESSED_DIR / "test_y.npy"), y_test)
    np.save(str(PROCESSED_DIR / "test_holdout_X.npy"), X_holdout)
    np.save(str(PROCESSED_DIR / "test_holdout_y.npy"), y_holdout)
    np.save(str(PROCESSED_DIR / "classes.npy"), np.array(label_map, dtype=object))
    
    # --- 7. Verify ---
    print(f"\n[VERIFY]")
    for name in ["train", "val", "test", "test_holdout"]:
        Xf = np.load(str(PROCESSED_DIR / f"{name}_X.npy"))
        yf = np.load(str(PROCESSED_DIR / f"{name}_y.npy"))
        classes_in = len(set(yf.tolist()))
        print(f"  {name:15s}: {Xf.shape[0]:6d} samples, {classes_in} classes, {Xf.shape[1:]} shape")
    
    print(f"\n{'='*60}")
    print("Tahap 4 - Split Data SELESAI.")
    print("="*60)

if __name__ == "__main__":
    main()
