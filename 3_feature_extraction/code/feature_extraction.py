"""
TA Mesin: Tahap 3 - Feature Extraction dengan MediaPipe HandLandmarker
======================================================================

Tujuan:
- Mendeteksi 21 landmarks tangan dari setiap gambar ASL Alphabet
- Mengekstrak 63 fitur numerik (21 titik x 3 koordinat [x, y, z])
- Menyimpan hasil ekstraksi ke file .npy untuk tahap training

Pipeline:
1. Load gambar grayscale 200x200 → konversi ke RGB
2. Deteksi tangan dengan MediaPipe HandLandmarker
3. Ekstrak landmark (x, y, z) — normalisasi koordinat
4. Simpan sebagai matriks fitur: X (n_samples, 63), y (labels), label_map

Output:
- data/processed/X.npy         : fitur (float32)
- data/processed/y.npy         : label index (int)
- data/processed/classes.npy   : nama kelas (string)

Catatan:
- Gambar tanpa tangan terdeteksi → landmark diisi 0.0
- Menggunakan model float16 hand_landmarker.task (~8 MB)
"""

from pathlib import Path
import numpy as np
import cv2
import time
import sys

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ─── Konfigurasi ─────────────────────────────────────────────────────
PROJECT_DIR = Path("D:/TA Mesin")
DATA_DIR = PROJECT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "3_feature_extraction"
IMG_DIR = OUTPUT_DIR / "images"
MODEL_DIR = (
    Path("D:/Miniconda/envs/signlang2/Lib/site-packages/mediapipe/modules/hand_landmarker")
)
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

N_LANDMARKS = 21
N_FEATURES = N_LANDMARKS * 3  # 63
IMG_SIZE = (200, 200)

# ─── Inisialisasi MediaPipe ─────────────────────────────────────────
print("[INFO] Loading MediaPipe HandLandmarker model...")
mp_hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
)
detector = HandLandmarker.create_from_options(mp_hand_options)
print("[INFO] Model loaded.")


def extract_landmarks(image_rgb: np.ndarray) -> np.ndarray:
    """
    Deteksi landmark tangan dari gambar RGB.
    Return array (63,) — 21 landmarks × [x, y, z].
    Jika tidak terdeteksi, return array nol.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        flat = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32).flatten()
        return flat
    else:
        return np.zeros(N_FEATURES, dtype=np.float32)


def process_dataset(source_dir: Path, label_map: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterasi semua gambar di source_dir, ekstrak landmark.
    Return X (n_samples, 63), y (n_samples,).
    """
    classes = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    all_X, all_y = [], []

    for cls_idx, cls_dir in enumerate(classes):
        cls_name = cls_dir.name
        image_files = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.jpeg")) + sorted(cls_dir.glob("*.png"))
        n_images = len(image_files)
        if n_images == 0:
            continue

        print(f"[{cls_idx+1}/{len(classes)}] Processing class '{cls_name}' ({n_images} images)...")
        t0 = time.time()

        for img_path in image_files:
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            gray = cv2.resize(gray, IMG_SIZE)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            features = extract_landmarks(rgb)
            all_X.append(features)
            all_y.append(cls_idx)

        elapsed = time.time() - t0
        print(f"  -> {n_images} images in {elapsed:.1f}s ({n_images/elapsed:.1f} img/s)")

    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.int32)


def visualize_landmarks(source_dir: Path, label_map: list[str], n_samples: int = 8):
    """Simpan gambar sample dengan landmark yang digambar."""
    import matplotlib.pyplot as plt

    classes = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    sample_classes = np.random.choice(classes, min(n_samples, len(classes)), replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for ax, cls_dir in zip(axes, sample_classes):
        image_files = list(cls_dir.glob("*.jpg"))[:1]
        if not image_files:
            ax.axis("off")
            continue
        img_path = image_files[0]
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, IMG_SIZE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        display = rgb.copy()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            h, w = display.shape[:2]
            for lm in result.hand_landmarks[0]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)
            # Draw connections
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
                (5,9),(9,13),(13,17)
            ]
            for i, j in connections:
                if i < len(result.hand_landmarks[0]) and j < len(result.hand_landmarks[0]):
                    x1, y1 = int(result.hand_landmarks[0][i].x*w), int(result.hand_landmarks[0][i].y*h)
                    x2, y2 = int(result.hand_landmarks[0][j].x*w), int(result.hand_landmarks[0][j].y*h)
                    cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 1)
            ax.set_title(f"{cls_dir.name} (detected)")
        else:
            ax.set_title(f"{cls_dir.name} (no hand)")

        ax.imshow(display)
        ax.axis("off")

    plt.tight_layout()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    save_path = IMG_DIR / "landmark_samples.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Sample visualization saved: {save_path}")


# ─── Main ───────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("TA Mesin - Tahap 3: Feature Extraction (MediaPipe)")
    print("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect class names ──
    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    label_map = classes
    n_classes = len(label_map)
    print(f"[INFO] {n_classes} classes found: {label_map[:5]}...{label_map[-3:]}")

    # ── 2. Process training data ──
    print(f"\n{'-'*50}")
    print("[STEP] Extracting landmarks from TRAINING set...")
    t_start = time.time()
    X_train, y_train = process_dataset(TRAIN_DIR, label_map)
    t_train = time.time() - t_start
    print(f"[OK] Training: {X_train.shape[0]} samples in {t_train:.1f}s")

    #     ── 3. Process test data (flat structure) ──
    print(f"\n{'-'*50}")
    print("[STEP] Extracting landmarks from TEST set...")
    test_files = sorted(TEST_DIR.glob("*.jpg")) + sorted(TEST_DIR.glob("*.jpeg")) + sorted(TEST_DIR.glob("*.png"))
    X_test_list, y_test_list = [], []
    t_start = time.time()
    for img_path in test_files:
        cls_name = img_path.stem  # filename without extension = class name
        if cls_name not in label_map:
            print(f"  [!] Unknown class '{cls_name}', skipping {img_path.name}")
            continue
        cls_idx = label_map.index(cls_name)
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        gray = cv2.resize(gray, IMG_SIZE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        features = extract_landmarks(rgb)
        X_test_list.append(features)
        y_test_list.append(cls_idx)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int32)
    t_test = time.time() - t_start
    print(f"[OK] Test: {X_test.shape[0]} samples in {t_test:.1f}s")

    # ── 4. Save to .npy ──
    print(f"\n{'-'*10}")
    print("[STEP] Saving features...")
    # Gabung train + test untuk split nanti
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    np.save(str(PROCESSED_DIR / "X.npy"), X_all)
    np.save(str(PROCESSED_DIR / "y.npy"), y_all)
    np.save(str(PROCESSED_DIR / "classes.npy"), np.array(label_map, dtype=object))

    print(f"  X shape : {X_all.shape}")
    print(f"  y shape : {y_all.shape}")
    print(f"  classes : {n_classes} ({label_map})")
    print(f"[OK] Saved to {PROCESSED_DIR}")

    # ── 5. Visualize ──
    print(f"\n{'-'*50}")
    print("[STEP] Generating visualization...")
    visualize_landmarks(TRAIN_DIR, label_map, n_samples=8)

    # ── 6. Summary ──
    print(f"\n{'-'*50}")
    print("[SUMMARY]")
    detected = np.any(X_all != 0, axis=1)
    print(f"  Samples with hand detected : {detected.sum()} / {len(detected)}")
    print(f"  Samples without hand       : {(~detected).sum()} / {len(detected)}")
    print(f"  Total samples              : {len(X_all)}")
    print(f"  Total features per sample  : {N_FEATURES}")
    print(f"  Processing speed (train)   : {len(X_train)/t_train:.1f} img/s")
    print(f"  Processing speed (test)    : {len(X_test)/t_test:.1f} img/s")

    # ── Cleanup ──
    detector.close()
    print(f"\n{'='*60}")
    print("Tahap 3 — Feature Extraction SELESAI.")
    print("=" * 60)


if __name__ == "__main__":
    main()
