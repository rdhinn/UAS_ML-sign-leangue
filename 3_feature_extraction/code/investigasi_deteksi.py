"""
Investigasi: Kenapa MediaPipe gagal deteksi tangan di 79% gambar?
Cek sample gambar + MediaPipe detection + statistik gambar.
"""

import cv2
import numpy as np
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
OUTPUT_DIR = PROJECT_DIR / "3_feature_extraction/images"
MODEL_PATH = Path("D:/Miniconda/envs/signlang2/Lib/site-packages/mediapipe/modules/hand_landmarker/hand_landmarker.task")

IMG_SIZE = (200, 200)

# Classes to test: some with hands, some without
TEST_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'nothing', 'space', 'del', 'Z']

print("="*60)
print("INVESTIGASI: MediaPipe Hand Detection Failure Analysis")
print("="*60)

# --- 1. Image Stats untuk sample ---
print("\n[1] Image Statistics per Class")
stats = []
for cls_name in TEST_CLASSES:
    cls_dir = TRAIN_DIR / cls_name
    if not cls_dir.exists():
        print(f"  [!] {cls_name} not found")
        continue
    images = sorted(cls_dir.glob("*.jpg"))[:50]
    brightness = []
    contrast = []
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, IMG_SIZE)
        brightness.append(img.mean())
        contrast.append(img.std())
    
    avg_bright = np.mean(brightness) if brightness else 0
    avg_contrast = np.mean(contrast) if contrast else 0
    stats.append((cls_name, avg_bright, avg_contrast))
    print(f"  {cls_name:10s} | brightness: {avg_bright:6.1f} | contrast: {avg_contrast:5.1f}")

# --- 2. Cek detection di berbagai confidence threshold ---
print("\n[2] Detection Rate vs Confidence Threshold (50 samples per class)")
detector_full = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
)

detector_low = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )
)

for cls_name in TEST_CLASSES[:6]:  # Only letter classes
    cls_dir = TRAIN_DIR / cls_name
    images = sorted(cls_dir.glob("*.jpg"))[:50]
    detected_05 = 0
    detected_03 = 0
    for p in images:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None: continue
        gray = cv2.resize(gray, IMG_SIZE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        r1 = detector_full.detect(mp_img)
        r2 = detector_low.detect(mp_img)
        if r1.hand_landmarks: detected_05 += 1
        if r2.hand_landmarks: detected_03 += 1
    
    print(f"  {cls_name:10s} | thr=0.5: {detected_05:3d}/50 | thr=0.3: {detected_03:3d}/50")

detector_full.close()
detector_low.close()

# --- 3. Visual: cek langsung gambar + area tangan ---
print("\n[3] Generating diagnostic visualization...")
import matplotlib.pyplot as plt

n_rows = 4
n_cols = 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
axes = axes.flatten()

idx = 0
for cls_name in TEST_CLASSES:
    cls_dir = TRAIN_DIR / cls_name
    images = sorted(cls_dir.glob("*.jpg"))[:1]  # 1 sample per class
    if not images:
        axes[idx].axis("off")
        idx += 1
        continue
    img_path = images[0]
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, IMG_SIZE)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detector = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
        )
    )
    result = detector.detect(mp_img)
    
    display = rgb.copy()
    status = "NO HAND"
    color = (255, 0, 0)
    if result.hand_landmarks:
        status = "DETECTED"
        color = (0, 255, 0)
        h, w = display.shape[:2]
        for lm in result.hand_landmarks[0]:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(display, (cx, cy), 4, (0, 255, 0), -1)
    
    ax = axes[idx]
    ax.imshow(display)
    ax.set_title(f"{cls_name}: {status}", fontsize=10, color='green' if status=='DETECTED' else 'red')
    ax.axis("off")
    detector.close()
    idx += 1

# Hide unused axes
for i in range(idx, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
save_path = OUTPUT_DIR / "investigasi_deteksi.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Diagnostic image saved: {save_path}")

# --- 4. Summary ---
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print("Cek file: 3_feature_extraction/images/investigasi_deteksi.png")
print("Untuk melihat langsung sample gambar + hasil deteksi MediaPipe.")
