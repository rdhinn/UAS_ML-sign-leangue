"""
PENJELASAN CODE - EKSPLORASI DATASET ASL ALPHABET

1. Setup & Konfigurasi (baris 56-64):
    Menentukan path ke folder train, test, dan output gambar.
    TRAIN_DIR = path ke data/train/
    TEST_DIR  = path ke data/test/
    IMG_DIR   = path ke eksplorasi_dataset/images/ (untuk menyimpan output grafik)

2. Load Daftar Kelas & Hitung Distribusi (baris 68-75):
    - Scan semua folder dalam data/train/ -> dapat 29 kelas
    - Hitung jumlah file .jpg per kelas menggunakan glob
    - Hasil: dictionary train_counts = {'A': 3000, 'B': 3000, ...}

3. Ambil Sample Gambar (baris 79-82):
    - Ambil gambar tengah dari tiap kelas (index ke-1500)
    - Digunakan sebagai sample untuk grid visualisasi

4. Ekstrak Info Gambar (baris 86-94):
    - OpenCV membaca satu sample test untuk mengekstrak:
      - Height (200px), Width (200px), Channels (3 = RGB)
    - Gunakan img.shape untuk mendapatkan dimensi

5. Visualisasi Distribusi Kelas (baris 102-117):
    - Matplotlib barchart 29 kelas
    - Warna dari colormap tab20
    - Setiap bar diberi label jumlah gambar di atasnya
    - Output: class_distribution.png

6. Grid Sample Gambar (baris 121-138):
    - Grid 6x5 (30 slot) diisi 29 sample (1 per kelas)
    - Sisa 1 slot dikosongkan
    - Menggunakan matplotlib subplots
    - Output: sample_grid.png

7. Infografis Dataset (baris 142-165):
    - Matplotlib figure tanpa axis
    - Menampilkan semua metrik dalam format teks monospace
    - Output: dataset_info.png

Output Files:
    - images/class_distribution.png  -> Bar chart distribusi 29 kelas
    - images/sample_grid.png         -> Grid 6x5 sample gambar per kelas
    - images/dataset_info.png        -> Infografis ringkasan dataset

Libraries:
    - os / pathlib : Navigasi file system
    - cv2 (OpenCV) : Baca gambar, ekstrak dimensi & channel
    - numpy        : Manipulasi array
    - matplotlib   : Visualisasi grafik & grid gambar
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. SETUP & KONFIGURASI
# ============================================================
# BASE_DIR mengarah ke D:/TA Mesin/ (parent dari eksplorasi_dataset/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Folder dataset training dan testing
TRAIN_DIR = BASE_DIR / 'data' / 'train'
TEST_DIR = BASE_DIR / 'data' / 'test'
# Folder output untuk menyimpan gambar hasil visualisasi
OUTPUT_DIR = BASE_DIR / '1_eksplorasi_dataset'
IMG_DIR = OUTPUT_DIR / 'images'
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOAD DAFTAR KELAS & HITUNG DISTRIBUSI
# ============================================================
# Scan semua folder dalam data/train/ -> 29 kelas (A-Z, del, nothing, space)
classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
n_classes = len(classes)

# Hitung jumlah file .jpg per kelas
train_counts = {}
for c in classes:
    imgs = list((TRAIN_DIR / c).glob('*.jpg'))
    train_counts[c] = len(imgs)

# Daftar file test
test_images = sorted([f.name for f in TEST_DIR.iterdir() if f.suffix.lower() == '.jpg'])

# ============================================================
# 3. AMBIL SAMPLE GAMBAR
# ============================================================
# Ambil gambar tengah dari tiap kelas (index ke-1500) sebagai sample
sample_paths = {}
for c in classes:
    imgs = sorted((TRAIN_DIR / c).glob('*.jpg'))
    sample_paths[c] = str(imgs[len(imgs)//2]) if imgs else None

# ============================================================
# 4. EKSTRAK INFO GAMBAR
# ============================================================
# Baca satu sample test untuk ekstrak dimensi (200x200) dan channel (RGB)
img_info = {}
sample_test = list(TEST_DIR.glob('*.jpg'))
if sample_test:
    img = cv2.imread(str(sample_test[0]))
    img_info['height'] = img.shape[0]
    img_info['width'] = img.shape[1]
    img_info['channels'] = img.shape[2] if len(img.shape) > 2 else 1
    img_info['dtype'] = str(img.dtype)

# Kumpulkan semua ekstensi file yang ada
all_extensions = set()
for c in classes:
    for f in (TRAIN_DIR / c).iterdir():
        all_extensions.add(f.suffix.lower())
test_ext = set(f.suffix.lower() for f in TEST_DIR.iterdir() if f.suffix.lower() == '.jpg')
all_extensions.update(test_ext)

total_train = sum(train_counts.values())
total_test = len(test_images)
grand_total = total_train + total_test

# ============================================================
# 5. VISUALISASI DISTRIBUSI KELAS
# ============================================================
# Barchart 29 kelas untuk melihat balance dataset
fig, ax = plt.subplots(figsize=(16, 8))
colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
bars = ax.bar(classes, [train_counts[c] for c in classes], color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Kelas', fontsize=14, fontweight='bold')
ax.set_ylabel('Jumlah Gambar', fontsize=14, fontweight='bold')
ax.set_title('Distribusi Jumlah Gambar per Kelas (Training Set)', fontsize=16, fontweight='bold')
ax.tick_params(axis='x', rotation=90, labelsize=10)
ax.set_ylim(0, max(train_counts.values()) * 1.15)
for bar, count in zip(bars, [train_counts[c] for c in classes]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(count),
            ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig(str(IMG_DIR / 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] class_distribution.png — {n_classes} classes, train={total_train}, test={total_test}")

# ============================================================
# 6. GRID SAMPLE GAMBAR
# ============================================================
# Grid 6x5 = 30 slot, diisi 29 sample (1 per kelas), 1 slot dikosongkan
rows, cols = 6, 5
fig, axes = plt.subplots(rows, cols, figsize=(20, 24))
axes = axes.flatten()
for idx, c in enumerate(classes):
    ax = axes[idx]
    if idx < n_classes and sample_paths.get(c):
        img = cv2.imread(sample_paths[c])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(c, fontsize=14, fontweight='bold')
    ax.axis('off')
for idx in range(n_classes, len(axes)):
    axes[idx].axis('off')
plt.suptitle('Sample Gambar per Kelas (Training Set)', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(str(IMG_DIR / 'sample_grid.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] sample_grid.png")

# ============================================================
# 7. INFOGRAFIS DATASET
# ============================================================
# Tampilkan semua metrik dalam satu gambar
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
info_text = (
    f"DATASET ASL ALPHABET - INFORMASI UMUM\n"
    f"{'='*50}\n\n"
    f"Total Training Images           : {total_train:,}\n"
    f"Total Test Images               : {total_test}\n"
    f"Grand Total                     : {grand_total:,}\n"
    f"Jumlah Kelas Training           : {n_classes}\n"
    f"Kelas Training                  : {', '.join(classes)}\n"
    f"\n"
    f"--- DETAIL GAMBAR ---\n"
    f"Format File                     : {', '.join(sorted(all_extensions))}\n"
    f"Image Width                     : {img_info.get('width', 'N/A')} px\n"
    f"Image Height                    : {img_info.get('height', 'N/A')} px\n"
    f"Channels                        : {img_info.get('channels', 'N/A')} (RGB)\n"
    f"Data Type                       : {img_info.get('dtype', 'N/A')}\n"
    f"\n"
    f"--- DISTRIBUSI ---\n"
    f"Balanced                        : {'YES' if len(set(train_counts.values())) == 1 else 'NO'}\n"
    f"Min Images per Class            : {min(train_counts.values())}\n"
    f"Max Images per Class            : {max(train_counts.values())}\n"
    f"\n"
    f"--- NOTE ---\n"
    f"- Test set TIDAK memiliki kelas 'del'\n"
    f"- Test images direname: A_test.jpg -> A.jpg\n"
    f"- Dataset asli: ASL Alphabet (Kaggle)"
)
ax.text(0.5, 0.5, info_text, transform=ax.transAxes, fontsize=13,
        verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace', linespacing=1.5)
plt.savefig(str(IMG_DIR / 'dataset_info.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] dataset_info.png")

# ============================================================
# 8. SELESAI
# ============================================================
print(f"\n{'='*50}")
print(f"Output gambar tersimpan di: 1_eksplorasi_dataset/images/")
print(f"File output: class_distribution.png, sample_grid.png, dataset_info.png")
print(f"{'='*50}")
