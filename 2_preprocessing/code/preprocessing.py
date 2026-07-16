"""
PENJELASAN CODE - PREPROCESSING DATASET ASL ALPHABET

1. Setup & Path (baris 60-68):
    - Tentukan path ke dataset train, folder output gambar, dan dokumentasi.

2. Load Sample Gambar (baris 72-82):
    - Ambil 3 sample dari tiap kelas (awal, tengah, akhir).
    - Untuk menampilkan variasi dalam kelas yang sama.
    - Gunakan cv2.imread() untuk load gambar.

3. Normalisasi Pixel (baris 86-93):
    - Konversi nilai pixel dari 0-255 menjadi 0-1.
    - Caranya: bagi array gambar dengan 255.0.
    - Normalisasi diperlukan agar model neural network lebih stabil saat training.

4. Augmentasi Data (baris 97-112):
    - Rotasi: memutar gambar +-15 derajat.
    - Width Shift: geser horizontal +-10%.
    - Height Shift: geser vertikal +-10%.
    - Zoom: perbesar/perkecil +-10%.
    - Augmentasi dilakukan secara acak menggunakan numpy random.
    - Dilakukan per sample untuk menghasilkan variasi.

5. Visualisasi Perbandingan (baris 116-147):
    - Tampilkan 1 sample per kelas: Original vs Normalized vs Augmented.
    - Grid 29 baris x 3 kolom.
    - Output: comparison_grid.png

6. Visualisasi Efek Normalisasi (baris 151-166):
    - Histogram pixel value sebelum dan sesudah normalisasi.
    - Menunjukkan bahwa data sudah discale ke range [0,1].
    - Output: normalization_histogram.png

7. Dokumentasi (baris 170-193):
    - Generate dokumentasi.md otomatis.
    - Berisi ringkasan preprocessing yang dilakukan.

Output Files:
    - images/comparison_grid.png       -> Perbandingan original vs normalized vs augmented
    - images/normalization_histogram.png -> Histogram sebelum & sesudah normalisasi

Libraries:
    - cv2 (OpenCV)   : Baca & proses gambar
    - numpy          : Manipulasi array & operasi matematika
    - matplotlib     : Visualisasi grid & histogram
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. SETUP & PATH
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_DIR = BASE_DIR / 'data' / 'train'
OUTPUT_DIR = BASE_DIR / '2_preprocessing'
IMG_DIR = OUTPUT_DIR / 'images'
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Load daftar kelas
classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
print(f"[INFO] Ditemukan {len(classes)} kelas")

# ============================================================
# 2. LOAD SAMPLE GAMBAR
# ============================================================
# Ambil 3 sample per kelas: index 0 (awal), tengah, dan akhir
samples_per_class = 3
sample_data = {}  # {kelas: [img_awal, img_tengah, img_akhir]}

for c in classes:
    imgs = sorted((TRAIN_DIR / c).glob('*.jpg'))
    idxs = [0, len(imgs)//2, len(imgs)-1]
    sample_data[c] = [cv2.imread(str(imgs[i])) for i in idxs]

print(f"[INFO] Loaded {len(classes) * samples_per_class} sample images")

# ============================================================
# 3. NORMALISASI PIXEL
# ============================================================
# Konversi 0-255 -> 0-1 dengan membagi 255.0
def normalize_image(img):
    return img / 255.0

# ============================================================
# 4. AUGMENTASI DATA
# ============================================================
def augment_image(img):
    h, w = img.shape[:2]
    angle = np.random.uniform(-15, 15)
    tx = np.random.uniform(-0.1, 0.1) * w
    ty = np.random.uniform(-0.1, 0.1) * h
    scale = np.random.uniform(0.9, 1.1)

    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT)

# ============================================================
# 5. VISUALISASI PERBANDINGAN
# ============================================================
rows, cols = len(classes), 3
fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
fig.suptitle('Perbandingan: Original | Normalized | Augmented', fontsize=16, fontweight='bold', y=0.98)

for row, c in enumerate(classes):
    original = sample_data[c][1]  # sample tengah
    normalized = normalize_image(original.copy())
    augmented = augment_image(original.copy())
    augmented_norm = normalize_image(augmented)

    for col, (img, title) in enumerate([
        (original, 'Original'),
        (normalized, 'Normalized'),
        (augmented_norm, 'Augmented')
    ]):
        ax = axes[row, col]
        if col == 0:
            ax.set_ylabel(c, fontsize=10, fontweight='bold')
        if img.max() <= 1.0:
            img_display = (img * 255).astype(np.uint8)
        else:
            img_display = img.astype(np.uint8)
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB) if img_display.ndim == 3 else img_display
        ax.imshow(img_rgb)
        if row == 0:
            ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

plt.tight_layout()
plt.savefig(str(IMG_DIR / 'comparison_grid.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] comparison_grid.png")

# ============================================================
# 6. VISUALISASI HISTOGRAM NORMALISASI
# ============================================================
sample_class = classes[0]
sample_img = sample_data[sample_class][1]
normalized = normalize_image(sample_img.copy())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ('b', 'g', 'r')
channel_names = ('Blue', 'Green', 'Red')

img_uint8 = (sample_img * 255).astype(np.uint8) if sample_img.max() <= 1.0 else sample_img.astype(np.uint8)
for ax, data, title, bins, rng in [
    (axes[0], img_uint8, 'Original (0-255)', 256, [0, 256]),
    (axes[1], normalized.reshape(-1, 3), 'Normalized (0-1)', 256, [0, 1])
]:
    if title.startswith('Original'):
        for ch, color, name in zip(range(3), colors, channel_names):
            hist = cv2.calcHist([data], [ch], None, [bins], rng)
            ax.plot(hist, color=color, label=name, linewidth=1.5)
    else:
        for ch, color, name in zip(range(3), colors, channel_names):
            hist, _ = np.histogram(data[:, ch], bins=bins, range=rng)
            ax.plot(hist, color=color, label=name, linewidth=1.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Pixel Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(IMG_DIR / 'normalization_histogram.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] normalization_histogram.png")

# ============================================================
# 7. GENERATE DOKUMENTASI
# ============================================================
normalisasi_info = """
Normalisasi pixel dilakukan dengan membagi setiap nilai pixel dengan 255.0,
sehingga range nilai berubah dari [0, 255] menjadi [0.0, 1.0].

Tujuan:
- Mempercepat konvergensi model saat training
- Menghindari dominasi fitur dengan skala besar
- Stabilisasi gradient descent
"""

augmentasi_info = """
Augmentasi data dilakukan untuk menambah variasi dataset dan mengurangi overfitting.
Parameter augmentasi:
- Rotasi: +-15 derajat
- Width Shift: +-10% dari lebar gambar
- Height Shift: +-10% dari tinggi gambar
- Zoom: 0.9x - 1.1x

Semua augmentasi menggunakan borderMode=cv2.BORDER_CONSTANT (isi hitam).
"""

table_plan = f"""
| Preprocessing Step | Status | Keterangan |
|---|---|---|
| Resize | Tidak diperlukan | Semua gambar sudah 200x200 px |
| Normalisasi | Diterapkan | Pixel 0-255 -> 0-1 (float32) |
| Augmentasi | Diterapkan | Rotasi, shift, zoom (acak) |
| Feature Extraction | Belum | Akan dilakukan di tahap selanjutnya |
"""

markdown_content = f"""# Dokumentasi Preprocessing Dataset ASL Alphabet

## Tujuan
Melakukan normalisasi dan augmentasi pada dataset sebelum ekstraksi fitur.

## Dataset Info
| Metrik | Nilai |
|---|---|
| Jumlah Kelas | {len(classes)} |
| Ukuran Gambar | 200 x 200 px |
| Format | JPG RGB |
| Total Sample | {len(classes) * samples_per_class} gambar ({len(classes)} kelas) |

## Preprocessing Steps

### 1. Normalisasi Pixel
{normalisasi_info}

### 2. Augmentasi Data
{augmentasi_info}

### 3. Progress Plan
{table_plan}

## Visualisasi Hasil

### Perbandingan Original vs Normalized vs Augmented
![Comparison Grid](images/comparison_grid.png)
*Setiap baris = 1 kelas. Kolom: Original -> Normalized -> Augmented*

### Histogram Normalisasi
![Normalization Histogram](images/normalization_histogram.png)
*Histogram pixel value sebelum (kiri) dan sesudah (kanan) normalisasi*

## Catatan
- Normalisasi dilakukan per gambar (tidak ada global statistics).
- Augmentasi dilakukan secara acak dengan seed berbeda tiap run.
- Gambar augmented tidak disimpan ke disk, diterapkan real-time saat training nanti.
- Setelah preprocessing selesai, akan dilanjutkan ke Feature Extraction dengan MediaPipe.
"""

doc_path = OUTPUT_DIR / 'dokumentasi.md'
doc_path.write_text(markdown_content, encoding='utf-8')
print(f"[OK] dokumentasi.md")
print(f"\n{'='*50}")
print(f"Output gambar tersimpan di: 2_preprocessing/images/")
print(f"File: comparison_grid.png, normalization_histogram.png")
print(f"{'='*50}")
