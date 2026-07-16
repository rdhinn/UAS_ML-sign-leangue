# Tahap 7: Deployment / Implementasi

## Tujuan
Mengaplikasikan model CNN yang sudah dilatih ke skenario nyata:
1. **Prediksi dari gambar file** — untuk testing gambar individual
2. **Real-time webcam** — deteksi langsung dari kamera

## Cara Penggunaan

### 1. Prediksi Gambar File
```bash
python 7_deployment/code/predict_image.py <path_gambar>
```
Contoh:
```bash
python 7_deployment/code/predict_image.py data/test/A.jpg
```
Output:
```
Prediction: A
Confidence: 98.50%
Top-3:
  1. A          98.50%
  2. F          0.80%
  3. U          0.30%
```

### 2. Webcam Real-Time
```bash
python 7_deployment/code/deploy_webcam.py
```
- Kamera akan aktif dan menampilkan prediksi huruf secara real-time
- Teks **hijau** = confidence ≥ 70% (threshold)
- Teks **kuning** = confidence < 70%
- Tekan **'q'** untuk keluar

## Spesifikasi Deployment

| Komponen | Detail |
|----------|--------|
| Input | 48×48 grayscale (diresize otomatis) |
| Model | CNN ~322k params (asl_cnn_best.pth) |
| Framework | PyTorch (CPU) |
| FPS estimasi | ~30-60+ fps di CPU (model sangat ringan) |
| Threshold confidence | 70% (dapat diubah di script) |

## Struktur File
```
7_deployment/
├── code/
│   ├── asl_cnn.py          ← Arsitektur model (standalone)
│   ├── predict_image.py    ← Prediksi gambar file
│   └── deploy_webcam.py    ← Webcam real-time
├── images/                 ← Screenshot hasil
└── dokumentasi.md          ← Dokumentasi ini
```

## Catatan
- Model memiliki **99.34% akurasi** pada test set — sangat andal untuk dataset siluet
- Webcam membutuhkan koneksi kamera yang berfungsi (USB/internal)
- Performa real-time sangat baik karena arsitektur ringan (322k parameter)
- Jika background webcam berbeda dari dataset (hitam pekat), pertimbangkan preprocessing tambahan (thresholding, crop tangan)
