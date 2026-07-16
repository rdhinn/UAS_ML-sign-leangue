---
title: "Problem Definition & Data Acquisition - ASL Sign Language Recognition"
author: "Nama Mahasiswa"
date: "2026"
---

# SOAL 1: Problem Definition & Data Acquisition

## 1. Problem Statement

### Latar Belakang

Bahasa isyarat (Sign Language) merupakan media komunikasi utama bagi penyandang tunarungu dan tunawicara di seluruh dunia. American Sign Language (ASL) adalah salah satu bahasa isyarat yang paling banyak digunakan, dengan lebih dari 500.000 pengguna di Amerika Utara. Namun, terdapat kesenjangan komunikasi yang signifikan antara penyandang tunarungu/wicara dengan masyarakat umum yang tidak memahami bahasa isyarat. Hal ini menciptakan hambatan dalam interaksi sehari-hari, akses layanan publik, dan kesempatan kerja.

Perkembangan teknologi computer vision dan machine learning dalam beberapa tahun terakhir membuka peluang untuk menjembatani kesenjangan tersebut melalui sistem pengenalan bahasa isyarat otomatis (Automatic Sign Language Recognition). Sistem seperti ini dapat memungkinkan penerjemahan real-time dari gestur tangan ke teks atau suara, memfasilitasi komunikasi dua arah tanpa memerlukan penerjemah manusia.

Dataset ASL Alphabet yang tersedia di Kaggle menyediakan 87.000 gambar tangan untuk 29 kelas (26 huruf A-Z, del, nothing, space) dengan berbagai variasi pencahayaan dan posisi tangan. Tantangan utama dalam proyek ini adalah membangun model yang tidak hanya akurat pada dataset benchmark, tetapi juga mampu melakukan generalisasi dengan baik pada input real-time dari webcam yang memiliki kondisi pencahayaan, background, dan posisi tangan yang berbeda.

Pendekatan yang digunakan meliputi:
1. **CNN berbasis pixel** : Model deep learning yang memproses langsung gambar grayscale 48x48
2. **MediaPipe Landmarks + MLP** : Ekstraksi 21 landmark tangan (63 fitur) menggunakan MediaPipe, kemudian diklasifikasikan dengan Multi-Layer Perceptron

Pendekatan kedua (landmark-based) diharapkan lebih robust terhadap variasi background dan pencahayaan karena model hanya melihat struktur geometris tangan, bukan piksel mentah.

### Tujuan Bisnis/Analisis

Tujuan utama dari proyek ini adalah:

1. **Membangun sistem pengenalan ASL real-time** yang dapat mendeteksi 29 kelas gestur tangan (A-Z, del, nothing, space) melalui webcam dengan akurasi tinggi dan latensi rendah.

2. **Membandingkan dua pendekatan modeling**: CNN berbasis pixel vs MLP berbasis landmark MediaPipe, untuk menentukan arsitektur terbaik dalam konteks deployment real-time di perangkat dengan resource terbatas (CPU).

3. **Mengembangkan aplikasi web interaktif** berbasis Streamlit yang memungkinkan pengguna untuk:
   - Melihat visualisasi eksplorasi data
   - Menguji model dengan gambar sendiri
   - Memahami bagaimana model membuat keputusan melalui interpretasi fitur

### Metrik Kesuksesan

Metrik yang digunakan untuk mengevaluasi kesuksesan proyek:

| Metrik | Target | Keterangan |
|--------|--------|------------|
| Test Accuracy | >85% | Akurasi klasifikasi pada test set holdout |
| Precision (macro avg) | >85% | Rata-rata precision untuk semua kelas |
| Recall (macro avg) | >85% | Rata-rata recall untuk semua kelas |
| F1-Score (macro avg) | >85% | Rata-rata F1 untuk semua kelas |
| Inference Time | <100ms per frame | Latensi prediksi di CPU untuk real-time deployment |
| Model Size | <5 MB | Ukuran model untuk deployment yang ringan |

### Referensi Problem Statement

- ASL Alphabet Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- MediaPipe Hand Landmarker: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- American Sign Language (ASL): https://www.nidcd.nih.gov/health/american-sign-language

---

## 2. Data Acquisition

### Sumber Dataset

Dataset yang digunakan adalah **ASL Alphabet Dataset** dari Kaggle, yang tersedia di tautan berikut:

- **URL**: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- **Lisensi**: CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike)
- **Format**: Gambar JPG grayscale ukuran 200x200 pixel
- **Jumlah File**: 87.028 file gambar

### Struktur Dataset

```
data/
├── train/                         # 87.000 gambar (29 kelas x 3000 gambar)
│   ├── A/                         # 3000 gambar huruf A
│   │   ├── A1.jpg
│   │   ├── A2.jpg
│   │   └── ...
│   ├── B/                         # 3000 gambar huruf B
│   ├── ...
│   ├── Z/                         # 3000 gambar huruf Z
│   ├── del/                       # 3000 gambar gesture delete
│   ├── nothing/                   # 3000 gambar gesture nothing
│   └── space/                     # 3000 gambar gesture space
└── test/                          # 28 gambar (1 per kelas)
    ├── A.jpg
    ├── B.jpg
    └── ...
```

### Statistik Deskriptif Awal

| Metrik | Nilai |
|--------|-------|
| Total gambar | 87.028 |
| Jumlah kelas | 29 (A-Z, del, nothing, space) |
| Gambar per kelas (train) | ~3.000 per kelas |
| Gambar test holdout | 28 (1 per kelas) |
| Ukuran gambar | 200 x 200 pixel |
| Channel | Grayscale (1 channel) |
| Format file | JPG |
| Total ukuran dataset | ~1.84 GB |

### Distribusi Data per Kelas

Semua kelas pada dataset train memiliki distribusi yang hampir seragam, yaitu sekitar 3.000 gambar per kelas. Hal ini ideal untuk klasifikasi karena tidak terdapat masalah class imbalance yang signifikan.

### Tools Akuisisi Data

- Dataset diunduh langsung dari Kaggle menggunakan browser
- Tidak diperlukan scraping atau API calls
- Dataset siap digunakan setelah ekstraksi file ZIP

### Etika dan Legalitas Data

Dataset ini dirilis di bawah lisensi **CC BY-NC-SA 4.0**, yang berarti:
- ✓ Dapat digunakan untuk tujuan non-komersial dan akademik
- ✓ Diperbolehkan untuk dimodifikasi dan dibagikan
- ✓ Diwajibkan untuk memberikan atribusi kepada pembuat dataset
- ✗ Tidak boleh digunakan untuk tujuan komersial tanpa izin

---

## 3. Ringkasan

Proyek ini bertujuan membangun sistem pengenalan ASL Alphabet real-time menggunakan dua pendekatan: CNN berbasis pixel dan MLP berbasis landmark MediaPipe. Dataset berasal dari Kaggle dengan 87.000 gambar terdistribusi seragam dalam 29 kelas. Keberhasilan proyek diukur melalui akurasi klasifikasi, precision, recall, F1-score, serta performa real-time (inference time dan model size).
