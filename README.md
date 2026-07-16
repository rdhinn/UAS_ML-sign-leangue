# ASL Sign Language Recognition - Machine Learning Capstone

## 📋 Deskripsi Proyek

Sistem pengenalan **American Sign Language (ASL) Alphabet** menggunakan dua pendekatan machine learning:
1. **CNN berbasis pixel** — Klasifikasi gambar grayscale 48x48
2. **MediaPipe Landmarks + MLP** — Ekstraksi 21 landmark tangan (63 fitur) + Multi-Layer Perceptron

Proyek ini dikembangkan sebagai **UAS Machine Learning** dengan fokus pada perbandingan performa, generalisasi, dan deployment real-time.

## 📊 Hasil Utama

| Model | Accuracy | F1-Score | Ukuran | Inference |
|-------|----------|----------|--------|-----------|
| CNN Pixel-based | 99.34% | 99.34% | 5.1 MB | ~5ms |
| Landmark MLP | 87.38% | 87.12% | 80 KB | <1ms |
| Random Forest | ~87% | ~87% | 2.3 MB | <1ms |
| XGBoost | ~87% | ~87% | 1.8 MB | <1ms |

> **Catatan**: CNN 99.34% overfit ke background putih dataset, sedangkan landmark model lebih generalizable.

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Streamlit Dashboard
```bash
streamlit run app/app.py
```

### 3. Webcam Real-time Prediction
```bash
python 7_deployment/code/deploy_landmark_webcam.py
```

### 4. Training Ulang
```bash
python 5_training/code/train_landmark_mlp.py
```

### 5. Evaluasi Model
```bash
python notebooks/02_modeling_evaluation.py
```

## 📁 Struktur Repository

```
├── data/
│   ├── raw/               # Dataset mentah (train/test)
│   └── processed/         # Landmark features (.npy)
├── notebooks/
│   ├── 01_eda_preprocessing.py
│   └── 02_modeling_evaluation.py
├── src/
│   └── utils.py           # Utility functions
├── models/
│   ├── landmark_mlp.pth   # PyTorch MLP model
│   ├── best_model.pkl     # Sklearn best model (RF)
│   ├── scaler.pkl         # StandardScaler
│   └── hand_landmarker.task  # MediaPipe model
├── app/
│   └── app.py             # Streamlit application
├── reports/
│   ├── eda_summary.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── shap_summary.png
├── 1_eksplorasi_dataset/
├── 2_preprocessing/
├── 3_feature_extraction/
├── 4_split_data/
├── 5_training/
├── 6_evaluasi/
├── 7_deployment/
├── requirements.txt
└── README.md
```

## 📈 Metrik Evaluasi

- **Accuracy**: Persentase prediksi benar
- **Precision (macro)**: Rata-rata precision semua kelas
- **Recall (macro)**: Rata-rata recall semua kelas
- **F1-Score (macro)**: Harmonic mean precision & recall
- **Confusion Matrix**: Matriks 29x29 prediksi vs aktual
- **ROC-AUC**: Area Under Curve (one-vs-rest)
- **SHAP**: Feature importance analysis

## 🔗 Link

- **Dataset**: [Kaggle - ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Deployment**: [Streamlit App]()
- **Presentation**: [YouTube]()

## 👨‍💻 Author

Nama Mahasiswa — Universitas

## 📄 Lisensi

CC BY-NC-SA 4.0
