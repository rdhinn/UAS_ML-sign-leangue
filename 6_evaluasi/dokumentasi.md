# Tahap 6: Evaluasi Model CNN

## Tujuan
Mengukur performa final model CNN pada test set (13.050 sampel) yang belum pernah dilihat model selama training.

## Hasil Evaluasi

### Test Set Accuracy: **99.34%**

| Metrik | Nilai |
|--------|-------|
| Accuracy | 99.34% |
| Precision (macro avg) | 99.35% |
| Recall (macro avg) | 99.34% |
| F1-Score (macro avg) | 99.34% |

### Performa per Kelas

**7 kelas dengan F1 = 1.000 (sempurna):**
C, P, R, S, K, M, V

**Kelas dengan error tertinggi:**
| Kelas | Precision | Recall | F1-Score | Errors |
|-------|-----------|--------|----------|--------|
| X | 0.9551 | 0.9933 | 0.9739 | 20 |
| Y | 0.9909 | 0.9711 | 0.9809 | 13 |
| Z | 0.9865 | 0.9778 | 0.9821 | 10 |
| T | 0.9932 | 0.9756 | 0.9843 | 11 |
| A | 0.9955 | 0.9800 | 0.9877 | 9 |

### Top 5 Pasangan Sering Salah

| Ground Truth | Predicted | Jumlah Error |
|-------------|-----------|:------:|
| T | X | 9 |
| Y | X | 7 |
| F | B | 6 |
| H | I | 6 |
| Y | Z | 6 |

### Interpretasi
- Model sangat andal dengan **>99% akurasi** pada 29 kelas
- Error dominan terjadi pada huruf dengan **bentuk tangan mirip** (T↔X, Y↔Z, F↔B)
- CNN bekerja sangat baik pada dataset siluet tangan grayscale
- Tidak diperlukan feature extraction manual (MediaPipe)

## Output Files

| File | Deskripsi |
|------|-----------|
| `images/confusion_matrix.png` | Confusion matrix 29x29 |
| `images/sample_predictions.png` | Sample prediksi benar (hijau) vs salah (merah) |
| `classification_report.txt` | Precision, recall, f1 per kelas |
