# Tahap 4: Split Data CNN

## Tujuan
Membagi dataset ASL Alphabet menjadi train/validation/test set untuk training CNN, dengan format siap pakai PyTorch (NCHW).

## Proses
1. **Load** 87.000 gambar dari `data/train/` (29 kelas × 3.000)
2. **Load** 28 gambar holdout dari `data/test/` (flat files, kelas dari nama file)
3. **Resize** ke 48×48 pixel (optimal untuk CPU, resolusi cukup untuk mengenali bentuk tangan)
4. **Normalisasi** pixel ÷ 255 → rentang [0, 1]
5. **Stratified split** — proporsi kelas sama di setiap split

## Hasil Split Data (Stratified 70/15/15)

| Split | Samples | Classes | Shape (NCHW) |
|-------|---------|---------|--------------|
| **Train** | 60.900 | 29 | (60900, 1, 48, 48) |
| **Validation** | 13.050 | 29 | (13050, 1, 48, 48) |
| **Test** | 13.050 | 29 | (13050, 1, 48, 48) |
| **Holdout Test** | 28 | 28 | (28, 1, 48, 48) |

Detail:
- **Train**: 70% = 60.900 sampel (2.100 per kelas)
- **Validation**: 15% = 13.050 sampel (450 per kelas)
- **Test**: 15% = 13.050 sampel (450 per kelas)
- **Holdout**: 28 gambar dari `data/test/` (1 per kelas, minus 1 kelas)

## Output Files (`data/processed/`)

| File | Isi |
|------|-----|
| `train_X.npy` | (60900, 1, 48, 48) float32 — fitur train |
| `train_y.npy` | (60900,) int32 — label train |
| `val_X.npy` | (13050, 1, 48, 48) float32 — fitur validation |
| `val_y.npy` | (13050,) int32 — label validation |
| `test_X.npy` | (13050, 1, 48, 48) float32 — fitur test |
| `test_y.npy` | (13050,) int32 — label test |
| `test_holdout_X.npy` | (28, 1, 48, 48) float32 — fitur holdout |
| `test_holdout_y.npy` | (28,) int32 — label holdout |
| `classes.npy` | (29,) string — nama kelas [A..Z, del, nothing, space] |

## Alasan Resolusi 48×48
- **Efisiensi**: Training CNN di CPU dengan resolusi 200×200 terlalu lambat. 48×48 mengurangi ukuran input ~17× lipat.
- **Kecukupan informasi**: Dataset berupa siluet tangan dengan latar hitam — detail bentuk tangan masih jelas di 48×48.
- **Augmentasi**: Pada tahap training nanti, random crop/rotation tetap bisa dilakukan di resolusi ini.

## Mapping Label
```
0=A, 1=B, ..., 25=Z, 26=del, 27=nothing, 28=space
```

## Catatan
- CNN bekerja langsung dari **raw pixel**, tidak ada feature extraction manual (Tahap 3 di-skip)
- Semua 87.028 gambar terpakai — **tidak ada data loss**
- Holdout test (28 gambar) digunakan untuk evaluasi final model terhadap data yang tidak pernah dilihat sama sekali (termasuk di validation/test split)
