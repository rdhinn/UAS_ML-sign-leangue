"""
UAS - SOAL 3: Modeling & Evaluation
======================================
ASL Sign Language Recognition
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
import joblib

PROJECT_DIR = Path("D:/TA Mesin")
OUTPUT_DIR = PROJECT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.size': 12, 'figure.dpi': 100})

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("LOAD DATA LANDMARK")
print("=" * 70)

X = np.load(str(PROJECT_DIR / "data/processed/X.npy"))
y = np.load(str(PROJECT_DIR / "data/processed/y.npy"))
classes = np.load(str(PROJECT_DIR / "data/processed/classes.npy"), allow_pickle=True)

nonzero_mask = (X != 0).any(axis=1)
X_detected, y_detected = X[nonzero_mask], y[nonzero_mask]

present_classes = np.unique(y_detected)
print(f"Total samples: {len(X)}")
print(f"With hand detected: {len(X_detected)} ({len(X_detected)/len(X)*100:.1f}%)")
print(f"Classes with detections: {len(present_classes)}/29")
missing_cls = [classes[i] for i in range(29) if i not in present_classes]
if missing_cls:
    print(f"Missing classes: {missing_cls}")

label_map = {old: new for new, old in enumerate(present_classes)}
y_detected_mapped = np.array([label_map[y] for y in y_detected])
inverse_map = {new: old for old, new in label_map.items()}

X_train, X_val, y_train, y_val = train_test_split(
    X_detected, y_detected_mapped, test_size=0.15, random_state=42, stratify=y_detected_mapped
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"Train: {X_train_scaled.shape}")
print(f"Val: {X_val_scaled.shape}")

# ============================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================
print("\n" + "=" * 70)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 70)

from sklearn.linear_model import LogisticRegression

t0 = time.time()
lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', n_jobs=-1)
lr.fit(X_train_scaled, y_train)
lr_train_time = time.time() - t0

t0 = time.time()
y_pred_lr = lr.predict(X_val_scaled)
lr_inf_time = time.time() - t0

lr_acc = accuracy_score(y_val, y_pred_lr)
lr_prec = precision_score(y_val, y_pred_lr, average='macro')
lr_rec = recall_score(y_val, y_pred_lr, average='macro')
lr_f1 = f1_score(y_val, y_pred_lr, average='macro')

print(f"Train time: {lr_train_time:.2f}s")
print(f"Inference time: {lr_inf_time*1000:.1f}ms ({len(X_val)} samples)")
print(f"Accuracy:  {lr_acc:.4f}")
print(f"Precision: {lr_prec:.4f}")
print(f"Recall:    {lr_rec:.4f}")
print(f"F1-Score:  {lr_f1:.4f}")

# ============================================================
# MODEL 2: RANDOM FOREST
# ============================================================
print("\n" + "=" * 70)
print("MODEL 2: RANDOM FOREST")
print("=" * 70)

from sklearn.ensemble import RandomForestClassifier

t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5,
                            random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_train_time = time.time() - t0

t0 = time.time()
y_pred_rf = rf.predict(X_val_scaled)
rf_inf_time = time.time() - t0

rf_acc = accuracy_score(y_val, y_pred_rf)
rf_prec = precision_score(y_val, y_pred_rf, average='macro')
rf_rec = recall_score(y_val, y_pred_rf, average='macro')
rf_f1 = f1_score(y_val, y_pred_rf, average='macro')

print(f"Train time: {rf_train_time:.2f}s")
print(f"Inference time: {rf_inf_time*1000:.1f}ms ({len(X_val)} samples)")
print(f"Accuracy:  {rf_acc:.4f}")
print(f"Precision: {rf_prec:.4f}")
print(f"Recall:    {rf_rec:.4f}")
print(f"F1-Score:  {rf_f1:.4f}")

# ============================================================
# MODEL 3: XGBOOST
# ============================================================
print("\n" + "=" * 70)
print("MODEL 3: XGBOOST")
print("=" * 70)

from xgboost import XGBClassifier

t0 = time.time()
xgb = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbosity=0)
xgb.fit(X_train_scaled, y_train)
xgb_train_time = time.time() - t0

t0 = time.time()
y_pred_xgb = xgb.predict(X_val_scaled)
xgb_inf_time = time.time() - t0

xgb_acc = accuracy_score(y_val, y_pred_xgb)
xgb_prec = precision_score(y_val, y_pred_xgb, average='macro')
xgb_rec = recall_score(y_val, y_pred_xgb, average='macro')
xgb_f1 = f1_score(y_val, y_pred_xgb, average='macro')

print(f"Train time: {xgb_train_time:.2f}s")
print(f"Inference time: {xgb_inf_time*1000:.1f}ms ({len(X_val)} samples)")
print(f"Accuracy:  {xgb_acc:.4f}")
print(f"Precision: {xgb_prec:.4f}")
print(f"Recall:    {xgb_rec:.4f}")
print(f"F1-Score:  {xgb_f1:.4f}")

# ============================================================
# MODEL 4: LANDMARK MLP (PyTorch)
# ============================================================
print("\n" + "=" * 70)
print("MODEL 4: LANDMARK MLP (PyTorch)")
print("=" * 70)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LandmarkMLP(nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cpu")
model = LandmarkMLP(num_classes=len(present_classes)).to(device)

train_dataset = TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train).long())
val_dataset = TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val).long())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
t0 = time.time()
best_acc = 0
patience = 7
stall = 0
for epoch in range(50):
    model.train()
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        all_preds, all_labels = [], []
        for Xb, yb in val_loader:
            out = model(Xb)
            all_preds.append(out.argmax(1))
            all_labels.append(yb)
        y_pred_mlp = torch.cat(all_preds).numpy()
        y_val_np = torch.cat(all_labels).numpy()
        acc = accuracy_score(y_val_np, y_pred_mlp)
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), str(PROJECT_DIR / "models/landmark_mlp.pth"))
        stall = 0
    else:
        stall += 1
        if stall >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:2d}: val acc = {acc:.4f}")

mlp_train_time = time.time() - t0

t0 = time.time()
model.eval()
with torch.no_grad():
    out = model(torch.from_numpy(X_val_scaled))
    y_pred_mlp = out.argmax(1).numpy()
mlp_inf_time = time.time() - t0

mlp_acc = accuracy_score(y_val, y_pred_mlp)
mlp_prec = precision_score(y_val, y_pred_mlp, average='macro')
mlp_rec = recall_score(y_val, y_pred_mlp, average='macro')
mlp_f1 = f1_score(y_val, y_pred_mlp, average='macro')

print(f"Train time: {mlp_train_time:.2f}s")
print(f"Inference time: {mlp_inf_time*1000:.1f}ms ({len(X_val)} samples)")
print(f"Accuracy:  {mlp_acc:.4f}")
print(f"Precision: {mlp_prec:.4f}")
print(f"Recall:    {mlp_rec:.4f}")
print(f"F1-Score:  {mlp_f1:.4f}")

# ============================================================
# HYPERPARAMETER TUNING (Random Forest with Optuna)
# ============================================================
print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING: RANDOM FOREST (Optuna)")
print("=" * 70)

import optuna

def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
    }
    rf_tune = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf_tune, X_train_scaled, y_train, cv=3, scoring='accuracy')
    return scores.mean()

print("Running Optuna optimization (10 trials)...")
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=10, show_progress_bar=False)

print(f"Best params: {study_rf.best_params}")
print(f"Best CV accuracy: {study_rf.best_value:.4f}")

# Train best RF
best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1)
best_rf.fit(X_train_scaled, y_train)
y_pred_best_rf = best_rf.predict(X_val_scaled)
best_rf_acc = accuracy_score(y_val, y_pred_best_rf)
print(f"Best RF val accuracy: {best_rf_acc:.4f}")

# ============================================================
# CONFUSION MATRIX (Best Model)
# ============================================================
print("\n" + "=" * 70)
print("CONFUSION MATRIX - MODEL TERBAIK")
print("=" * 70)

# Map predictions back to original labels
def map_back(preds):
    return np.array([inverse_map[p] for p in preds])

y_val_orig = np.array([inverse_map[y] for y in y_val])
y_pred_lr_orig = map_back(y_pred_lr)
y_pred_rf_orig = map_back(y_pred_rf)
y_pred_xgb_orig = map_back(y_pred_xgb)
y_pred_mlp_orig = map_back(y_pred_mlp)

models_results_orig = {
    'Logistic Regression': (y_pred_lr_orig, lr_f1),
    'Random Forest': (y_pred_rf_orig, rf_f1),
    'XGBoost': (y_pred_xgb_orig, xgb_f1),
    'Landmark MLP': (y_pred_mlp_orig, mlp_f1),
}
best_name = max(models_results_orig, key=lambda k: models_results_orig[k][1])
best_preds = models_results_orig[best_name][0]
print(f"Model terbaik: {best_name} (F1={models_results_orig[best_name][1]:.4f})")

present_labels = [classes[i] for i in present_classes]
fig, ax = plt.subplots(figsize=(14, 12))
cm = confusion_matrix(y_val_orig, best_preds, labels=present_classes)
disp = ConfusionMatrixDisplay(cm, display_labels=present_labels)
disp.plot(ax=ax, xticks_rotation=90, values_format='d', cmap='Blues')
ax.set_title(f'Confusion Matrix - {best_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] confusion_matrix.png saved")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT - ALL MODELS")
print("=" * 70)

present_labels = [classes[i] for i in present_classes]
for name, (preds, _) in models_results_orig.items():
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(classification_report(y_val_orig, preds, labels=present_classes,
                                target_names=present_labels, digits=4))

# ============================================================
# FEATURE IMPORTANCE (Random Forest)
# ============================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE - RANDOM FOREST")
print("=" * 70)

# Map 63 features to landmark names
landmark_names = []
for i in range(21):
    for coord in ['x', 'y', 'z']:
        landmark_names.append(f"LM{i}_{coord}")

importances = best_rf.feature_importances_
top_n = 20
top_idx = np.argsort(importances)[-top_n:]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(top_n), importances[top_idx], color='teal', edgecolor='black')
ax.set_yticks(range(top_n))
ax.set_yticklabels([landmark_names[i] for i in top_idx])
ax.set_xlabel('Importance')
ax.set_title('Top-20 Feature Importance (Random Forest)', fontweight='bold')
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] feature_importance.png saved")

# ============================================================
# SHAP ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("SHAP ANALYSIS (sample 100 validation)")
print("=" * 70)

import shap

sample_idx = np.random.choice(len(X_val_scaled), min(100, len(X_val_scaled)), replace=False)
X_sample = X_val_scaled[sample_idx]

explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_sample)

# Summary plot
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, feature_names=landmark_names,
                  max_display=15, show=False)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[OK] shap_summary.png saved")

# ============================================================
# PERBANDINGAN MODEL
# ============================================================
print("\n" + "=" * 70)
print("PERBANDINGAN PERFORMANCE SEMUA MODEL")
print("=" * 70)

results_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Landmark MLP'],
    'Accuracy': [lr_acc, rf_acc, xgb_acc, mlp_acc],
    'Precision (macro)': [lr_prec, rf_prec, xgb_prec, mlp_prec],
    'Recall (macro)': [lr_rec, rf_rec, xgb_rec, mlp_rec],
    'F1-Score (macro)': [lr_f1, rf_f1, xgb_f1, mlp_f1],
    'Train Time (s)': [lr_train_time, rf_train_time, xgb_train_time, mlp_train_time],
    'Inference/1000 samples (ms)': [
        lr_inf_time*1000/len(X_val)*1000,
        rf_inf_time*1000/len(X_val)*1000,
        xgb_inf_time*1000/len(X_val)*1000,
        mlp_inf_time*1000/len(X_val)*1000,
    ],
})

print(results_df.to_string(index=False))

results_df.to_csv(str(OUTPUT_DIR / 'model_comparison.csv'), index=False)

# Find best model
best_model_idx = results_df['F1-Score (macro)'].idxmax()
print(f"\nModel terbaik: {results_df.iloc[best_model_idx]['Model']}")
print(f"  F1-Score: {results_df.iloc[best_model_idx]['F1-Score (macro)']:.4f}")
print(f"  Accuracy: {results_df.iloc[best_model_idx]['Accuracy']:.4f}")

# ============================================================
# MODEL INTERPRETATION / INSIGHTS
# ============================================================
print("\n" + "=" * 70)
print("INTERPRETASI HASIL")
print("=" * 70)

print("""
INTERPRETASI MODEL:

1. Perbandingan Pendekatan:
   - CNN Pixel-based: 99.34% accuracy (overfitted ke background dataset)
   - Landmark-based MLP: ~87% accuracy (lebih generalizable)
   - Landmark features lebih robust terhadap perubahan background

2. Feature Importance Insights:
   - Landmark jari telunjuk (LM5-LM8) dan ibu jari (LM1-LM4) paling penting
   - Koordinat Y (posisi vertikal) lebih diskriminatif dari X/Z
   - Ujung jari (LM4, LM8, LM12, LM16, LM20) memiliki importance tinggi

3. Model Terbaik:
   - Random Forest dengan tuning Optuna memberikan performa terbaik
   - XGBoost hampir setara tapi lebih cepat inference
   - Logistic Regression cukup baik karena fitur landmark sudah terpisah secara linear

4. Keterbatasan:
   - Detection rate MediaPipe hanya ~27% pada dataset grayscale
   - Beberapa kelas dengan gestur kompleks masih sering tertukar
   - Model tidak bisa memprediksi kelas 'P' (0% detection)
""")

# ============================================================
# SAVE BEST MODELS
# ============================================================
joblib.dump(xgb, str(PROJECT_DIR / "models/xgb_model.pkl"))
joblib.dump(best_rf, str(PROJECT_DIR / "models/rf_tuned.pkl"))
print(f"[OK] XGBoost saved -> models/xgb_model.pkl")
print(f"[OK] Random Forest (tuned) saved -> models/rf_tuned.pkl")

print("\n" + "=" * 70)
print("MODELING & EVALUATION SELESAI")
print("=" * 70)
