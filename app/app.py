"""
UAS - Streamlit Application: ASL Sign Language Recognition
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import time
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="ASL Recognition", layout="wide")
PROJECT_DIR = Path(__file__).parent.parent

CLASSES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]

st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", [
    "Dashboard EDA",
    "Model Demo",
    "Evaluasi Model",
    "Interpretasi Hasil",
    "Dokumentasi"
])

# ─────────────────────────────────────────────
# PAGE 1: DASHBOARD EDA
# ─────────────────────────────────────────────
if page == "Dashboard EDA":
    st.title("📊 Dashboard EDA - ASL Alphabet Dataset")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Gambar", "87.028")
    col2.metric("Jumlah Kelas", "29")
    col3.metric("Ukuran Gambar", "200x200")
    
    st.subheader("Distribusi Kelas")
    st.image(str(PROJECT_DIR / "1_eksplorasi_dataset/images/class_distribution.png"))
    
    st.subheader("Sample Gambar per Kelas")
    st.image(str(PROJECT_DIR / "1_eksplorasi_dataset/images/sample_grid.png"))
    
    st.subheader("Ringkasan Dataset")
    st.image(str(PROJECT_DIR / "1_eksplorasi_dataset/images/dataset_info.png"))

    st.subheader("Insight Visualisasi")
    insight_tab1, insight_tab2 = st.tabs(["Landmark Analysis", "Detection Rate"])
    with insight_tab1:
        st.image(str(PROJECT_DIR / "reports/eda_summary.png"))
    
    with insight_tab2:
        X = np.load(str(PROJECT_DIR / "data/processed/X.npy"))
        y = np.load(str(PROJECT_DIR / "data/processed/y.npy"))
        classes = np.load(str(PROJECT_DIR / "data/processed/classes.npy"), allow_pickle=True)
        nonzero_mask = (X != 0).any(axis=1)
        
        detect_rates = []
        for i in range(len(classes)):
            cls_mask = (y == i)
            rate = (nonzero_mask & cls_mask).sum() / cls_mask.sum() * 100
            detect_rates.append({"Kelas": classes[i], "Detection Rate (%)": round(rate, 1)})
        
        df = pd.DataFrame(detect_rates)
        st.dataframe(df, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 2: MODEL DEMO
# ─────────────────────────────────────────────
elif page == "Model Demo":
    st.title("🧪 Model Demo - Prediksi ASL")
    
    model_type = st.selectbox("Pilih Model", ["Landmark MLP", "Random Forest", "XGBoost"])
    
    input_method = st.radio("Metode Input", ["Upload Gambar", "Generate Sample Landmark"])
    
    if input_method == "Upload Gambar":
        uploaded = st.file_uploader("Upload gambar tangan", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="Input Image", width=300)
            
            if st.button("Prediksi"):
                with st.spinner("Processing..."):
                    time.sleep(1)
                    st.info("Untuk prediksi real-time, gunakan webcam deployment: deploy_landmark_webcam.py")
    else:
        st.subheader("Input Landmark Features (63 nilai)")
        st.info("Masukkan 63 nilai landmark (format: x1,y1,z1,x2,y2,z2,...)")
        input_text = st.text_area("Landmark values (comma-separated)", height=100)
        
        if st.button("Prediksi"):
            try:
                vals = np.array([float(x.strip()) for x in input_text.split(",")])
                if len(vals) != 63:
                    st.error(f"Expected 63 values, got {len(vals)}")
                else:
                    scaler = joblib.load(str(PROJECT_DIR / "models/scaler.pkl"))
                    vals_scaled = scaler.transform(vals.reshape(1, -1))
                    
                    if model_type == "Random Forest":
                        model = joblib.load(str(PROJECT_DIR / "models/rf_tuned.pkl"))
                    elif model_type == "XGBoost":
                        model = joblib.load(str(PROJECT_DIR / "models/xgb_model.pkl"))
                    else:
                        import torch
                        import sys
                        sys.path.insert(0, str(PROJECT_DIR / "5_training/code"))
                        from train_landmark_mlp import LandmarkMLP
                        model_mlp = LandmarkMLP(num_classes=29)
                        model_mlp.load_state_dict(torch.load(
                            str(PROJECT_DIR / "models/landmark_mlp.pth"),
                            map_location="cpu", weights_only=True))
                        model_mlp.eval()
                        with torch.no_grad():
                            out = model_mlp(torch.from_numpy(vals_scaled))
                            pred = out.argmax(1).item()
                            probs = torch.softmax(out, dim=1)[0].numpy()
                        st.success(f"Prediction: **{CLASSES[pred]}**")
                        st.metric("Confidence", f"{probs[pred]*100:.1f}%")
                        st.stop()
                    
                    pred = model.predict(vals_scaled)[0]
                    probs = model.predict_proba(vals_scaled)[0]
                    
                    st.success(f"Prediction: **{CLASSES[pred]}**")
                    st.metric("Confidence", f"{probs[pred]*100:.1f}%")
                    
                    top3 = np.argsort(probs)[-3:][::-1]
                    st.write("Top-3 Predictions:")
                    for i, idx in enumerate(top3):
                        st.write(f"  {i+1}. {CLASSES[idx]}: {probs[idx]*100:.1f}%")
            except Exception as e:
                st.error(f"Error: {e}")

# ─────────────────────────────────────────────
# PAGE 3: EVALUASI MODEL
# ─────────────────────────────────────────────
elif page == "Evaluasi Model":
    st.title("📈 Evaluasi Model")
    
    if Path(str(PROJECT_DIR / "reports/model_comparison.csv")).exists():
        df = pd.read_csv(str(PROJECT_DIR / "reports/model_comparison.csv"))
        st.subheader("Perbandingan Performa Model")
        st.dataframe(df, use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        metrics = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)']
        x = np.arange(len(metrics))
        width = 0.2
        for i, model in enumerate(df['Model']):
            ax.bar(x + i*width, [df.iloc[i][m] for m in metrics], width, label=model)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Score')
        ax.set_title('Perbandingan Metrics antar Model')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.subheader("Confusion Matrix")
    if Path(str(PROJECT_DIR / "reports/confusion_matrix.png")).exists():
        st.image(str(PROJECT_DIR / "reports/confusion_matrix.png"))
    
    st.subheader("Feature Importance")
    if Path(str(PROJECT_DIR / "reports/feature_importance.png")).exists():
        st.image(str(PROJECT_DIR / "reports/feature_importance.png"))
    
    st.subheader("SHAP Analysis")
    if Path(str(PROJECT_DIR / "reports/shap_summary.png")).exists():
        st.image(str(PROJECT_DIR / "reports/shap_summary.png"))

# ─────────────────────────────────────────────
# PAGE 4: INTERPRETASI HASIL
# ─────────────────────────────────────────────
elif page == "Interpretasi Hasil":
    st.title("🔍 Interpretasi Model & Insight Bisnis")
    
    st.header("Kesimpulan Analisis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Kelebihan Landmark-based")
        st.markdown("""
        - **Background invariant**: Model hanya melihat struktur tangan, bukan pixel
        - **Ringan**: MLP hanya 18.717 parameter (80 KB)
        - **Cepat**: Inference <1ms per frame
        - **Generalizable**: Tidak overfit ke background putih dataset
        """)
    with col2:
        st.subheader("❌ Kekurangan")
        st.markdown("""
        - **Detection rate rendah**: MediaPipe hanya detect 27% di grayscale
        - **Kelas 'P' tidak terdeteksi**: 0% detection rate
        - **Akurasi lebih rendah**: ~87% vs 99% CNN pixel (tapi lebih realistis)
        - **Tergantung MediaPipe**: Tambahan dependency
        """)
    
    st.header("Rekomendasi Bisnis")
    st.markdown("""
    1. **Gunakan hybrid approach**: CNN pixel + Landmark MLP ensemble
    2. **Real-time deployment**: Landmark MLP cocok untuk edge device (CPU)
    3. **Data collection**: Tambah variasi background dan lighting di training
    4. **User experience**: Tambah feedback visual (hand tracking) seperti di deploy_landmark_webcam.py
    """)
    
    st.header("Feature Importance Insights")
    st.markdown("""
    Berdasarkan analisis Random Forest dan SHAP:
    - **Landmark ujung jari** (index 4, 8, 12, 16, 20) paling penting
    - **Koordinat Y** lebih diskriminatif dari X dan Z
    - **Ibu jari (LM1-LM4)** penting untuk membedakan gestur huruf
    """)

# ─────────────────────────────────────────────
# PAGE 5: DOKUMENTASI
# ─────────────────────────────────────────────
elif page == "Dokumentasi":
    st.title("📚 Dokumentasi Proyek")
    
    st.header("Dataset: ASL Alphabet")
    st.markdown("""
    - **Sumber**: Kaggle (https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
    - **Lisensi**: CC BY-NC-SA 4.0
    - **Total gambar**: 87.028 (training 87.000 + test 28)
    - **Kelas**: 29 (A-Z, del, nothing, space)
    - **Ukuran**: 200x200 pixel, grayscale
    """)
    
    st.header("Metodologi")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Data Acquisition")
        st.write("Dataset dari Kaggle, ekstraksi landmark via MediaPipe")
    with col2:
        st.subheader("2. EDA & Preprocessing")
        st.write("Analisis distribusi, kualitas data, scaling, train/val split")
    with col3:
        st.subheader("3. Modeling")
        st.write("4 model: LR, RF, XGBoost, MLP + tuning Optuna")
    
    st.header("Cara Menggunakan")
    st.code("""
    # Install dependencies
    pip install -r requirements.txt
    
    # Run Streamlit app
    streamlit run app/app.py
    
    # Run webcam deployment
    python 7_deployment/code/deploy_landmark_webcam.py
    """)
    
    st.header("Struktur Repository")
    st.code("""
    TA Mesin/
    ├── data/          # Dataset
    ├── notebooks/     # EDA & Modeling notebooks
    ├── src/           # Source code
    ├── models/        # Trained models
    ├── app/           # Streamlit app
    ├── reports/       # Laporan & visualisasi
    ├── 1-7_*/          # Per tahap pipeline
    └── requirements.txt
    """)
