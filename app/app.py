"""
UAS - ASL Sign Language Recognition Web App
Deploy: streamlit run app/app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import time
import joblib
import os
from pathlib import Path

st.set_page_config(page_title="ASL Recognition", layout="wide", page_icon="🤟")

BASE_DIR = Path(__file__).resolve().parent.parent
CLASSES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]

st.sidebar.title("🤟 ASL Recognition")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigasi", [
    "🏠 Dashboard EDA",
    "🧪 Webcam Demo",
    "📸 Upload & Prediksi",
    "📈 Evaluasi Model",
    "🔍 Interpretasi",
    "📚 Dokumentasi"
])
st.sidebar.markdown("---")
st.sidebar.info("UAS Machine Learning - ASL Alphabet Recognition")

# ─── Load Models (cached) ──────────────────────────────
@st.cache_resource
def load_xgboost():
    return joblib.load(str(BASE_DIR / "models/xgb_model.pkl"))

@st.cache_resource
def load_scaler():
    return joblib.load(str(BASE_DIR / "models/scaler.pkl"))

@st.cache_resource
def load_mlp():
    import torch
    import sys
    sys.path.insert(0, str(BASE_DIR / "5_training/code"))
    from train_landmark_mlp import LandmarkMLP
    model = LandmarkMLP(num_classes=29)
    model.load_state_dict(torch.load(
        str(BASE_DIR / "models/landmark_mlp.pth"),
        map_location="cpu", weights_only=True))
    model.eval()
    return model

@st.cache_resource
def load_class_map():
    import pickle
    with open(str(BASE_DIR / "models/class_map.pkl"), "rb") as f:
        return pickle.load(f)

def extract_landmarks_from_image(image_bgr):
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        model_path = str(BASE_DIR / "models/hand_landmarker.task")
        if not Path(model_path).exists():
            return None
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE, num_hands=1, min_hand_detection_confidence=0.3)
        detector = HandLandmarker.create_from_options(options)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        detector.close()
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            return np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float32).flatten()
    except Exception as e:
        st.warning(f"MediaPipe error: {e}")
    return None

def predict_landmarks(features, model_type="XGBoost"):
    scaler = load_scaler()
    features_scaled = scaler.transform(features.reshape(1, -1))
    if model_type == "XGBoost":
        model = load_xgboost()
        pred = model.predict(features_scaled)[0]
        probs = model.predict_proba(features_scaled)[0]
    else:
        model = load_mlp()
        import torch
        with torch.no_grad():
            out = model(torch.from_numpy(features_scaled))
            pred = out.argmax(1).item()
            probs = torch.softmax(out, dim=1)[0].numpy()
    return pred, probs

# ─── PAGE 1: DASHBOARD EDA ─────────────────────────────
if "Dashboard" in page:
    st.title("📊 Dashboard EDA - ASL Alphabet Dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Gambar", "87.028")
    col2.metric("Jumlah Kelas", "29")
    col3.metric("Ukuran Gambar", "200×200")
    col4.metric("Model Terbaik", "XGBoost 93.9%")

    tab1, tab2, tab3 = st.tabs(["Distribusi Kelas", "Sample Gambar", "Ringkasan"])

    with tab1:
        st.image(str(BASE_DIR / "1_eksplorasi_dataset/images/class_distribution.png"),
                 caption="Distribusi 29 kelas ASL", use_container_width=True)
    with tab2:
        st.image(str(BASE_DIR / "1_eksplorasi_dataset/images/sample_grid.png"),
                 caption="Sample per kelas", use_container_width=True)
    with tab3:
        st.image(str(BASE_DIR / "1_eksplorasi_dataset/images/dataset_info.png"),
                 caption="Informasi dataset", use_container_width=True)

    st.subheader("Analisis Landmark")
    st.image(str(BASE_DIR / "reports/eda_summary.png"), use_container_width=True)

# ─── PAGE 2: WEBCAM DEMO ───────────────────────────────
elif "Webcam" in page:
    st.title("🧪 Webcam Demo - Prediksi Real-time")
    st.markdown("Ambil gambar dari webcam browser untuk prediksi ASL.")

    model_choice = st.selectbox("Model", ["XGBoost", "Landmark MLP"])
    img = st.camera_input("Ambil gambar")

    if img is not None:
        bytes_data = img.getvalue()
        nparr = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(frame, channels="BGR", caption="Gambar dari webcam", width=320)

        with col2:
            with st.spinner("Mendeteksi landmark tangan..."):
                features = extract_landmarks_from_image(frame)

            if features is not None:
                pred, probs = predict_landmarks(features, model_choice)
                label = CLASSES[pred]
                confidence = probs[pred]

                st.success(f"**Prediksi: {label}**")
                st.metric("Confidence", f"{confidence*100:.1f}%",
                          delta=f"{'High' if confidence>0.7 else 'Low'} confidence")

                if confidence < 0.6:
                    st.warning("Confidence rendah — coba dengan pencahayaan lebih baik")

                st.write("**Top-3 Prediksi:**")
                top3 = np.argsort(probs)[-3:][::-1]
                for i, idx in enumerate(top3):
                    pct = probs[idx] * 100
                    bar = "█" * int(pct / 5)
                    st.write(f"{i+1}. **{CLASSES[idx]}**: {pct:.1f}% {bar}")
            else:
                st.error("Tidak ada tangan terdeteksi. Coba posisikan tangan di depan kamera.")
                st.info("Tips: gunakan background polos, pencahayaan cukup, tangan di tengah frame.")

# ─── PAGE 3: UPLOAD & PREDIKSI ─────────────────────────
elif "Upload" in page:
    st.title("📸 Upload Gambar & Prediksi")
    st.markdown("Upload gambar gestur tangan ASL untuk prediksi.")

    model_choice = st.selectbox("Model", ["XGBoost", "Landmark MLP"], key="up_model")
    uploaded = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

    if uploaded:
        bytes_data = uploaded.getvalue()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, channels="BGR", caption="Input", width=320)

        with col2:
            if st.button("🔮 Prediksi", type="primary"):
                with st.spinner("Mendeteksi landmark..."):
                    features = extract_landmarks_from_image(img)

                if features is not None:
                    pred, probs = predict_landmarks(features, model_choice)
                    label = CLASSES[pred]
                    confidence = probs[pred]

                    st.success(f"**Hasil: {label}**")
                    st.metric("Confidence", f"{confidence*100:.1f}%")

                    top3 = np.argsort(probs)[-3:][::-1]
                    st.write("**Top-3:**")
                    for i, idx in enumerate(top3):
                        st.write(f"{i+1}. {CLASSES[idx]}: {probs[idx]*100:.1f}%")
                else:
                    st.error("Tangan tidak terdeteksi. Coba gambar lain.")
                    st.info("Atau gunakan input manual landmark di bawah.")

        st.divider()
        st.subheader("Atau input landmark manual (63 nilai)")
        st.caption("Format: x1,y1,z1,x2,y2,z2,...,x21,y21,z21 (dipisah koma)")
        manual_input = st.text_area("Landmark (63 angka comma-separated)", height=80,
                                    placeholder="0.5,0.3,0.1,0.6,0.35,0.12,...")
        if manual_input and st.button("Prediksi Manual"):
            try:
                vals = np.array([float(x.strip()) for x in manual_input.split(",")])
                if len(vals) != 63:
                    st.error(f"Harus 63 nilai, saat ini {len(vals)}")
                else:
                    pred, probs = predict_landmarks(vals, model_choice)
                    st.success(f"**Prediksi: {CLASSES[pred]}**")
                    st.metric("Confidence", f"{probs[pred]*100:.1f}%")
            except Exception as e:
                st.error(f"Error: {e}")

# ─── PAGE 4: EVALUASI MODEL ────────────────────────────
elif "Evaluasi" in page:
    st.title("📈 Evaluasi Model")

    if Path(str(BASE_DIR / "reports/model_comparison.csv")).exists():
        df = pd.read_csv(str(BASE_DIR / "reports/model_comparison.csv"))
        st.subheader("Perbandingan Performa")
        st.dataframe(df.style.highlight_max(subset=['Accuracy', 'F1-Score (macro)']),
                     use_container_width=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        metrics = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)']
        x = np.arange(len(metrics))
        width = 0.2
        colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
        for i, (_, row) in enumerate(df.iterrows()):
            vals = [row[m] for m in metrics]
            ax.bar(x + i*width, vals, width, label=row['Model'], color=colors[i])
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Perbandingan Metrics antar Model', fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.85, 1.0)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        if Path(str(BASE_DIR / "reports/confusion_matrix.png")).exists():
            st.image(str(BASE_DIR / "reports/confusion_matrix.png"), use_container_width=True)
    with col2:
        st.subheader("Feature Importance")
        if Path(str(BASE_DIR / "reports/feature_importance.png")).exists():
            st.image(str(BASE_DIR / "reports/feature_importance.png"), use_container_width=True)

    st.subheader("SHAP Analysis")
    if Path(str(BASE_DIR / "reports/shap_summary.png")).exists():
        st.image(str(BASE_DIR / "reports/shap_summary.png"), use_container_width=True)

# ─── PAGE 5: INTERPRETASI ──────────────────────────────
elif "Interpretasi" in page:
    st.title("🔍 Interpretasi Model & Insights")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Kelebihan Landmark-based")
        st.markdown("""
        - **Background invariant** — model hanya lihat struktur tangan
        - **Ringan** — MLP 18.717 parameter (80 KB)
        - **Cepat** — inference <1ms
        - **Generalizable** — tidak overfit ke background putih
        """)
    with col2:
        st.subheader("❌ Kekurangan")
        st.markdown("""
        - Detection rate MediaPipe rendah (27% di grayscale)
        - Kelas 'P' tidak terdeteksi sama sekali
        - Membutuhkan MediaPipe (dependency tambahan)
        """)

    st.divider()
    st.subheader("Feature Importance Insights")
    st.markdown("""
    | Peringkat | Landmark | Deskripsi |
    |-----------|----------|-----------|
    | 1 | LM4 (ujung ibu jari) | Paling diskriminatif untuk gestur huruf |
    | 2 | LM8 (ujung telunjuk) | Kritis untuk gestur pointing |
    | 3 | LM12 (ujung tengah) | Membedakan gestur vertikal |
    | 4 | LM20 (ujung kelingking) | Penting untuk gestur lebar |
    | 5 | LM0 (pergelangan) | Posisi dasar tangan |
    """)

    st.subheader("Kesimpulan")
    st.success("""
    **Model terbaik: XGBoost** dengan F1-Score 92.88% pada landmark features.
    Pendekatan landmark-based lebih robust daripada CNN pixel-based
    karena invariant terhadap background dan pencahayaan.
    """)

# ─── PAGE 6: DOKUMENTASI ───────────────────────────────
elif "Dokumentasi" in page:
    st.title("📚 Dokumentasi Proyek")

    st.header("Dataset")
    st.markdown("""
    - **ASL Alphabet** dari Kaggle
    - 87.028 gambar, 29 kelas (A-Z, del, nothing, space)
    - Ukuran 200×200 pixel, grayscale
    - [Link Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
    """)

    st.header("Metodologi")
    st.markdown("""
    1. **Feature Extraction** — MediaPipe HandLandmarker (21 landmarks × 3 koordinat = 63 fitur)
    2. **Preprocessing** — Filter undetected, StandardScaler, stratified split
    3. **Modeling** — Logistic Regression, Random Forest, XGBoost, MLP
    4. **Tuning** — Optuna untuk Random Forest
    5. **Interpretasi** — SHAP, Feature Importance, Confusion Matrix
    """)

    st.header("Hasil")
    st.markdown("""
    - **XGBoost**: Accuracy 93.85%, F1 92.88% (model terbaik)
    - **Landmark MLP**: Accuracy 93.89%, F1 92.08%
    - **Random Forest**: Accuracy 90.76%, F1 90.05%
    - **Logistic Regression**: Accuracy 90.65%, F1 89.00%
    """)

    st.header("Cara Deploy")
    st.code("""
    # 1. Clone repo
    git clone https://github.com/rdhinn/UAS_ML-sign-leangue.git
    cd UAS_ML-sign-leangue

    # 2. Install dependencies
    pip install -r requirements.txt

    # 3. Jalankan Streamlit
    streamlit run app/app.py
    """, language="bash")
