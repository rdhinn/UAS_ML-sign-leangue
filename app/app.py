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
import io
from PIL import Image, ImageDraw
import time
import joblib
from pathlib import Path

import sys
import types

class _Cv2Proxy(types.ModuleType):
    def __getattr__(self, name):
        return _Cv2Proxy(f"cv2.{name}")

if 'cv2' not in sys.modules:
    sys.modules['cv2'] = _Cv2Proxy('cv2')
    sys.modules['cv2'].__version__ = '0.0.0'

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

# ─── Singleton: MediaPipe Detector ────────────────────
_mp_detector = None

def get_detector():
    global _mp_detector
    if _mp_detector is not None:
        return _mp_detector
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    model_path = str(BASE_DIR / "5_training/models/hand_landmarker.task")
    if not Path(model_path).exists():
        st.error(f"MediaPipe model not found: {model_path}")
        return None
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE, num_hands=1, min_hand_detection_confidence=0.5)
    _mp_detector = HandLandmarker.create_from_options(options)
    return _mp_detector

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
    cmap = load_class_map()
    n_classes = len(cmap['present_classes'])
    model = LandmarkMLP(num_classes=n_classes)
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

@st.cache_resource
def load_models_for_webcam():
    return load_scaler(), load_xgboost(), load_class_map()

def extract_landmarks_fast(image_rgb, detector):
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        return np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float32).flatten()
    return None

def predict_landmarks(features, model_type="XGBoost"):
    scaler = load_scaler()
    features_scaled = scaler.transform(features.reshape(1, -1))
    cmap = load_class_map()
    if model_type == "XGBoost":
        model = load_xgboost()
        pred_mapped = model.predict(features_scaled)[0]
        pred = cmap['present_classes'][pred_mapped]
        probs_mapped = model.predict_proba(features_scaled)[0]
        probs = np.zeros(29)
        for i, cls in enumerate(cmap['present_classes']):
            probs[cls] = probs_mapped[i]
    else:
        model = load_mlp()
        import torch
        with torch.no_grad():
            out = model(torch.from_numpy(features_scaled))
            pred_mapped = out.argmax(1).item()
            pred = cmap['present_classes'][pred_mapped]
            probs_mapped = torch.softmax(out, dim=1)[0].numpy()
            probs = np.zeros(29)
            for i, cls in enumerate(cmap['present_classes']):
                probs[cls] = probs_mapped[i]
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
                 caption="Distribusi 29 kelas ASL", width='stretch')
    with tab2:
        st.image(str(BASE_DIR / "1_eksplorasi_dataset/images/sample_grid.png"),
                 caption="Sample per kelas", width='stretch')
    with tab3:
        st.image(str(BASE_DIR / "1_eksplorasi_dataset/images/dataset_info.png"),
                 caption="Informasi dataset", width='stretch')
    st.subheader("Analisis Landmark")
    st.image(str(BASE_DIR / "reports/eda_summary.png"), width='stretch')

# ─── PAGE 2: WEBCAM DEMO ───────────────────────────────
elif "Webcam" in page:
    st.title("🧪 Webcam Real-time - ASL Prediction")
    st.markdown("Streaming langsung — gestur ASL diprediksi otomatis.")

    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av

    with st.status("Memuat model...", expanded=True) as status:
        _detector = get_detector()
        _scaler = load_scaler()
        _model = load_xgboost()
        _cmap = load_class_map()
        if _detector is not None:
            st.success(f"MediaPipe HandLandmarker siap")
        else:
            st.error("MediaPipe detector gagal dimuat — cek path model")
        st.success(f"XGBoost siap ({len(_cmap['present_classes'])} kelas)")
        status.update(label="Model siap", state="complete")

    RTC_CONFIG = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    })

    _frame_count = 0

    class ASLProcessor:
        def __init__(self):
            self.detector = _detector
            self.scaler = _scaler
            self.model = _model
            self.cmap = _cmap
            self.fc = 0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            self.fc += 1
            try:
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]

                overlay = Image.fromarray(img[:, :, ::-1])
                draw = ImageDraw.Draw(overlay)

                if self.detector is not None:
                    features = extract_landmarks_fast(img[:, :, ::-1], self.detector)
                    if features is not None:
                        fs = self.scaler.transform(features.reshape(1, -1))
                        pm = self.model.predict(fs)[0]
                        pred = self.cmap['present_classes'][pm]
                        pmb = self.model.predict_proba(fs)[0]
                        label = CLASSES[pred]
                        conf = float(pmb[pm])
                        draw.rectangle([(4, 4), (w - 5, 28)], fill=(0, 0, 0, 180))
                        color = (0, 255, 0) if conf >= 0.6 else (255, 200, 0)
                        draw.text((10, 8), f"{label}  {conf*100:.0f}%", fill=color)
                    else:
                        draw.rectangle([(4, 4), (w - 5, 28)], fill=(0, 0, 0, 180))
                        draw.text((10, 8), "No hand detected", fill=(200, 200, 200))
                else:
                    draw.rectangle([(4, 4), (w - 5, 28)], fill=(0, 0, 0, 180))
                    draw.text((10, 8), "Detector not ready", fill=(255, 100, 100))

                draw.text((w - 80, 8), f"#{self.fc}", fill=(180, 180, 180))
                img = np.array(overlay)[:, :, ::-1]
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception:
                return av.VideoFrame.from_ndarray(frame.to_ndarray(format="bgr24"), format="bgr24")

    st.caption("Tunjukkan gestur ASL di depan kamera. Prediksi muncul sebagai overlay di video.")

    webrtc_streamer(
        key="asl-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        video_processor_factory=ASLProcessor,
        async_processing=True,
    )

    st.info("Tips: Pastikan tangan terlihat jelas, pencahayaan cukup, dan gestur menghadap kamera.")

# ─── PAGE 3: UPLOAD & PREDIKSI ─────────────────────────
elif "Upload" in page:
    st.title("📸 Upload Gambar & Prediksi")
    st.markdown("Upload gambar gestur tangan ASL untuk prediksi.")

    model_choice = st.selectbox("Model", ["XGBoost", "Landmark MLP"], key="up_model")
    uploaded = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = np.array(Image.open(io.BytesIO(uploaded.getvalue())).convert('RGB'))

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Input", width=320)

        with col2:
            if st.button("🔮 Prediksi", type="primary"):
                with st.spinner("Mendeteksi landmark..."):
                    detector = get_detector()
                    if detector:
                        features = extract_landmarks_fast(img, detector)
                    else:
                        features = None

                if features is not None:
                    pred, probs = predict_landmarks(features, model_choice)
                    st.success(f"**Hasil: {CLASSES[pred]}**")
                    st.metric("Confidence", f"{probs[pred]*100:.1f}%")
                    top3 = np.argsort(probs)[-3:][::-1]
                    st.write("**Top-3:**")
                    for i, idx in enumerate(top3):
                        st.write(f"{i+1}. {CLASSES[idx]}: {probs[idx]*100:.1f}%")
                else:
                    st.error("Tangan tidak terdeteksi. Coba gambar lain.")

        st.divider()
        st.subheader("Atau input landmark manual (63 nilai)")
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
                     width='stretch')

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
            st.image(str(BASE_DIR / "reports/confusion_matrix.png"), width='stretch')
    with col2:
        st.subheader("Feature Importance")
        if Path(str(BASE_DIR / "reports/feature_importance.png")).exists():
            st.image(str(BASE_DIR / "reports/feature_importance.png"), width='stretch')

    st.subheader("SHAP Analysis")
    if Path(str(BASE_DIR / "reports/shap_summary.png")).exists():
        st.image(str(BASE_DIR / "reports/shap_summary.png"), width='stretch')

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
    """)

# ─── PAGE 6: DOKUMENTASI ───────────────────────────────
elif "Dokumentasi" in page:
    st.title("📚 Dokumentasi Proyek")

    st.header("Dataset")
    st.markdown("""
    - **ASL Alphabet** dari Kaggle — 87.028 gambar, 29 kelas (A-Z, del, nothing, space)
    - Ukuran 200×200 pixel, grayscale
    - [Link Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
    """)

    st.header("Metodologi")
    st.markdown("""
    1. **Feature Extraction** — MediaPipe HandLandmarker (21 landmarks × 3 koordinat = 63 fitur)
    2. **Preprocessing** — Filter undetected, StandardScaler, stratified split
    3. **Modeling** — Logistic Regression, Random Forest, XGBoost, MLP + Optuna tuning
    4. **Interpretasi** — SHAP, Feature Importance, Confusion Matrix
    """)

    st.header("Hasil")
    st.markdown("""
    - **XGBoost**: Accuracy 93.85%, F1 92.88%
    - **Landmark MLP**: Accuracy 93.89%, F1 92.08%
    """)

    st.header("Cara Deploy")
    st.code("""
    git clone https://github.com/rdhinn/UAS_ML-sign-leangue.git
    cd UAS_ML-sign-leangue
    pip install -r requirements.txt
    streamlit run app/app.py
    """, language="bash")
