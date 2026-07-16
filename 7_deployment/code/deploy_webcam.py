"""
ASL Alphabet — Real-time Webcam Prediction
===========================================
Press 'q' to quit.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from asl_cnn import ASL_CNN

PROJECT_DIR = Path("D:/TA Mesin")
MODEL_PATH = PROJECT_DIR / "5_training/models/asl_cnn_best.pth"
CLASSES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]
IMG_SIZE = (48, 48)
CONFIDENCE_THRESHOLD = 0.7

# Load model once
model = ASL_CNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True))
model.eval()

# Preprocess buffer (reuse to avoid re-allocation)
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    tensor = torch.from_numpy(resized.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0
    return tensor

print(f"[INFO] Loading model from {MODEL_PATH}")
print("[INFO] Starting webcam...")
print("[INFO] Press 'q' to quit")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access webcam")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict
    tensor = preprocess(frame)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred_idx = out.argmax(1).item()
        confidence = probs[pred_idx].item()

    label = CLASSES[pred_idx]

    # Display
    color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
    cv2.putText(frame, f"{label}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 4)
    cv2.putText(frame, f"{confidence*100:.1f}%", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    if confidence < CONFIDENCE_THRESHOLD:
        cv2.putText(frame, "(low confidence)", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("ASL Alphabet - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam stopped.")
