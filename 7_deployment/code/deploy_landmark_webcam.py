import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

PROJECT_DIR = Path("D:/TA Mesin")
MODEL_PATH = PROJECT_DIR / "5_training/models/landmark_mlp.pth"
HAND_MODEL_PATH = PROJECT_DIR / "5_training/models/hand_landmarker.task"
CLASSES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]
N_FEATURES = 63
CONFIDENCE_THRESHOLD = 0.6

N_COLS = 10

class LandmarkMLP(nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def init_hand_detector():
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )
    return HandLandmarker.create_from_options(hand_options)


def extract_landmarks(frame_rgb, detector):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)
    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        flat = np.array(
            [(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32
        ).flatten()
        return flat, result.hand_landmarks[0]
    return None, None


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    for i, j in connections:
        x1, y1 = int(landmarks[i].x * w), int(landmarks[i].y * h)
        x2, y2 = int(landmarks[j].x * w), int(landmarks[j].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def main():
    print("=" * 60)
    print("ASL Alphabet - Landmark MLP Webcam")
    print("=" * 60)

    # --- Load MLP model ---
    print(f"[INFO] Loading MLP model from {MODEL_PATH}...")
    device = torch.device("cpu")
    model = LandmarkMLP(num_classes=len(CLASSES))
    model.load_state_dict(
        torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
    )
    model.eval()
    print("[OK] MLP model loaded.")

    # --- Init MediaPipe ---
    print("[INFO] Initializing MediaPipe HandLandmarker...")
    detector = init_hand_detector()
    print("[OK] MediaPipe ready.")

    # --- Webcam ---
    print("[INFO] Starting webcam (press 'q' to quit)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect landmarks
        features, landmarks = extract_landmarks(rgb, detector)

        if features is not None:
            # Draw landmarks
            draw_landmarks(display, landmarks)

            # Predict
            tensor = torch.from_numpy(features).unsqueeze(0)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)[0]
                pred_idx = out.argmax(1).item()
                confidence = probs[pred_idx].item()

            label = CLASSES[pred_idx]

            # Display prediction
            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.putText(
                display, f"{label}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 4,
            )
            cv2.putText(
                display, f"{confidence*100:.1f}%", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
            )

            if confidence < CONFIDENCE_THRESHOLD:
                cv2.putText(
                    display, "(low confidence)", (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
                )

            # Top-3 predictions
            top3 = torch.topk(probs, 3)
            y_offset = 220
            for i in range(3):
                idx = top3.indices[i].item()
                pct = top3.values[i].item() * 100
                bar_len = int(pct * 2)
                bar = "█" * bar_len
                cv2.putText(
                    display, f"{CLASSES[idx]:8s} {pct:5.1f}% {bar}",
                    (20, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                )
        else:
            cv2.putText(
                display, "No hand detected", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
            )

        cv2.imshow("ASL Alphabet - Landmark MLP (press 'q' to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
