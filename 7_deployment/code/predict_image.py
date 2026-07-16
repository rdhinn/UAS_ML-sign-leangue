"""
ASL Alphabet Prediction — Single Image
=======================================
Usage: python predict_image.py <image_path>
"""

import sys
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


def predict(img_path: str):
    # Load & preprocess
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Cannot read {img_path}")
        return
    img_resized = cv2.resize(img, IMG_SIZE)
    img_tensor = torch.from_numpy(img_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0

    # Load model
    model = ASL_CNN(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True))
    model.eval()

    # Predict
    with torch.no_grad():
        out = model(img_tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred_idx = out.argmax(1).item()
        confidence = probs[pred_idx].item()

    label = CLASSES[pred_idx]

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence*100:.2f}%")

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"(Below threshold {CONFIDENCE_THRESHOLD*100:.0f}% — low confidence)")

    # Show top-3
    top3 = torch.topk(probs, 3)
    print("\nTop-3 predictions:")
    for i in range(3):
        idx = top3.indices[i].item()
        print(f"  {i+1}. {CLASSES[idx]:10s} {top3.values[i].item()*100:.2f}%")

    # Display
    display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(display, f"{label} ({confidence*100:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("ASL Prediction", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])
