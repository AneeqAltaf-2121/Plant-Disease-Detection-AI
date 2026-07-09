"""Run predictions on individual leaf images with the trained model."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import val_transform
from model import create_model

MODELS_DIR = Path(__file__).resolve().parent / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = create_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return model, class_names


def predict(image_path: str, model=None, class_names=None):
    if model is None or class_names is None:
        model, class_names = load_model()

    image = Image.open(image_path).convert("RGB")
    input_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    label = class_names[predicted_idx.item()]
    return label, confidence.item()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    predicted_label, predicted_confidence = predict(sys.argv[1])
    print(f"Predicted class: {predicted_label}")
    print(f"Confidence: {predicted_confidence:.4f}")
