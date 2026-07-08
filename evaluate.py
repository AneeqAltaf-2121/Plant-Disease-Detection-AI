"""Evaluate the trained model on the validation set."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets

from dataset import VAL_DIR, val_transform
from model import create_model

MODELS_DIR = Path(__file__).resolve().parent / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
IMAGES_DIR = Path(__file__).resolve().parent / "images"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_val_loader(batch_size: int = BATCH_SIZE):
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader, val_dataset.classes


def load_model():
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = create_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return model, class_names


def collect_predictions(model, val_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return all_labels, all_preds


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix — validation set")
    plt.tight_layout()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(IMAGES_DIR / "confusion_matrix.png")
    plt.show()


def evaluate():
    model, class_names = load_model()
    val_loader, _ = get_val_loader()

    all_labels, all_preds = collect_predictions(model, val_loader)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}\n")

    print("Classification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    evaluate()
