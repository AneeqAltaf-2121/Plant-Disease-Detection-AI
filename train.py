"""Training and evaluation loop for the plant disease classifier."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import create_model

NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
MODELS_DIR = Path(__file__).resolve().parent / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def run_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, running_correct / total


def train():
    train_loader, val_loader, class_names = get_dataloaders()
    model = create_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_accuracy": best_val_accuracy,
                },
                BEST_MODEL_PATH,
            )
            print(f"  Saved new best model (val_acc={best_val_accuracy:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    train()
