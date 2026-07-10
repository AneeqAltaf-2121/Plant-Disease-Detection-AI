"""FastAPI REST API for serving the Plant Disease Detection model."""

import io

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from dataset import val_transform
from predict import device, load_model

app = FastAPI(title="Plant Disease Detection API")

model, class_names = load_model()


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Plant Disease Detection API is running."}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")

    input_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    return {
        "predicted_class": class_names[predicted_idx.item()],
        "confidence": confidence.item(),
    }
