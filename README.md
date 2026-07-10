# Plant Disease Detection AI

Plant Disease Detection AI is a Computer Vision and Deep Learning project built with Python, PyTorch, Torchvision, and FastAPI. The system automatically identifies plant diseases from leaf images using transfer learning with a pretrained ResNet18 convolutional neural network. It includes dataset exploration, image preprocessing, data augmentation, model training, evaluation, single-image prediction, a command-line application, and a FastAPI REST API for real-time inference.

---

## Features

- Classifies plant leaf diseases from images
- Uses transfer learning with a pretrained ResNet18 model
- Performs image preprocessing and normalization
- Applies data augmentation to improve model generalization
- Trains and evaluates a deep learning model
- Automatically saves the best-performing model checkpoint
- Predicts diseases from individual leaf images
- Command-line application for local predictions
- FastAPI REST API for real-time image classification
- Interactive Swagger UI for API testing
- Modular architecture with separate training, evaluation, prediction, and deployment components

---

## Project Structure

```text
Plant-Disease-Detection-AI/
│
├── notebooks/
│   └── data_exploration.py
│
├── api.py
├── app.py
├── dataset.py
├── evaluate.py
├── model.py
├── predict.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore

Generated locally:

├── data/
│   └── PlantVillage/
├── models/
│   └── best_model.pth
└── images/
    └── confusion_matrix.png
```

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- Transfer Learning
- NumPy
- Pandas
- Matplotlib
- Pillow
- Scikit-Learn
- FastAPI
- Uvicorn
- Git
- GitHub

---

## Dataset

This project uses the **PlantVillage Dataset**, which contains thousands of labeled images of healthy and diseased plant leaves across multiple crop species.

**Dataset Source**

https://www.kaggle.com/datasets/mohitsingh1804/plantvillage

Dataset includes:

- 54,303 leaf images
- 38 plant disease classes
- Healthy and diseased leaves
- Predefined training and validation sets

After downloading the dataset, organize it as:

```text
data/
└── PlantVillage/
    ├── train/
    └── val/
```

---

## Deep Learning Pipeline

```text
Plant Leaf Image
        │
        ▼
Dataset Exploration
        │
        ▼
Image Preprocessing
        │
        ▼
Data Augmentation
        │
        ▼
Transfer Learning (ResNet18)
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Best Model Checkpoint
        │
        ▼
Prediction
(Command Line & FastAPI)
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/AneeqAltaf-2121/Plant-Disease-Detection-AI.git
```

Navigate into the project

```bash
cd Plant-Disease-Detection-AI
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

**Windows (PowerShell)**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```cmd
venv\Scripts\activate.bat
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Model Checkpoint

The trained model checkpoint (`best_model.pth`) is not included in this repository.

To generate it locally, download the PlantVillage dataset and run:

```bash
python train.py
```

The training script automatically saves the best-performing model to:

```text
models/best_model.pth
```

The evaluation script, prediction pipeline, command-line application, and FastAPI API all use this checkpoint.

---

## Usage

### Train the model

```bash
python train.py
```

The training script automatically saves the checkpoint with the highest validation accuracy and supports early stopping.

### Evaluate the model

```bash
python evaluate.py
```

The evaluation script reports:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report

and generates a confusion matrix.

### Command-line prediction

```bash
python app.py
```

Enter the path to a leaf image when prompted.

Example:

```text
Plant Disease Detection AI

Enter the path to a leaf image:

data\PlantVillage\val\Corn_(maize)___healthy\example.jpg

Predicted disease: Corn_(maize)___healthy
Confidence: 99.97%
```

### FastAPI

Run:

```bash
python -m uvicorn api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Upload a leaf image using the Swagger UI to receive a prediction.

---

## Example Output

### Command-Line Prediction

<img width="1260" height="530" alt="Prediction" src="https://github.com/user-attachments/assets/4dc86478-53f4-4066-b735-1b699f6283a1" />

---

### FastAPI Swagger UI

*(Insert your Swagger screenshot here.)*

---

### Confusion Matrix

<img width="1000" height="1000" alt="confusion_matrix" src="https://github.com/user-attachments/assets/7fead5d1-0928-4642-892e-5a0eb2cb8b79" />

---

## API Response

Example response:

```json
{
  "predicted_class": "Corn_(maize)___healthy",
  "confidence": 0.9996761083602905
}
```

---

## Skills Demonstrated

- Computer Vision
- Deep Learning
- Transfer Learning
- PyTorch
- Convolutional Neural Networks (CNNs)
- Image Preprocessing
- Data Augmentation
- Model Training
- Model Evaluation
- FastAPI
- REST API Development
- Git & GitHub

---

## Future Improvements

- Streamlit web interface
- Mobile application integration
- Grad-CAM model explainability
- Docker containerization
- Cloud deployment (Azure, Render, Railway)
- Support for additional plant species and diseases

---

## Author

**Aneeq Altaf**

GitHub: https://github.com/AneeqAltaf-2121