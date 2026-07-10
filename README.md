# Plant Disease Detection AI

Plant Disease Detection AI is a Computer Vision and Deep Learning project built with Python, PyTorch, Torchvision, and FastAPI. The system automatically identifies plant diseases from leaf images using a transfer learning approach based on the ResNet18 convolutional neural network. It performs image preprocessing, data augmentation, model training, evaluation, and real-time disease prediction through both a command-line application and a FastAPI REST API.

---

## Features

- Classifies plant leaf diseases from images
- Uses Transfer Learning with ResNet18
- Performs image preprocessing and normalization
- Applies data augmentation to improve model generalization
- Trains and evaluates a deep learning model
- Saves the best-performing model automatically
- Predicts diseases from individual leaf images
- Command-line interface for local predictions
- FastAPI REST API for real-time image classification
- Interactive Swagger UI for API testing
- Modular project structure for future deployment

---

## Project Structure

```
Plant-Disease-Detection-AI/
│
├── data/
│   └── PlantVillage/
│       ├── train/
│       └── val/
│
├── images/
│   └── confusion_matrix.png
│
├── models/
│
├── notebooks/
│   └── data_exploration.py
│
├── app.py
├── api.py
├── dataset.py
├── evaluate.py
├── model.py
├── predict.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Pillow
- Scikit-Learn
- FastAPI
- Uvicorn
- Transfer Learning
- ResNet18
- Git
- GitHub

---

## Dataset

This project uses the **PlantVillage Dataset**, which contains thousands of labeled images of healthy and diseased plant leaves across multiple crop species.

**Dataset Source:**

https://www.kaggle.com/datasets/mohitsingh1804/plantvillage

Dataset includes:

- 54,303 leaf images
- 38 plant disease classes
- Healthy and diseased leaves
- Predefined training and validation sets

---

## Deep Learning Pipeline

```
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
Best Model Selection
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

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Train the model

```bash
python train.py
```

Predict using the command-line application

```bash
python app.py
```

Enter the path to a plant leaf image when prompted.

Example:

```text
Plant Disease Detection AI

Enter the path to a leaf image:

data\PlantVillage\val\Corn_(maize)___healthy\example.jpg

Predicted disease: Corn_(maize)___healthy
Confidence: 99.97%
```

---

## Example Output

### Command-Line Prediction

<img width="1260" height="530" alt="Prediction" src="https://github.com/user-attachments/assets/4dc86478-53f4-4066-b735-1b699f6283a1" />


---

### Confusion Matrix

<img width="1000" height="1000" alt="confusion_matrix" src="https://github.com/user-attachments/assets/7fead5d1-0928-4642-892e-5a0eb2cb8b79" />


---

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

The best-performing model is automatically saved for future predictions.

---

## FastAPI REST API

Run the API server:

```bash
uvicorn api:app --reload
```

Once the server is running, open:

```text
http://127.0.0.1:8000/docs
```

Swagger UI allows image uploads for real-time disease prediction.

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
- Convolutional Neural Networks (CNNs)
- ResNet18
- Image Preprocessing
- Data Augmentation
- Model Training
- Model Evaluation
- PyTorch
- FastAPI
- REST API Development
- Git & GitHub

---

## Future Improvements

- Web interface using Streamlit
- Mobile application support
- Grad-CAM visualization for model explainability
- Docker deployment
- Cloud deployment (Render / Railway / Azure)
- Support for additional plant species and diseases

---

## Author

**Aneeq Altaf**

GitHub Profile:
https://github.com/AneeqAltaf-2121
