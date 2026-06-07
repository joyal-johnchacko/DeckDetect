# 🃏 DeckDetect — AI Playing Card Classifier

> **Live Demo:** [Click here to try DeckDetect](#) ← paste your Render link here

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0+-black?style=flat-square&logo=flask)
![Accuracy](https://img.shields.io/badge/Accuracy-96.98%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 Overview

**DeckDetect** is an AI-powered playing card classifier built with PyTorch and Flask. Upload any playing card image and the model will instantly identify it from all 53 possible classes (Ace through King across all four suits, plus Joker) with high confidence.

Trained using **Transfer Learning** on MobileNetV2, achieving **96.98% validation accuracy** in just 15 epochs on a CPU.

---

## ✨ Features

- 🔍 Identifies all **53 playing card classes** instantly
- ⚡ Prediction in under **1 second**
- 🎨 Neon-themed responsive web UI
- 🧠 MobileNetV2 Transfer Learning backbone
- 🌐 Deployed and accessible via browser — no installation needed

---

## 🧠 Model Architecture

| Component | Detail |
|---|---|
| Base Model | MobileNetV2 (pretrained on ImageNet) |
| Fine-tuned Layers | Last 5 convolutional layers |
| Classifier Head | Dropout → Linear(1280, 512) → ReLU → Linear(512, 53) |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam (lr=0.0005) |
| Scheduler | CosineAnnealingLR |
| Input Size | 224 × 224 RGB |

---

## 📊 Training Results

| Epoch | Train Acc | Val Acc |
|---|---|---|
| 1 | 31.85% | 64.91% |
| 5 | 77.82% | 93.21% |
| 10 | 89.61% | 95.47% |
| 15 | 93.77% | **96.98%** |

> Trained entirely on **CPU** using the Kaggle Cards Image Dataset (7,624 training images across 53 classes)

---

## 🗂️ Project Structure

```
DeckDetect/
│
├── app.py               # Flask web server & prediction API
├── model.py             # MobileNetV2 transfer learning model
├── train.py             # Training pipeline
├── best_model.pth       # Saved model weights (96.98% accuracy)
├── class_names.json     # 53 card class labels
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment config
│
└── templates/
    └── index.html       # Neon UI frontend
```

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/joyal-johnchacko/DeckDetect.git
cd DeckDetect
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

---

## 🌐 API Usage

### Endpoint: `POST /predict`

**Request:**
```
Content-Type: multipart/form-data
Body: file = <image file>
```

**Response:**
```json
{
  "predictions": [
    { "card": "ace of spades", "confidence": 98.5 },
    { "card": "ace of clubs",  "confidence": 0.9 },
    ...
  ]
}
```

---

## 📦 Dataset

- **Source:** [Kaggle — Cards Image Dataset by gpiosenka](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)
- **Training images:** 7,624
- **Validation images:** 265
- **Classes:** 53 (Ace–King × 4 suits + Joker)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | PyTorch + TorchVision |
| Base Model | MobileNetV2 |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render |

---

## ⚠️ Disclaimer

This is not a 100% trained model and can make mistakes, especially with non-standard card designs or low-quality images. Results are AI predictions only.

---

## 👨‍💻 Author

**Joyal Johnchacko**
- GitHub: [@joyal-johnchacko](https://github.com/joyal-johnchacko)

---

## 📄 License

This project is licensed under the MIT License.
