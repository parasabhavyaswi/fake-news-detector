# 🛡️ VeritasAI — Fake News & Misinformation Detection System

[![Live Demo](https://img.shields.io/badge/LIVE%20DEMO-CLICK%20HERE-success?style=for-the-badge&logo=render&logoColor=white)](https://fake-news-detector-7ir9.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## 🌐 Live Application Link

> ### 🔗 **[Open Live Demo: https://fake-news-detector-7ir9.onrender.com/](https://fake-news-detector-7ir9.onrender.com/)**

---

## ✨ Features

- 🧠 **Machine Learning Classification**: TF-IDF n-gram vectorization paired with Logistic Regression for probability calibration.
- 🔍 **Explainable AI (XAI)**: Highlights influential words and tokens that pushed the prediction towards *Real* or *Fake*.
- 📊 **Linguistic Diagnostics**: Analyzes clickbait sensationalism scores, uppercase emphasis ratio, and punctuation metrics.
- 📄 **Multi-Format Input**: Analyze plain text, pasted headlines, or upload `.txt` and `.pdf` documents directly.
- ⚡ **Instant Sample Loader**: 1-click preset articles for quick demonstration of real vs. fake news.
- 📜 **Session Verification History**: Automatic logging table to track and compare analyzed articles in real time.
- 🌐 **REST API**: Built-in `/api/predict` endpoint for programmatic JSON integrations.

---

## 📁 Repository Structure

```
fake-news-detector/
├── app.py                  # Main Flask application & REST API endpoints
├── model_trainer.py        # Dataset generation & ML model training pipeline
├── requirements.txt        # Python package dependencies
├── Procfile                # Production process declaration for Render
├── .gitignore              # Ignored files and caches
├── models/
│   ├── model.pkl           # Trained Logistic Regression classifier
│   └── vectorizer.pkl      # Fitted TF-IDF vectorizer
├── tests/
│   └── test_app.py         # Automated test suite
├── templates/
│   └── index.html          # Modern dashboard web interface
└── static/
    ├── css/
    │   └── style.css       # Custom styles and dark glassmorphic theme
    └── js/
        └── app.js          # Dynamic UI interactions and API handling
```

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the repository
```bash
git clone https://github.com/parasabhavyaswi/fake-news-detector.git
cd fake-news-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train the ML Model
```bash
python model_trainer.py
```

### 4. Run the Web Application
```bash
python app.py
```
Open your browser and go to: **`http://127.0.0.1:5000`**

### 5. Run Automated Tests
```bash
python tests/test_app.py
```

---

## ☁️ Deployment (Render)

This repository is pre-configured for 1-click deployment on **[Render.com](https://render.com)**:

- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt && python model_trainer.py`
- **Start Command**: `gunicorn app:app`
- **Plan**: `Free`

---

## 📡 REST API Usage

### Endpoint: `POST /api/predict`

#### Request:
```bash
curl -X POST https://fake-news-detector-7ir9.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA James Webb Space Telescope reveals detailed atmospheric composition of exoplanet."}'
```

#### Response:
```json
{
  "status": "success",
  "verdict": "REAL",
  "confidence": 66.71,
  "probabilities": {
    "fake": 33.29,
    "real": 66.71
  },
  "linguistics": {
    "word_count": 13,
    "sensational_score": 0,
    "uppercase_ratio": 7.7
  },
  "key_factors": [
    {
      "term": "telescope",
      "weight": 0.45,
      "supports": "REAL"
    }
  ]
}
```

---

## 📄 License
This project is open-source and available under the **MIT License**.
