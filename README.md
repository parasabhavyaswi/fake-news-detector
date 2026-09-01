# 🛡️ VeritasAI — Fake News Detector

[![Live Demo](https://img.shields.io/badge/LIVE%20DEMO-CLICK%20HERE-success?style=for-the-badge&logo=render&logoColor=white)](https://fake-news-detector-7ir9.onrender.com/)

An AI-powered web application that detects misinformation, clickbait, and authentic news using Natural Language Processing and Machine Learning.

👉 **Live Demo:** [https://fake-news-detector-7ir9.onrender.com/](https://fake-news-detector-7ir9.onrender.com/)

---

## ⚡ Features

- **AI Classification**: TF-IDF + Logistic Regression with confidence scores.
- **Explainable AI**: Highlights key words driving Real vs. Fake verdicts.
- **Linguistic Analysis**: Detects clickbait sensationalism and emotional patterns.
- **Multi-Input**: Paste text or upload `.txt` / `.pdf` documents directly.
- **REST API**: JSON endpoint for programmatic predictions.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python app.py
```
> Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 📡 API Usage

```bash
curl -X POST https://fake-news-detector-7ir9.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA James Webb Space Telescope discovers water vapor on exoplanet."}'
```

---

## 📄 License
MIT License
