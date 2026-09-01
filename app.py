import os
import re
import string
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')
VEC_PATH = os.path.join(MODELS_DIR, 'vectorizer.pkl')

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

model = None
vectorizer = None

def load_artifacts():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)
    else:
        from model_trainer import train_and_save
        train_and_save()
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)

load_artifacts()

def clean_text(text: str) -> str:
    """Sanitize and normalize text input."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

SENSATIONAL_WORDS = {
    'shocking', 'miracle', 'secret', 'banned', 'exposed', 'conspiracy', 'cure', 
    'unbelievable', 'urgent', 'disaster', 'hoax', 'proof', 'hidden', 'supernatural',
    'destroy', 'alien', 'illuminati', 'mind-control', 'nanobots', 'plot', 'revealed',
    'forbidden', 'bombshell', 'scam', 'whistleblower'
}

JOURNALISTIC_WORDS = {
    'study', 'published', 'research', 'officials', 'confirmed', 'spokesperson', 
    'organization', 'announced', 'according', 'analysis', 'reported', 'data', 
    'findings', 'institute', 'investigation', 'clinical', 'peer-reviewed', 'regulatory'
}

def analyze_linguistics(raw_text: str):
    """Calculates linguistic metrics: sensationalism score, capitalization ratio, punctuation counts."""
    words = raw_text.split()
    total_words = len(words)
    if total_words == 0:
        return {
            "word_count": 0,
            "char_count": 0,
            "uppercase_ratio": 0,
            "exclamation_count": 0,
            "question_count": 0,
            "sensational_score": 0,
            "sensational_terms_found": [],
            "credible_terms_found": []
        }

    raw_lower = raw_text.lower()
    exclamation_count = raw_text.count('!')
    question_count = raw_text.count('?')
    
    caps_words = [w for w in words if w.isupper() and len(w) > 1]
    uppercase_ratio = round((len(caps_words) / total_words) * 100, 1)

    found_sensational = [w for w in SENSATIONAL_WORDS if re.search(r'\b' + re.escape(w) + r'\b', raw_lower)]
    found_credible = [w for w in JOURNALISTIC_WORDS if re.search(r'\b' + re.escape(w) + r'\b', raw_lower)]

    sensational_score = min(100, int((len(found_sensational) * 20) + (exclamation_count * 5) + (uppercase_ratio * 1.5)))

    return {
        "word_count": total_words,
        "char_count": len(raw_text),
        "uppercase_ratio": uppercase_ratio,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "sensational_score": sensational_score,
        "sensational_terms_found": found_sensational[:8],
        "credible_terms_found": found_credible[:8]
    }

def explain_prediction(cleaned_text: str, pred_label: str):
    """Extracts top contributing TF-IDF feature weights supporting Real or Fake classification."""
    if not vectorizer or not model:
        return []
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_[0]
    
    doc_vec = vectorizer.transform([cleaned_text]).toarray()[0]
    present_indices = np.where(doc_vec > 0)[0]
    
    if len(present_indices) == 0:
        return []

    word_scores = []
    for idx in present_indices:
        term = feature_names[idx]
        weight = coef[idx] * doc_vec[idx]
        word_scores.append({
            "term": term,
            "weight": round(float(weight), 4),
            "supports": "REAL" if weight > 0 else "FAKE"
        })

    word_scores = sorted(word_scores, key=lambda x: abs(x["weight"]), reverse=True)
    return word_scores[:10]

def predict_news(text: str):
    """Main inference function returning prediction verdict, calibrated probability, and explainability."""
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "status": "error",
            "message": "Input text is empty or contains only punctuation/symbols."
        }

    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    prob_fake = float(probs[0])
    prob_real = float(probs[1])

    linguistics = analyze_linguistics(text)

    is_real = prob_real >= 0.5
    verdict = "REAL" if is_real else "FAKE"
    confidence = round((prob_real if is_real else prob_fake) * 100, 2)

    explanations = explain_prediction(cleaned, verdict)

    return {
        "status": "success",
        "verdict": verdict,
        "confidence": confidence,
        "probabilities": {
            "real": round(prob_real * 100, 2),
            "fake": round(prob_fake * 100, 2)
        },
        "linguistics": linguistics,
        "key_factors": explanations
    }

def extract_text_from_file(file_storage):
    """Extracts readable text from uploaded .txt or .pdf files."""
    filename = file_storage.filename.lower()
    if filename.endswith('.txt'):
        return file_storage.read().decode('utf-8', errors='ignore')
    elif filename.endswith('.pdf'):
        reader = PdfReader(file_storage)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    else:
        return file_storage.read().decode('utf-8', errors='ignore')

# ----------------- ROUTES -----------------

@app.route('/')
def home():
    """Renders main dashboard UI."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Web interface prediction endpoint supporting form text, uploaded documents, or JSON."""
    article_text = ""
    
    if request.is_json:
        data = request.get_json()
        article_text = data.get('text', '')
    else:
        article_text = request.form.get('articleText', '')
        
        if 'articleFile' in request.files:
            file = request.files['articleFile']
            if file and file.filename != '':
                extracted = extract_text_from_file(file)
                if extracted.strip():
                    article_text = extracted

    if not article_text.strip():
        return jsonify({"status": "error", "message": "No news content provided. Please enter text or upload a document."}), 400

    result = predict_news(article_text)
    result["preview"] = article_text[:300] + ("..." if len(article_text) > 300 else "")
    return jsonify(result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for programmatic verification."""
    data = request.get_json(force=True, silent=True) or {}
    text = data.get('text', '')
    if not text.strip():
        return jsonify({"error": "Missing 'text' parameter in JSON payload."}), 400
    
    res = predict_news(text)
    return jsonify(res)

@app.route('/samples', methods=['GET'])
def get_samples():
    """Returns preset authentic and fake sample articles."""
    samples = [
        {
            "id": 1,
            "type": "real",
            "title": "NASA Space Telescope Atmospheric Discovery",
            "content": "The James Webb Space Telescope has captured detailed atmospheric spectra of a gas giant exoplanet located 700 light-years away, confirming signatures of water vapor and carbon dioxide according to research published by an international team of astrophysicists."
        },
        {
            "id": 2,
            "type": "real",
            "title": "Federal Reserve Rate Adjustment Report",
            "content": "The Federal Reserve announced a quarter-point moderation in its benchmark federal funds rate following official economic indicators showing sustained stabilization in quarterly consumer price index figures across regional retail sectors."
        },
        {
            "id": 3,
            "type": "fake",
            "title": "Secret Antarctic Pyramid Mind-Control Alert",
            "content": "SHOCKING PROOF: Anonymous whistleblower leaks classified satellite footage exposing a giant alien pyramid hidden beneath the Antarctic ice! Secret global elites have been using microwave frequency lasers to mind-control world leaders! Share this before it gets deleted!"
        },
        {
            "id": 4,
            "type": "fake",
            "title": "Miracle $2 Kitchen Spice Cancer Cure",
            "content": "Doctors are furious! This bizarre $2 kitchen spice destroys 100% of cancer cells in 24 hours while you sleep! Big Pharma is actively trying to ban this ancient Himalayan secret so they can keep profiting off expensive medications!"
        }
    ]
    return jsonify(samples)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
