import os
import re
import string
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
VEC_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

def clean_text(text: str) -> str:
    """Preprocess raw text by removing URLs, HTML, punctuation, numbers, and excess whitespace."""
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

def generate_training_data():
    """Generates synthetic news articles categorized into Authentic (1) and Fake/Clickbait (0)."""
    np.random.seed(42)
    data = []

    patterns_real = [
        "Officials from the ministry reported that {topic} has resulted in a {metric}% change over the previous fiscal period.",
        "According to a study published by researchers at {institution}, clinical trials for {treatment} showed significant positive outcomes.",
        "The national meteorological agency issued an advisory regarding {weather_event} expected to affect coastal regions this weekend.",
        "In a formal press briefing, government representatives outlined new regulatory measures for {industry} oversight.",
        "Statistical data released by the labor department indicates consistent employment gains across {sector} industries.",
        "The international scientific consortium finalized peer-reviewed analysis of {discovery} gathered during recent orbital missions.",
        "Local authorities announced scheduled maintenance of civic power grids to improve electrical efficiency and reliability."
    ]

    patterns_fake = [
        "ALERT: Insiders leak undeniable proof that {conspiracy} is taking place right under our noses! Wake up before it's too late!",
        "Miracle discovery: {quackery} cures all known illnesses in 10 minutes flat, but the elites are trying to ban it!",
        "SHOCKING: Leaked secret memo proves {celebrity_or_official} is secretly orchestrating {hoax} across the globe!",
        "Big corporations HATE this simple grandma trick that saves you thousands and exposes the {scam} system!",
        "BOMBSHELL: Proof that {entity} is weaponizing secret frequency rays to control everyone's thoughts!",
        "Doctors stunned as bizarre {substance} eliminates deadly disease overnight with zero side effects!"
    ]

    topics = ["renewable energy", "urban public transit", "agricultural irrigation", "digital infrastructure", "public healthcare"]
    institutions = ["Oxford University", "Stanford Medicine", "MIT Laboratories", "Max Planck Institute", "Johns Hopkins"]
    treatments = ["monoclonal antibody therapy", "targeted mRNA immunotherapy", "cardiovascular intervention", "enzyme replacement therapy"]
    weather_events = ["atmospheric river conditions", "tropical low pressure systems", "moderate seasonal cold fronts"]
    industries = ["telecommunications", "aviation logistics", "pharmaceutical manufacturing", "clean energy utilities"]
    sectors = ["manufacturing", "technology and software", "renewable energy", "healthcare services"]
    discoveries = ["deep-space spectroscopic measurements", "geomagnetic fluctuations", "cryospheric core samples"]

    conspiracies = ["secret government mind-control antennas", "underground cloned politicians", "poisonous synthetic rain"]
    quackeries = ["drinking colloidal silver with celery juice", "staring at the sun at 3 AM", "placing magnets in your shoes"]
    celebrities = ["a top globalist leader", "a Hollywood celebrity", "an anonymous billionaire"]
    hoaxes = ["a staged planetary simulation", "a massive fake reality matrix", "a covert alien takeover"]
    scams = ["banking cartel", "pharma monopoly", "fuel empire"]
    entities = ["secret military shadow organization", "unseen globalist network", "rogue laboratory"]
    substances = ["backyard weed tea", "ancient Himalayan salt paste", "mysterious glowing root"]

    for _ in range(50):
        for p in patterns_real:
            txt = p.format(
                topic=np.random.choice(topics),
                metric=np.random.randint(5, 45),
                institution=np.random.choice(institutions),
                treatment=np.random.choice(treatments),
                weather_event=np.random.choice(weather_events),
                industry=np.random.choice(industries),
                sector=np.random.choice(sectors),
                discovery=np.random.choice(discoveries)
            )
            data.append({"text": txt, "label": 1})

        for p in patterns_fake:
            txt = p.format(
                conspiracy=np.random.choice(conspiracies),
                quackery=np.random.choice(quackeries),
                celebrity_or_official=np.random.choice(celebrities),
                hoax=np.random.choice(hoaxes),
                scam=np.random.choice(scams),
                entity=np.random.choice(entities),
                substance=np.random.choice(substances)
            )
            data.append({"text": txt, "label": 0})

    df = pd.DataFrame(data)
    return df

def train_and_save():
    """Trains TF-IDF + Logistic Regression pipeline and persists artifacts to models/ directory."""
    print("Generating dataset...")
    df = generate_training_data()
    print(f"Total dataset size: {len(df)} samples ({df['label'].value_counts().to_dict()})")

    df['cleaned'] = df['text'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=8000,
        stop_words='english'
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression Model...")
    model = LogisticRegression(C=3.0, max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"Validation Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, preds, target_names=["FAKE", "REAL"]))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    print(f"Saved model to {MODEL_PATH} and vectorizer to {VEC_PATH}")

if __name__ == "__main__":
    train_and_save()
