import os
import re
import string
import json
import joblib
import numpy as np
import nltk
import os

nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))
from flask import Flask, request, jsonify, render_template

import nltk
for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

print("Loading model...", end=' ', flush=True)
vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.joblib'))
model = joblib.load(os.path.join(MODEL_DIR, 'model.joblib'))

with open(os.path.join(MODEL_DIR, 'meta.json')) as f:
    meta = json.load(f)

print(f"OK  ({meta['model_name']}, accuracy: {meta['accuracy']*100:.2f}%)")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('index.html', model_name=meta['model_name'], accuracy=f"{meta['accuracy']*100:.2f}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    if len(text) < 20:
        return jsonify({'error': 'Text too short. Please provide a full news article or headline.'}), 400

    processed = preprocess(text)
    vec = vectorizer.transform([processed])

    pred = model.predict(vec)[0]

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vec)[0]
        confidence = float(proba[pred])
    elif hasattr(model, 'decision_function'):
        df = model.decision_function(vec)[0]
        confidence = float(1 / (1 + np.exp(-abs(df))))
    else:
        confidence = 1.0

    label = 'Real' if pred == 1 else 'Fake'

    return jsonify({
        'label': label,
        'confidence': round(confidence * 100, 1),
        'word_count': len(text.split()),
        'char_count': len(text)
    })

@app.route('/meta')
def model_meta():
    return jsonify(meta)

if __name__ == '__main__':
    print(f"\n  Fake News Detector running at: http://127.0.0.1:5000\n")
    app.run(debug=False, host='127.0.0.1', port=5000)
