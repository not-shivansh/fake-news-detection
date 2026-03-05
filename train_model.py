import os
import re
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import joblib
import nltk

for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("=" * 60)
print("  Fake News Detection - Improved Training Script")
print("=" * 60)

print("\n[1/6] Loading data...")
data_dir = os.path.dirname(os.path.abspath(__file__))
d_fake = pd.read_csv(os.path.join(data_dir, 'Fake.csv'))
d_true = pd.read_csv(os.path.join(data_dir, 'True.csv'))

d_fake['class'] = 0
d_true['class'] = 1

data = pd.concat([d_fake, d_true], axis=0, ignore_index=True)
print(f"   Total samples: {len(data):,} (Fake: {len(d_fake):,}, Real: {len(d_true):,})")

print("\n[2/6] Feature engineering (title + text combined)...")
data['combined'] = data['title'].fillna('') + ' ' + data['text'].fillna('')

print("\n[3/6] Preprocessing text (stopwords + lemmatization)...")
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

data['processed'] = data['combined'].apply(preprocess)
print(f"   Preprocessing complete.")

print("\n[4/6] Splitting data (75% train / 25% test)...")
X = data['processed']
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

print("\n[5/6] Vectorizing with improved TF-IDF (bigrams, sublinear TF)...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_features=150_000,
    analyzer='word'
)
Xv_train = vectorizer.fit_transform(X_train)
Xv_test = vectorizer.transform(X_test)
print(f"   Vocabulary size: {len(vectorizer.vocabulary_):,}")

print("\n[6/6] Training models...\n")

models = {
    'Logistic Regression (tuned)': LogisticRegression(max_iter=1000, C=5.0, solver='lbfgs'),
    'Passive Aggressive':          PassiveAggressiveClassifier(max_iter=1000, C=0.5),
}

results = {}
for name, model in models.items():
    print(f"   Training: {name}...")
    model.fit(Xv_train, y_train)
    preds = model.predict(Xv_test)
    acc = accuracy_score(y_test, preds)
    results[name] = (acc, model)
    print(f"   [OK] Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, preds, target_names=['Fake', 'Real']))

best_name = max(results, key=lambda k: results[k][0])
best_acc, best_model = results[best_name]
print(f"\n  Best model: {best_name}  ({best_acc*100:.2f}%)\n")

model_dir = os.path.join(data_dir, 'model')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.joblib'))
joblib.dump(best_model, os.path.join(model_dir, 'model.joblib'))

import json
meta = {
    'model_name': best_name,
    'accuracy': best_acc,
    'features': 'title + text (combined)',
    'vectorizer': 'TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, max_features=150000)',
    'preprocessing': 'stopwords + lemmatization'
}
with open(os.path.join(model_dir, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"  Model saved to: {model_dir}/")
print(f"  [OK] vectorizer.joblib")
print(f"  [OK] model.joblib")
print(f"  [OK] meta.json")
print("\n" + "=" * 60)
print("  Training complete! Run 'python app.py' to start the web app.")
print("=" * 60 + "\n")
