# TruthLens – Fake News Detector

An AI-powered web application that detects fake news using machine learning. Paste any news article or headline and get an instant **REAL / FAKE** verdict with a confidence score.

## Features

- **ML-powered classification** – Logistic Regression trained on ~45,000 news articles
- **NLP preprocessing** – stopword removal, lemmatization, TF-IDF vectorization with bigrams
- **Confidence score** – percentage confidence for each prediction
- **Modern UI** – dark-mode glassmorphism design with smooth animations
- **REST API** – `/predict` endpoint accepts JSON for programmatic use

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML | scikit-learn (Logistic Regression + TF-IDF) |
| NLP | NLTK (stopwords, WordNetLemmatizer) |
| Frontend | HTML, CSS, Vanilla JS |
| Model Storage | joblib |

## Project Structure

```
├── app.py              # Flask web server
├── train_model.py      # Model training script
├── requirements.txt    # Python dependencies
├── model/
│   ├── model.joblib        # Trained classifier
│   ├── vectorizer.joblib   # TF-IDF vectorizer
│   └── meta.json           # Model metadata (name, accuracy)
├── static/
│   ├── app.js          # Frontend logic
│   └── style.css       # Styles
└── templates/
    └── index.html      # Main page
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/not-shivansh/fake-news-detection.git
cd fake-news-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

The pre-trained model is included — no training required.

```bash
python app.py
```

Visit **http://127.0.0.1:5000** in your browser.

### 4. (Optional) Retrain the model

Download [Fake.csv and True.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle and place them in the project root, then run:

```bash
python train_model.py
```

## API Usage

**POST** `/predict`

```json
{
  "text": "Your news article or headline here..."
}
```

**Response:**

```json
{
  "label": "Fake",
  "confidence": 97.3,
  "word_count": 120,
  "char_count": 742
}
```

## Dataset

The model was trained on the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle (~45k articles). The CSVs are excluded from this repo due to their size.

## Author

Made by **Shivansh Thakur**  
[LinkedIn](https://www.linkedin.com/in/thakur-shivansh/) · [GitHub](https://github.com/not-shivansh)
