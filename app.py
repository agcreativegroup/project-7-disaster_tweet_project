from flask import Flask, render_template, request
import joblib
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
# Load model and preprocessing tools
model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
scaler = joblib.load("models/scaler.pkl")

# Feature columns in correct order
feature_cols = ['char_count', 'word_count', 'hashtag_count', 'mention_count', 'neg', 'neu', 'pos', 'compound']
history = []
# Define basic clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# Simulate sentiment score (you can replace with actual VADER)
def real_sentiment(text):
    return sia.polarity_scores(text)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        tweet = request.form["tweet"]
        cleaned = clean_text(tweet)
        tfidf_vec = vectorizer.transform([cleaned])
        
        # Extract features
        sentiment = real_sentiment(tweet)
        meta = pd.DataFrame([{
            'char_count': len(tweet),
            'word_count': len(tweet.split()),
            'hashtag_count': tweet.count("#"),
            'mention_count': tweet.count("@"),
            **sentiment
        }])[feature_cols]

        scaled_meta = scaler.transform(meta)
        final_input = hstack((tfidf_vec, scaled_meta))

        pred = model.predict(final_input)[0]
        result = "Disaster Tweet" if pred == 1 else " Not a Disaster Tweet"
        
        #Append only if tweet exists (POST method)
        history.append((tweet, result))

    return render_template("index.html",
                       result=result,
                       history=list(reversed(history[-5:])))

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)