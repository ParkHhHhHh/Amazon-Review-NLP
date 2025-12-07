import pickle
from src.preprocess import clean_text

tfidf = pickle.load(open("models/tfidf.pkl", "rb"))
model = pickle.load(open("models/baseline_lr.pkl", "rb"))

def predict_sentiment(text):
    clean = clean_text(text)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    return pred, proba
