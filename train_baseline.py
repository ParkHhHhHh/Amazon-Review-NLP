import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from src.preprocess import clean_text
import os

# ======================================================
# 1. Load Kaggle Amazon Review Dataset
# ======================================================

# 🔥 IMPORTANT:
# Kaggle dataset example:
# https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
#
# Your CSV must contain:
# - reviewText (text)
# - overall (rating: 1~5)

DATA_PATH = "data/amazon_reviews.csv"   

df = pd.read_csv(DATA_PATH)

# ======================================================
# 2. Convert to sentiment label
# ======================================================
# rating 1~2  -> negative (0)
# rating 4~5  -> positive (1)
# rating 3    -> neutral → 제거 or exclude

df = df[ df["overall"] != 3 ]  # remove neutral
df["label"] = df["overall"].apply(lambda x: 1 if x >= 4 else 0)

df = df[["reviewText", "label"]].dropna()

print("Dataset size:", len(df))

# ======================================================
# 3. Clean text
# ======================================================

print("Cleaning text... (this may take a moment)")
df["clean"] = df["reviewText"].apply(clean_text)

# ======================================================
# 4. Train-test split
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42
)

# ======================================================
# 5. TF-IDF Vectorizer
# ======================================================

tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1,2),
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ======================================================
# 6. Baseline Model: Logistic Regression
# ======================================================

model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

preds = model.predict(X_test_tfidf)

print("\n=== Model Accuracy ===")
print(accuracy_score(y_test, preds))
print("\n=== Classification Report ===")
print(classification_report(y_test, preds))

# ======================================================
# 7. Save model & vectorizer
# ======================================================

os.makedirs("models", exist_ok=True)

with open("models/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("models/baseline_lr.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nSaved model and TF-IDF vectorizer to /models/")
print("✔ models/tfidf.pkl")
print("✔ models/baseline_lr.pkl")
