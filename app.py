import streamlit as st
import pandas as pd
import pickle
import nltk
from src.preprocess import clean_text
from src.predict import predict_sentiment
from src.topic_model import build_topic_model


# Download NLTK resources (needed on HF Spaces)
nltk.download("stopwords")
nltk.download("wordnet")

st.title("📦 Amazon Review NLP Intelligence System")
st.write("Sentiment Analysis + Topic Modeling")
st.write("Powered by Logistic Regression + TF-IDF + LDA")

# -------- Sentiment Analysis --------
st.header("🔍 Sentiment Analysis")

review = st.text_area("Enter an Amazon review:")

if st.button("Analyze Sentiment"):
    pred, proba = predict_sentiment(review)

    st.write("### 🎯 Result")
    st.write(f"**Sentiment:** {'Positive' if pred == 1 else 'Negative'}")
    st.write(f"Confidence:** {max(proba):.4f}")

# -------- Topic Modeling --------
st.header("🧠 Topic Modeling (Upload Multiple Reviews)")

uploaded = st.file_uploader("Upload CSV with column `reviewText`")

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Dataset preview:")
    st.write(df.head())

    texts = df["reviewText"].fillna("").tolist()

    st.write("Building topic model... (this may take time)")
    lda, cv = build_topic_model(texts)

    st.write("### 🔎 Topics")
    for idx, topic in enumerate(lda.components_):
        st.write(f"#### Topic {idx}")
        words = [
            cv.get_feature_names_out()[i] 
            for i in topic.argsort()[-10:]
        ]
        st.write(words)

st.write("---")
st.write("Created for Data Science Transfer Portfolio 🚀")
