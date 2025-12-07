## 📦 Amazon Review NLP Project
Sentiment Analysis + Topic Modeling + Streamlit Web App

This project analyzes Amazon product reviews using Natural Language Processing (NLP).
I built a full pipeline that includes text preprocessing, sentiment classification, topic modeling, and an interactive Streamlit dashboard.
This project is part of my Data Science transfer application portfolio.

## 🚀 1. Project Overview

Amazon product reviews contain valuable information, but they are unstructured text.
To make them easier to understand, I built a system that:

✔ Predicts whether a review is positive or negative
✔ Shows a confidence score
✔ Finds common topics in many reviews
✔ Provides a web interface for users

(Implemented using Streamlit and deployed on HuggingFace Spaces)

This project helped me practice:

Python NLP processing

Classical machine learning

Unsupervised topic modeling

Model deployment and UI design

## 📊 2. Dataset

Source: Kaggle — Amazon Reviews Dataset

I used the fields:

reviewText (written review)

overall (rating from 1 to 5)

Labeling

To simplify the classification task:

1–2 → Negative (0)

4–5 → Positive (1)

3 → removed (neutral)

This helps the model focus on clear examples.

## 🧹 3. Preprocessing

Before training the model, each review is cleaned:

Lowercase conversion

Removing special characters

Stopword removal

Lemmatization

These steps improve the quality of the text features.

## 🤖 4. Sentiment Classification Model

I used a simple but effective classical machine learning approach:

TF-IDF for text vectorization

Logistic Regression for classification

This baseline model is fast, interpretable, and works surprisingly well for text.
Test accuracy is usually around 90% depending on the dataset size.

Model files saved:

models/tfidf.pkl

models/baseline_lr.pkl

## 🧠 5. Topic Modeling

For analyzing many reviews at once, I used:

LDA (Latent Dirichlet Allocation)

Extracts major topics and keywords

Helps understand general trends in reviews (e.g., battery, size, quality)

These topics are shown inside the Streamlit dashboard.

## 🖥️ 6. Streamlit Web Application

The web app includes:

1) Single Review Analysis

Enter any review text

See sentiment prediction

View confidence score

2) CSV Upload for Topic Modeling

Upload a CSV with a reviewText column

The app extracts top words for each topic

Helps summarize user feedback quickly

This app is deployed on HuggingFace Spaces.

## 🧪 7. How to Run the App Locally
1) Install dependencies
pip install -r requirements.txt

2) Train the model (optional)
python train_baseline.py

3) Run the app
streamlit run app.py

## 📁 8. Folder Structure


📦 amazon-review-nlp
 ┣ app.py
 ┣ requirements.txt
 ┣ train_baseline.py
 ┣ src/
 │   ┣ preprocess.py
 │   ┣ predict.py
 │   ┗ topic_model.py
 ┣ models/
 │   ┣ tfidf.pkl
 │   ┗ baseline_lr.pkl
 ┗ data/
     ┗ amazon_reviews.csv

## 📈 9. Results
Component	Summary
Model	Logistic Regression + TF-IDF
Accuracy	~0.90 (varies by dataset)
Topic Modeling	LDA (5 topics default)
Deployment	HuggingFace Spaces + Streamlit

Example prediction:

Review: "The product works well, but shipping was slow."
Sentiment: Positive  
Confidence: 0.84

## 🔧 10. Skills I Practiced

Python (Pandas, scikit-learn, NLTK)

Building machine learning pipelines

Working with real-world text data

Topic modeling (LDA)

Creating user-friendly dashboards

Deploying AI apps on the web

## 🔭 11. Future Improvements

Add DistilBERT for better sentiment accuracy

Use BERTopic for more advanced topic modeling

Add results visualization (bar charts, word clouds)

Improve error analysis

## 📬 Contact

Email:roy040315@gmail.com
GitHub: http://
