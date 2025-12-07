from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def build_topic_model(texts, n_topics=5):
    cv = CountVectorizer(max_df=0.9, min_df=10, stop_words='english')
    dtm = cv.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method='batch'
    )
    lda.fit(dtm)
    return lda, cv
