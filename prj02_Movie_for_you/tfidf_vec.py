import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread
import pickle

df_review_one_sentences = pd.read_csv(
    './crawling/one_sentence_review_2017_2020.csv',
    index_col=0)
print(df_review_one_sentences.info())

Tfidf = TfidfVectorizer(sublinear_tf=True)
Tfidf_matrix = Tfidf.fit_transform(df_review_one_sentences['reviews'])

with open('./models/tfidf.pickle', 'wb') as f:
    pickle.dump(Tfidf, f)

mmwrite('./models/tfidf_movie_review.mtx',
        Tfidf_matrix)


















