import  pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmwrite, mmread
import pickle

df_review_one_sentence = pd.read_csv(
    './crawling/one_sentence_review_2017_2020.csv',
    index_col=0)

Tfidf_matrix = mmread('./models/tfidf_movie_review.mtx').tocsr()
with open('./models/tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    simScore = sorted(simScore, key=lambda x:x[1],
                      reverse=True)
    simScore = simScore[1:11]
    movieidx = [i[0] for i in simScore]
    recMovieList = df_review_one_sentence.iloc[movieidx]
    return recMovieList

#df.iloc[row,col]
#df.loc['tom','math']

# movie_idx = df_review_one_sentence[
#     df_review_one_sentence[
#         'titles']=='라이온 킹 (The Lion King)'].index[0]
movie_idx = 1521           #라이온 킹 (The Lion King)
print(df_review_one_sentence.iloc[movie_idx,0])
cosine_sim = linear_kernel(Tfidf_matrix[movie_idx],
                           Tfidf_matrix)
recommendation = getRecommendation(cosine_sim)
print(recommendation.iloc[:,0])



