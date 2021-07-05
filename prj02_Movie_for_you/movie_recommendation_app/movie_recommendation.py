import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QStringListModel
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmwrite, mmread
import pickle
form_window = uic.loadUiType('./movie_recommend.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.df_review = pd.read_csv(
            '../crawling/movie_review_2017_2020.csv',
            index_col=0)
        self.Tfidf_matrix = mmread(
            '../models/tfidf_movie_review.mtx').tocsr()
        self.embedding_model = Word2Vec.load(
            '../models/word2VecModel_2017_2020.model')
        with open('../models/tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        titles = list(self.df_review.title)
        #titles.sort()
        titles = sorted(titles)
        for title in titles:
            self.cmb_title.addItem(title)

        model = QStringListModel()
        model.setStringList(list(titles))
        completer = QCompleter()
        completer.setModel(model)
        self.le_title.setCompleter(completer)

        self.btn_recommend.clicked.connect(
            self.btn_recommend_slot)
        self.cmb_title.currentIndexChanged.connect(
            self.cmb_title_slot)
    def cmb_title_slot(self):
        title = self.cmb_title.currentText()
        movie_idx = self.df_review[
            self.df_review[
                'title'] == title].index[0]
        cosine_sim = linear_kernel(
            self.Tfidf_matrix[movie_idx],
            self.Tfidf_matrix)
        recommend = '\n'.join(
            list(self.getRecommendation(cosine_sim))[1:])
        self.lbl_result.setText(recommend)

    def getRecommendation(self, cosine_sim):
        simScores = list(enumerate(cosine_sim[-1]))
        simScores = sorted(simScores, key=lambda x: x[1],
                           reverse=True)
        simScores = simScores[0:10]
        movieidx = [i[0] for i in simScores]
        RecMovielist = self.df_review.iloc[movieidx]
        #print(RecMovielist)
        return RecMovielist.title

    def btn_recommend_slot(self):
        title = self.le_title.text()

        if title in list(self.df_review['title']):
            movie_idx = self.df_review[
                self.df_review[
                    'title']==title].index[0]
            cosine_sim = linear_kernel(
                self.Tfidf_matrix[movie_idx],
                self.Tfidf_matrix)
            recommend = '\n'.join(
                list(self.getRecommendation(cosine_sim))[1:])

        else:
            sentence = [title] * 10
            if title in self.embedding_model.wv.vocab:
                sim_word = self.embedding_model.wv.most_similar(title, topn=10)
                labels = []
                for label, _ in sim_word:
                    labels.append(label)
                print(labels)

                for i, word in enumerate(labels):
                    sentence += [word] * (9 - i)

            sentence = ' '.join(sentence)
            sentence_vec = self.Tfidf.transform([sentence])
            cosine_sim = linear_kernel(sentence_vec,
                                       self.Tfidf_matrix)
            recommend = '\n'.join(
                list(self.getRecommendation(cosine_sim))[:-1])
        self.lbl_result.setText(recommend)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Exam()
    w.show()
    sys.exit(app.exec_())
