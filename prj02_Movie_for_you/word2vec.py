import pandas as pd
from gensim.models import Word2Vec

review_word = pd.read_csv(
    './crawling/cleaned_review_2017_2020.csv',
    index_col=0)
#print(review_word.info())
cleaned_token_review = list(review_word['cleaned_reviews'])
#print(len(cleaned_token_review))
cleaned_tokens = []
count = 0
for sentence in cleaned_token_review:
    token = sentence.split(' ')
    cleaned_tokens.append(token)
#print(len(cleaned_tokens))
#print(cleaned_token_review[0])
#print(cleaned_tokens[0])
embedding_model = Word2Vec(cleaned_tokens, vector_size=100,
                    window=4, min_count=20,
                    workers=4, epochs=100, sg=1)
embedding_model.save('./models/word2VecModel_2017_2020.model')
print(embedding_model.wv.vocab.keys())
print(len(embedding_model.wv.vocab.keys()))





