import pandas as pd

df = pd.read_csv('./crawling/cleaned_review_2020.csv',
                 index_col=0)
df.dropna(inplace=True)
df.to_csv('./crawling/cleaned_review_2020.csv')
one_sentences = []
for idx, title in enumerate(df['titles'].unique()):
    print(idx)
    print(title)
    temp = df[df['titles']==title]['cleaned_sentences']
    one_sentence = ' '.join(temp)
    one_sentences.append(one_sentence)
df_one_sentences = pd.DataFrame(
    {'titles':df['titles'].unique(),
     'reviews':one_sentences})
print(df_one_sentences.head())
df_one_sentences.to_csv('./crawling/one_sentence_review_2020.csv')


















