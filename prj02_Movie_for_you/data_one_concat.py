import pandas as pd

# #중복 제거
# df_dup = pd.read_csv('./crawling/cleaned_review_2020.csv',
#                      index_col=0)
# df_undup = df_dup.drop_duplicates()
# print(df_undup.duplicated().sum())
# df_undup.to_csv('./crawling/cleaned_review_2020.csv')
# exit()

df = pd.read_csv('./crawling/one_sentence_review_2017.csv',
                      index_col=0)
print(df.info())
df.drop_duplicates()
df.dropna(inplace=True)
print(df.info())

df.columns = ['titles', 'reviews']
df.to_csv('./crawling/one_sentence_review_2017.csv')

for i in range(18,22):
    df_temp = pd.read_csv(
        './crawling/one_sentence_review_20{}.csv'.format(i),
        index_col=0)
    df_temp.dropna(inplace=True)
    df_temp.drop_duplicates()
    df_temp.columns = ['titles', 'reviews']
    df_temp.to_csv('./crawling/one_sentence_review_20{}.csv'.format(i))
    df = pd.concat([df, df_temp],ignore_index=True)
print(df.info())
df.to_csv('./crawling/one_sentence_review_2017_2021.csv')



























