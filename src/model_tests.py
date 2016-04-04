import numpy as np
import pandas as pd

from nltk.stem.snowball import SnowballStemmer
import time, sys
from GP_regression import GPregression
from randomforest_regression import RandomForest

stemmer = SnowballStemmer('english')

slice = 1000
df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")[:slice]
df_test = pd.read_csv('../data/test.csv', encoding="ISO-8859-1")[:slice]
# df_attr = pd.read_csv('../input/attributes.csv')
df_pro_desc = pd.read_csv('../data/product_descriptions.csv')[:slice]

num_train = df_train.shape[0]
print 'num_train: ', num_train

def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])


def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())


df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

print df_all.axes

start_time = time.time()
df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x: str_stemmer(x))
print("---Doing Stemming : %s minutes ---" % round(((time.time() - start_time) / 60), 2))


df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)


df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']


df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))

df_all['word_in_description'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))


df_all = df_all.drop(['search_term', 'product_title', 'product_description', 'product_info'], axis=1)

print df_all.axes

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
print df_train.axes, " ss ", df_test.shape[0]
id_test = df_test['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(['id', 'relevance', 'product_uid'], axis=1).values
X_test = df_test.drop(['id', 'relevance', 'product_uid'], axis=1).values

print 'x-train: axes- ', X_train[0:2]
print 'y-train:', y_train[0:2]
print 'x-test axes: ', X_test[0:2]

gpreg = GPregression(X_train, y_train)
gpreg.BuildModel()
y_mean, y_var = gpreg.predict(X_test)

print y_mean.flatten()
print y_var.flatten()

pd.DataFrame({"id": id_test, "relevance": y_mean.ravel()}).to_csv('submission2.csv', index=False)

'''
rf = RandomForest()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv', index=False)
'''
