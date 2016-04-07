import numpy as np
import pandas as pd

from nltk.stem.snowball import SnowballStemmer
import time, sys
from GP_regression import GPregression
import FeatureExtraction
from randomforest_regression import RandomForest
from SparseGP import SparseGP

autoload = False  # When i dont want to waste time on repeated computation of attribute values
df_all = None
num_train = 0


def check_preprocessed_csv_exists(name):
    import os
    for file in os.listdir(os.getcwd()):
        if file.find(name) >= 0:
            return True
    return False


if not autoload:  # compute attribute values manually
    df_all = FeatureExtraction.extract_features()
    num_train = FeatureExtraction.df_train.shape[0]
    df_all.to_csv('my_df_all_naheed.csv')
else:
    if check_preprocessed_csv_exists("my_df_all_naheed.csv"):
        df_all = FeatureExtraction.autoload_featurevectors(
            "my_df_all_naheed.csv")  # autoload feature vectors from csv file
        num_train = FeatureExtraction.df_train.shape[0]

assert df_all is not None


print df_all.axes

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]

id_test = df_test['id']

'''
y_train = df_train['relevance'].values
X_train = df_train.drop(['id', 'relevance', 'product_uid'], axis=1).values
X_test = df_test.drop(['id', 'relevance', 'product_uid'], axis=1).values
'''
y_train = df_train['relevance'].values
X_train = df_train.drop(['id', 'relevance'], axis=1).values
X_test = df_test.drop(['id', 'relevance'], axis=1).values

print 'x-train: axes- ', X_train[0:2]
print 'y-train:', y_train[0:2]
print 'x-test axes: ', X_test[0:2]

'''
gpreg = GPregression(X_train, y_train)
gpreg.BuildModel("sparse")
y_mean, y_var = gpreg.predict(X_test)

print y_mean.flatten()
print y_var.flatten()

pd.DataFrame({"id": id_test, "relevance": y_mean.ravel()}).to_csv('submission_gp.csv', index=False)
'''

rf = RandomForest()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv', index=False)
