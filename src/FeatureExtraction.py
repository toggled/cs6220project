__author__ = 'Naheed'

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import time, sys
from nltk.util import ngrams
from typodict import get_correction

stemmer = SnowballStemmer('english')
slice = None

if slice:
    df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")[:slice]
    df_test = pd.read_csv('../data/test.csv', encoding="ISO-8859-1")[:slice]
    # df_attr = pd.read_csv('../input/attributes.csv')
    df_pro_desc = pd.read_csv('../data/product_descriptions.csv')[:slice]
    df_attr = pd.read_csv('../data/attributes.csv', encoding='ISO-8859-1')
    df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
else:
    df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('../data/test.csv', encoding="ISO-8859-1")
    # df_attr = pd.read_csv('../input/attributes.csv')
    df_pro_desc = pd.read_csv('../data/product_descriptions.csv')
    df_attr = pd.read_csv('../data/attributes.csv', encoding='ISO-8859-1')
    df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})


def extract_features():
    def str_stemmer(s):
        if isinstance(s, str):
            s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
            return s
        return "null"

    def str_common_word(str1, str2):
        return sum(int(str2.find(word) >= 0) for word in str1.split())

    def ngram_similarity(str1, str2, n=3):

        def jaccard_dis(s1, s2):
            return float(len(s1.intersection(s2))) / len(s1.union(s2))

        str1 = str1.split()
        str2 = str2.split()
        ngram1 = []
        ngram2 = []
        for i in range(n):
            ngram1 = ngram1 + list(ngrams(str1, n - i))

        for i in range(n):
            ngram2 = ngram2 + list(ngrams(str2, n - i))
        return jaccard_dis(set(ngram1), set(ngram2))

    def str_correct(s):
        if get_correction(s.lower()) != None:
            return get_correction(s.lower())
        else:
            return s

    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')

    print df_all.axes

    start_time = time.time()
    df_all['search_term'] = df_all['search_term'].map(lambda x: str_correct(x))
    df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))
    df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))
    df_all['product_description'] = df_all['product_description'].map(lambda x: str_stemmer(x))
    df_all['brand'] = df_all['brand'].map(lambda x: str_stemmer(x))
    print("---Doing Stemming : %s minutes ---" % round(((time.time() - start_time) / 60), 2))

    df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)

    df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all[
        'product_description']

    df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))

    df_all['word_in_description'] = df_all['product_info'].map(
        lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))

    ######################
    df_all['len_of_title'] = df_all['product_title'].map(lambda x: len(x.split())).astype(np.int64)
    df_all['len_of_description'] = df_all['product_description'].map(lambda x: len(x.split())).astype(np.int64)
    df_all['len_of_brand'] = df_all['brand'].map(lambda x: len(x.split())).astype(np.int64)
    #####################
    df_all['attr'] = df_all['search_term'] + "\t" + df_all['brand']
    df_all['word_in_brand'] = df_all['attr'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
    df_all['ratio_brand'] = df_all['word_in_brand'] / df_all['len_of_brand']

    df_all['ngram_similarity_title'] = df_all['product_info'].map(
        lambda x: ngram_similarity(x.split('\t')[0], x.split('\t')[1]))
    df_all['ngram_similarity_description'] = df_all['product_info'].map(
        lambda x: ngram_similarity(x.split('\t')[1], x.split('\t')[1]))

    ########

    df_all = df_all.drop(['search_term', 'product_title', 'product_description', 'product_info', 'attr', 'brand'],
                         axis=1)


def autoload_featurevectors(name):
    """
    :param name: name of the generated csv file containing previously computed feature vectors
    :return: loaded feature set (this reduces repeated computation of features)
    """
    if slice:
        df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")[:slice]
        df_all = pd.read_csv(name, encoding="ISO-8859-1")[:slice]
    else:
        df_train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")[:slice]
        df_all = pd.read_csv(name, encoding="ISO-8859-1")[:slice]
