# -*- coding: ISO-8859-1 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
import time
from typodict import get_correction
from nltk.util import ngrams
from nltk.metrics.distance import jaccard_distance
from word2vec.scripts_interface import word2vec
import re

stemmer = SnowballStemmer('english')

df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('product_descriptions.csv')
df_attr = pd.read_csv('/Users/Oyang/Documents/workspace/6220/tmp/'+'attributes.csv', encoding='ISO-8859-1')
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

num_train = df_train.shape[0]
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}

def str_stemmer(s):
    if isinstance(s, str):
#         s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A

#         s = s.replace("  "," ")
#         s = s.replace(",","") #could be number / segment later
#         s = s.replace("$"," ")
#         s = s.replace("?"," ")
#         s = s.replace("-"," ")
#         s = s.replace("//","/")
#         s = s.replace("..",".")
#         s = s.replace(" / "," ")
#         s = s.replace(" \\ "," ")
#         s = s.replace("."," . ")
#         s = re.sub(r"(^\.|/)", r"", s)
#         s = re.sub(r"(\.|/)$", r"", s)
#         s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
#         s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
#         s = s.replace(" x "," xbi ")
#         s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
#         s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
#         s = s.replace("*"," xbi ")
#         s = s.replace(" by "," xbi ")
#         s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
#         s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
#         s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
#         s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
#         s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
#         s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
#         s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
#         s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
#         s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
#         s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
#         s = s.replace("°"," degrees ")
#         s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
#         s = s.replace(" v "," volts ")
#         s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
#         s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
#         s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
#         s = s.replace("  "," ")
#         s = s.replace(" . "," ")
#         #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
#         s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
#         
#         s = s.lower()
#         s = s.replace("toliet","toilet")
#         s = s.replace("airconditioner","air conditioner")
#         s = s.replace("vinal","vinyl")
#         s = s.replace("vynal","vinyl")
#         s = s.replace("skill","skil")
#         s = s.replace("snowbl","snow bl")
#         s = s.replace("plexigla","plexi gla")
#         s = s.replace("rustoleum","rust-oleum")
#         s = s.replace("whirpool","whirlpool")
#         s = s.replace("whirlpoolga", "whirlpool ga")
#         s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"

def str_correct(s):
    if get_correction(s.lower()) != None:
        return get_correction(s.lower())
    else:
        return s

def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())

def ngram_similarity(str1, str2, n = 3):
    str1 = str1.split()
    str2 = str2.split()
    ngram1 = []
    ngram2 = []
    for i in range(n):
        ngram1 = ngram1 + list(ngrams(str1,n-i))
    
    for i in range(n):
        ngram2 = ngram2 + list(ngrams(str2,n-i))
    return jaccard_dis(set(ngram1),set(ngram2))
        
def jaccard_dis(s1,s2):
    return float(len(s1.intersection(s2)))/len(s1.union(s2))

def embedding():
    word2vec()

def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')


start_time = time.time()
df_all['search_term'] = df_all['search_term'].map(lambda x: str_correct(x))
df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x: str_stemmer(x))
df_all['brand'] = df_all['brand'].map(lambda x:str_stemmer(x))

print("---Doing Stemming : %s minutes ---" % round(((time.time() - start_time) / 60), 2))

df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
######################
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
#####################
df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']
df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
#####################
df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
#####################
df_all['ngram_similarity_title'] = df_all['product_info'].map(lambda x: ngram_similarity(x.split('\t')[0], x.split('\t')[1]))
df_all['ngram_similarity_description'] = df_all['product_info'].map(lambda x: ngram_similarity(x.split('\t')[1], x.split('\t')[1]))


df_all = df_all.drop(['search_term', 'product_title', 'product_description', 'product_info','attr','brand'], axis=1)

df_all.to_csv('my_df_all.csv')

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(['id', 'relevance'], axis=1).values
X_test = df_test.drop(['id', 'relevance'], axis=1).values

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)

start_time = time.time()
clf.fit(X_train, y_train)
print("--- Training Model : %s minutes ---" % round(((time.time() - start_time) / 60), 2))

start_time = time.time()
y_pred = clf.predict(X_test)
print("--- Testing Model : %s minutes ---" % round(((time.time() - start_time) / 60), 2))

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission2.csv', index=False)
