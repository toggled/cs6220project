from pandas import DataFrame
from pandas import Series
from collections import defaultdict
import cPickle as pk
import pandas
import numpy as np
from time import time
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from GP_regression import GPregression

class SecondPhase():
    def __init__(self,):
        pass
    # assume here all the feature extraction has been finished and the df_all has been produced
    def train_data_construct(self, bins, train_set, iteration, realtime = False):
        train_bins = defaultdict(tuple)

        print 'start to construct the train data bins'
        if realtime:
            idx = 0
            for bin in bins:
                if len(bin) > 0:
                    feature_bin = DataFrame()
                    lable_bin = Series()
                    for uid in bin:
                        tmp = train_set[train_set['product_uid'] == int(uid)]
                        if not tmp.empty:
                            feature_bin = feature_bin.append(tmp)
                            # should drop the relevance data here
                            lable_bin = lable_bin.append(tmp['relevance'])
                    train_bins[idx] = (feature_bin,lable_bin)
                    print len(train_bins[idx][0]), ' entries in bin', idx
                    # if idx == 0:
                    #     feature_bin.to_csv('feature_bin.csv')
                    idx += 1
            f1 = file('data/train_bins'+str(iteration)+'.pkl','wb')
            pk.dump(train_bins,f1)
        else:
            f1 = file('data/train_bins'+str(iteration)+'.pkl','rb')
            train_bins=pk.load(f1)
        print 'finish constructing training bins'

        return train_bins

    def test_data_construct(self, testset):
        X_test = testset.drop(['product_uid', 'id', 'relevance'], axis=1)
        y_test = testset['relevance']
        return (X_test,y_test)

    def select_model(self, type,X_train=[],y_train=[]):
        if type == 0:
            rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
            clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
            return rf
        elif type == 1:
            return GPregression(X_train, y_train)


    def weighted_sum(self, matrix, weight_vector):
        matrix = matrix.T
        if len(weight_vector) == 0:
            return matrix.mean(axis=1)
        else:
            ws = matrix.dot(weight_vector).A1
            return ws

