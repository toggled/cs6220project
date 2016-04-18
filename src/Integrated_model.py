from Lda import Lda
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
import csv
# from GP_regression import GPregression

bins_dir = '../'
df_dir = '/Users/Oyang/Documents/workspace/cs6220/src/'
# if equals 0, use random forest, if equals 1, use gaussian process regression
MODEL_TYPE = 0
# if weighted = 0, chose the model with highest probability, it weighted = 1, use weighted sum
WEIGHTED = 0
# Ture means for running on the fly(usually for the first time), False for running on the local stored data.
realtime = False
K_fold = 2


# assume here all the feature extraction has been finished and the df_all has been produced
def train_data_construct(bins, train_set, iteration, realtime = False):
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
        f1 = file('../data/train_bins'+str(iteration)+'.pkl','wb')
        pk.dump(train_bins,f1)
    else:
        f1 = file('../data/train_bins'+str(iteration)+'.pkl','rb')
        train_bins=pk.load(f1)
    print 'finish constructing training bins'

    return train_bins

def test_data_construct(testset):
    X_test = testset.drop(['product_uid', 'id', 'relevance'], axis=1)
    y_test = testset['relevance']
    return (X_test,y_test)

def select_model(type):
    if type == 0:
        rf = RandomForestRegressor(n_estimators=15, max_depth=10, random_state=0)
        clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
        return rf
    elif type == 1:
        return 0

def weighted_sum(matrix,uids,weight_vector,weighted=1):
    predictions = []
    matrix = matrix.T

    for n in range(len(uids)):
        if weighted == 0:
            vector = matrix[n].A1
            li = weight_vector[uids[n]]
            try:
                pred = vector[np.array(weight_vector[uids[n]]).argmax()]
            except:
                pred = np.mean(matrix[n])
        else:
            li = weight_vector[uids[n]]
            try:
                pred = matrix[n].dot(li).A1[0]
            except:
                pred = np.mean(matrix[n])
        predictions.extend([pred])
    return predictions



if __name__ == '__main__':
    timezero = time()
    num_train = 74000
    df_all = pandas.read_csv(df_dir+'my_df_all.csv')
    # df_all = df_all.drop(['id','search_term', 'product_title', 'product_description', 'product_info', 'attr', 'brand'],
    #                      axis=1)

    """ Assume here that all the df_all only contain the train.csv data,
        since we don't use the test.csv for our project"""

    df = df_all[:num_train].drop(['Unnamed: 0'], axis = 1)
    # df = df_all.drop(['Unnamed: 0'], axis = 1)

    models = defaultdict(object)
    y_pred = defaultdict(list)
    errors = []
    #########change
    # weight_d_t is the documenet-topic probability, which is a list
    f = file('../doc_topic.pkl','rb')
    weight_d_t = pk.load(f)
    ########change
    f1 = file(bins_dir+'topicbins.pkl','rb')
    bins = pk.load(f1)
    newbins = []
    for bin in bins:
        if len(bin)>0:
            newbins.append(bin)

    clf = select_model(MODEL_TYPE)

    ######################
    ## Cross Validation ##
    ######################

    kf = KFold(df.shape[0], n_folds=K_fold)
    iteration = 0
    ## each iteration contains one fold CV, and the result is the RSME for this iteration
    for train_index, test_index in kf:
        result_matrix = []

        print '\nIteration ', iteration, ' starts'
        train_set = df.iloc[train_index]
        test_set = df.iloc[test_index]
        uids = test_set['product_uid'].values
        # print uids
        # print uids[0]

        time0 = time()
        train_bin = train_data_construct(newbins, train_set, iteration, realtime)
        test_data = test_data_construct(test_set)
        print 'train bins prepared, time used: ', time() - time0
        time0_0 = time()
        print 'start to train models'
        for i in range(0, len(train_bin.keys())):
            time0 = time()
            X_train = train_bin[i][0]

            # xx = X_train.drop(['product_uid','id','relevance'],axis=1)
            try:
                # X_train = X_train.drop(['product_uid','Unnamed: 0','relevance'],axis=1).values
                X_train = X_train.drop(['product_uid', 'id', 'relevance'], axis=1).values
            except:
                print i
                continue

            y_train = train_bin[i][1].values

            clf.fit(X_train, y_train)
            #try using Gaussian Processing Regression:
            # if MODEL_TYPE == 1:
            #     clf = GPregression(X_train,y_train)
            #     clf.BuildModel(model='full')
            # else:
            #     clf.fit(X_train, y_train)

            models[i] = clf
            result = clf.predict(test_data[0].values)
            result_matrix.append(result)
            # print 'model ', i , ' trained and predicted, time used: ', time() - time0
        y_predicted = weighted_sum(np.matrix(result_matrix),uids, weight_d_t, WEIGHTED)
        error = np.sqrt(mean_squared_error(y_predicted,test_data[1].values))

        #record the error of each iteration
        errors= errors + [error]

        print 'Iteration ', iteration, ': All models trained, time used:', time() - time0_0
        iteration += 1

        with open('../data/results.csv','a') as f:
            j = 0
            writer = csv.writer(f)
            for i in test_index:
                ll = [i, y_predicted[j],test_data[1].values[j], abs(y_predicted[j]-test_data[1].values[j])]
                writer.writerow(ll)
                j += 1


    error_f = np.mean(errors)

    print '\nJOB DONE: the ', K_fold, ' fold Cross Validation has completed, time used: ', time() - timezero
    print 'The mean of RMSE is: ', error_f

    #Save the trained models
    f1 = file('trained_model.pkl','wb')
    pk.dump(models,f1)