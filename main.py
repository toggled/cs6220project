from src.Lda import Lda
from src.SecondPhase import SecondPhase
from collections import defaultdict
import cPickle as pk
import pandas
import numpy as np
from time import time
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error



__author__ = 'Naheed'

bins_dir = '/Users/Oyang/Documents/workspace/6220/tmp/'
df_dir = 'src/'
# if equals 0, use random forest, if equals 1, use gaussian process regression
MODEL_TYPE = 1
# Ture means for running on the fly(usually for the first time), False for running on the local stored data.
realtime = False
K_fold = 2
topic_bins = []
load_pickled = True  # Whether want to load pickled bins from pk file

def run_phaseone():
    if load_pickled:
        import pickle
        with open('./test/topicbins.pk') as f:
            topic_bins = pickle.load(f)

    else:
        ld = Lda()
        ld.runlda()
        topic_bins = ld.bin  # List of List(product_id)

    for i in topic_bins:
        if i:
            print len(i)

def run_phasetwo():
    '''
    TO DO:
     1. Extract Features from training set, product description and attributes of the docs belonging to each bin.
     2. Train K models on them.
     3.
    '''

    process = SecondPhase()
    timezero = time()
    num_train = 74067
    df_all = pandas.read_csv(df_dir + 'my_df_all.csv')
    # df_all = df_all.drop(['id','search_term', 'product_title', 'product_description', 'product_info', 'attr', 'brand'],
    #                      axis=1)

    """ Assume here that all the df_all only contain the train.csv data,
        since we don't use the test.csv for our project"""

    df = df_all[:num_train].drop(['Unnamed: 0'], axis=1)
    # df = df_all.drop(['Unnamed: 0'], axis = 1)

    models = defaultdict(object)
    errors = []
    # weight_d_t is the documenet-topic probability, which is a list
    weight_d_t = []

    f1 = file('topicbins.pk', 'rb')
    bins = pk.load(f1)
    newbins = []
    for bin in bins:
        if len(bin) > 0:
            newbins.append(bin)

    print 'number of non-empty bins: ',len(newbins)

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

        time0 = time()
        train_bin = process.train_data_construct(newbins, train_set, iteration, realtime)
        test_data = process.test_data_construct(test_set)
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

            clf = process.select_model(MODEL_TYPE,X_train,y_train)
            if MODEL_TYPE == 0:
                clf.fit(X_train, y_train)
            else:
                clf.BuildModel(model='sparse')

            # try using Gaussian Processing Regression:
            # if MODEL_TYPE == 1:
            #     clf = GPregression(X_train, y_train)
            #     clf.BuildModel(model='full')
            # else:
            #     clf.fit(X_train, y_train)


            models[i] = clf
            if MODEL_TYPE == 1:
                mean,var = clf.predict(test_data[0].values)
                result = mean.flatten()
            else:
                result = clf.predict(test_data[0].values)

            result_matrix.append(result)
            print 'model ', i , ' trained and predicted, time used: ', time() - time0
        if MODEL_TYPE == 0:
            y_predicted = process.weighted_sum(np.matrix(result_matrix), weight_d_t)
        else:
            y_predicted = process.weighted_sum(np.matrix(result_matrix), weight_d_t)

        error = np.sqrt(mean_squared_error(y_predicted, test_data[1]))

        errors = errors + [error]

        print 'Iteration ', iteration, ': All models trained, time used:', time() - time0_0
        iteration += 1
    error_f = np.mean(errors)

    print '\nJOB DONE: the ', K_fold, ' fold Cross Validation has completed, time used: ', time() - timezero
    print 'The mean of RMSE is: ', error_f



def get_prediction():
    '''
    Function which will predict the rank for test query.
    TO DO:
     1. Get the product title and compute similarity with the topics.
     2. assign normalized similarity as weights to each to topics./
     3. Compute features based on Product title,search term
     4. Get Rank from K Models and compute weighted average
    '''
    pass


if __name__ == '__main__':
    run_phasetwo()
