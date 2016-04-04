__author__ = 'Naheed'

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import numpy as np
import pandas as pd
import time


class RandomForest:
    def __init__(self):
        self.rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
        self.clf = BaggingRegressor(self.rf, n_estimators=45, max_samples=0.1, random_state=25)

    def fit(self, X_train, y_train):
        start_time = time.time()
        self.clf.fit(X_train, y_train)
        print("--- Training Model : %s minutes ---" % round(((time.time() - start_time) / 60), 2))

    def predict(self, X_test):
        start_time = time.time()
        y_pred = self.clf.predict(X_test)
        print("--- Testing Model : %s minutes ---" % round(((time.time() - start_time) / 60), 2))
        return y_pred
