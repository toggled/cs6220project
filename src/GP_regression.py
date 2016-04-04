__author__ = 'Naheed'

import FullGP_RBF
import numpy as np
import pandas as pd
import time


class GPregression:
    def __init__(self, trainx, trainy):
        self.trainx = trainx
        self.trainy = np.array([[i] for i in trainy])

    def BuildModel(self):
        start_time = time.time()
        self.model = FullGP_RBF.FullGP_RBF(self.trainx, self.trainy)
        self.model.InferHypersHMC(1000)
        print("--- Training Model : %s minutes ---" % round(((time.time() - start_time) / 60), 2))

    def predict(self, testsetx):
        start_time = time.time()
        self.p_mean, self.p_variance = self.model.predict(testsetx)
        print("--- Testing Model : %s minutes ---" % round(((time.time() - start_time) / 60), 2))
        return self.p_mean, self.p_variance

    def geterror(self, testsety):
        error = ((testsety - self.p_mean) ** 2).mean()
