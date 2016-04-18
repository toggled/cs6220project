import GPy
import numpy as np


class SparseGP():
    def __init__(self, X, Y, num_inducinginputs):
        #input_dim = (X[0].shape)[0]
        #Z = np.random.rand(num_inducinginputs, input_dim) * num_inducinginputs
        #self.model = GPy.models.SparseGPRegression(X, Y, Z=Z)
        self.model = GPy.models.SparseGPRegression(X, Y,num_inducing = num_inducinginputs) # 10 points selected from data
        self.model.optimize('bfgs')

    def predict(self, x_new):
        return self.model.predict(x_new)
