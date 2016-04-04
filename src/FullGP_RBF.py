__author__ = 'Naheed'

import GPy


class FullGP_RBF():
    """
    full GP prediction with rbf kernel
    we can't use optimise method of GPRegression because it work very slowly for input size = 15k
    so we can just create the model, set the parameters infered from HMC and make a prediction
    """

    def __init__(self, X, Y):
        self.model = GPy.models.GPRegression(X, Y)

    # not used anymore
    def setParameters(self, params):
        # params is a list [rbf.variance, rbf.lengthscale, Gaussian_noise.variance ]
        self.model[:] = params

    def predict(self, x_new):
        return self.model.predict(x_new)

    def InferHypersHMC(self, number_of_samples):
        self.model.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1., 10.))
        self.model.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1., 10.))
        self.model.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1., 10.))
        hmc = GPy.inference.mcmc.HMC(self.model, stepsize=5e-2)
        s = hmc.sample(num_samples=number_of_samples)
        cutoff_edge = number_of_samples / 3
        samples = s[cutoff_edge:]  # cut out the burn-in period

        self.model.kern.variance[:] = samples[:, 0].mean()
        self.model.kern.lengthscale[:] = samples[:, 1].mean()
        self.model.likelihood.variance[:] = samples[:, 2].mean()
