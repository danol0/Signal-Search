from utils.distributions import numbaImplementation, analyticalImplementation
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
from tqdm import tqdm
from scipy.stats import chi2
import matplotlib.gridspec as gs
import pickle

# set random seed
np.random.seed(3)

# set boundaries
α = 5
β = 5.6

# create instances
af = analyticalImplementation(α, β)
nf = numbaImplementation(α, β)

# define parameters
f1 = 0.1
f2 = 0.05
lam = 0.5
mu1 = 5.28
mu2 = 5.35
sg = 0.018

theta  = [f1, f2, lam, mu1, mu2, sg]


# *************************************************************************************************************
# ************************************ Set Up Simulation Study ************************************************
# *************************************************************************************************************

def two_peak_inverse_cdf_generator(f1, f2, samples):
    u = np.random.random(samples)
    n = np.random.random(samples)
    events = [nf.signal_inv_cdf(u[i], mu1, sg) if n[i] < f1 
              else nf.signal_inv_cdf(u[i], mu2, sg) if n[i] < f1 + f2 
              else nf.background_inv_cdf(u[i], lam) for i in range(samples)]
    return events


# fit function: fits h0 and h1 to the sample and returns the Minuit instances
def unbinned_fit(sample):
    nll = UnbinnedNLL(sample, nf.pdf)

    # H0: f1 = 0
    mi_h0 = Minuit(nll, f=0, lam=0.55, mu=5.3, sg=0.02)
    mi_h0.limits = [(0, 10), (0, None), (α, β), (0, None)]
    #mi_h0.errors = [0.1, 0.1, 0.1, 0.1]
    mi_h0.fixed["f"] = True
    mi_h0.fixed["mu"] = True
    mi_h0.fixed["sg"] = True
    mi_h0.errordef = Minuit.LIKELIHOOD
    mi_h0.migrad()

    # H1: f = 0.1
    mi_h1 = Minuit(nll, f=0.11, lam=0.55, mu=5.3, sg=0.02)
    mi_h1.limits = [(0, 10), (0, None), (α, β), (0, None)]
    #mi_h0.errors = [0.1, 0.1, 0.1, 0.1]
    mi_h1.errordef = Minuit.LIKELIHOOD
    mi_h1.migrad()

    return mi_h0, mi_h1
