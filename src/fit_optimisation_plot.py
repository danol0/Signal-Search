from utils.distributions import numbaImplementation
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL, BinnedNLL
import pickle
plt.style.use('src/utils/mphil.mplstyle')

# *************************************************************************************************************
# ************************************* Initialise Parameters *************************************************
# *************************************************************************************************************

# set random seed
np.random.seed(0)

# set boundaries
α = 5
β = 5.6
xlim = [α, β]

# create pdf instance
numba = numbaImplementation(*xlim)

pdf = numba.two_signal_pdf
single_peak_pdf = numba.pdf

# define parameters
f1 = 0.1
f2 = 0.05
lam = 0.5
mu1 = 5.28
mu2 = 5.35
sg = 0.018

theta = [f1, f2, lam, mu1, mu2, sg]

# *************************************************************************************************************
# ********************************* Define Naive & Optimised Fits *********************************************
# *************************************************************************************************************


def minuit_fit_optimised(sample, binned=False):
    if binned:
        hist, bin_edges = np.histogram(sample, bins=int(len(sample) ** 0.5), range=xlim)
        nll = BinnedNLL(hist, bin_edges, numba.two_signal_cdf)
    else:
        nll = UnbinnedNLL(sample, numba.two_signal_pdf)

    # H0:
    mi_h0 = Minuit(nll, f1=0.05, f2=0, lam=lam, mu1=5.3, mu2=5.3, sg=0.02)
    mi_h0.limits = [(0, 1), (0, 1), (0, None), (α, β), (α, β), (0.009, 0.06)]
    mi_h0.fixed["f2"] = True
    mi_h0.fixed["mu2"] = True
    mi_h0.simplex()
    mi_h0.migrad()
    mi_h0.hesse()

    # H1:
    mi_h1 = Minuit(nll, *mi_h0.values)
    mi_h1.limits = [(0, 1), (0, 1), (0, None), (α, β), (α, β), (0.009, 0.06)]
    mi_h1.simplex()
    mi_h1.migrad()
    mi_h1.hesse()

    return mi_h0, mi_h1

def minuit_fit_naive(sample, binned=False):
    if binned:
        hist, bin_edges = np.histogram(sample, bins=int(len(sample) ** 0.5), range=xlim)
        nll = BinnedNLL(hist, bin_edges, numba.two_signal_cdf)
    else:
        nll = UnbinnedNLL(sample, numba.two_signal_pdf)

    # H0:
    mi_h0 = Minuit(nll, f1=0.05, f2=0, lam=0.8, mu1=5.3, mu2=5.3, sg=0.02)
    mi_h0.limits = [(0, 1), (0, 1), (0, None), (α, β), (α, β), (0.009, 0.06)]
    mi_h0.fixed["f2"] = True
    mi_h0.fixed["mu2"] = True
    mi_h0.migrad()
    mi_h0.hesse()

    # H1:
    mi_h1 = Minuit(nll, f1=0.05, f2=0.05, lam=0.8, mu1=5.3, mu2=5.3, sg=0.02)
    mi_h1.limits = [(0, 1), (0, 1), (0, None), (α, β), (α, β), (0.009, 0.06)]
    mi_h1.migrad()
    mi_h1.hesse()

    return mi_h0, mi_h1

# *************************************************************************************************************
# ****************************************** Fit and plot the sample ******************************************
# *************************************************************************************************************


with open('src/utils/fit_optimisation.pkl', 'rb') as f:
    ex1_sample, ex2_sample = pickle.load(f)

ex1_h0_opt, ex1_h1_opt = minuit_fit_optimised(ex1_sample, binned=True)
ex1_h0_naive, ex1_h1_naive = minuit_fit_naive(ex1_sample, binned=True)

ex2_h0_opt, ex2_h1_opt = minuit_fit_optimised(ex2_sample, binned=True)
ex2_h0_naive, ex2_h1_naive = minuit_fit_naive(ex2_sample, binned=True)

x = np.linspace(α, β-0.001, 100)

fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharey=True, sharex=True)

ax[0, 0].plot([], [], ' ', label=f'$T$ = {(ex1_h0_naive.fval-ex1_h1_naive.fval):.2f}')
ax[0, 0].plot(x, pdf(x, *ex1_h0_naive.values), label='$H_0$', color='tab:orange')
ax[0, 0].plot(x, pdf(x, *ex1_h1_naive.values), label='$H_1$', linestyle='--', color='tab:blue')
ax[0, 0].hist(ex1_sample, bins=100, density=True, alpha=0.5, label='Sample', color='k', histtype='step')
ax[0, 0].legend()
ax[0, 0].set_title('Naive')
ax[0, 0].set_ylabel('Density')
ax[0, 0].set_xlim(α, β)
plt.suptitle('Sample 1', fontsize=15)

ax[0, 1].plot(x, pdf(x, *ex1_h0_opt.values), color='tab:orange', linewidth=1.5)
ax[0, 1].plot(x, pdf(x, *ex1_h1_opt.values), color='tab:blue', linestyle='--')
ax[0, 1].hist(ex1_sample, bins=100, density=True, alpha=0.5, color='k', histtype='step')
ax[0, 1].plot([], [], ' ', label=f'$T$ = {(ex1_h0_opt.fval-ex1_h1_opt.fval):.2f}')
ax[0, 1].legend()
ax[0, 1].set_title('Optimised')
ax[0, 1].set_xlim(α, β)

ax[1, 0].plot([], [], ' ', label=f'$T$ = {(ex2_h0_naive.fval-ex2_h1_naive.fval):.2f}')
ax[1, 0].plot(x, pdf(x, *ex2_h0_naive.values), color='tab:orange')
ax[1, 0].plot(x, pdf(x, *ex2_h1_naive.values), linestyle='--', color='tab:blue')
ax[1, 0].hist(ex2_sample, bins=100, density=True, alpha=0.5, color='k', histtype='step')
ax[1, 0].legend()
ax[1, 0].set_ylabel('Density')
ax[1, 0].set_xlabel('M')
ax[1, 0].set_title('Naive')
ax[1, 0].set_xlim(α, β)

ax[1, 1].plot(x, pdf(x, *ex2_h0_opt.values), color='tab:orange', linewidth=1.5)
ax[1, 1].plot(x, pdf(x, *ex2_h1_opt.values), color='tab:blue', linestyle='--')
ax[1, 1].hist(ex2_sample, bins=100, density=True, alpha=0.5, color='k', histtype='step')
ax[1, 1].plot([], [], ' ', label=f'$T$ = {(ex2_h0_opt.fval-ex2_h1_opt.fval):.2f}')
ax[1, 1].legend()
ax[1, 1].set_xlabel('M')
ax[1, 1].set_xlim(α, β)
ax[1, 1].set_title('\n\nOptimised')

# Add text in figure coordinates
plt.figtext(0.5, 0.49, 'Sample 2', ha='center', va='center', fontsize=15)

plt.savefig('report/figures/fit_optimisation.png')
