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

def plot_function(ax, row, ex_sample, ex_h0_naive, ex_h1_naive, ex_h0_opt, ex_h1_opt):

    ax[row, 0].plot(x, pdf(x, *ex_h1_naive.values), label='$H_1$ Fit', color='tab:orange', linewidth=2)
    ax[row, 0].plot(x, pdf(x, *ex_h0_naive.values), label='$H_0$ Fit', linestyle='--', color='tab:green', linewidth=2)
    ax[row, 0].hist(ex_sample, bins=100, density=True, label='Sample', color='k', histtype='step', alpha=0.35)
    ax[row, 0].set_title('Naive')
    ax[row, 0].set_ylabel('Density')
    ax[row, 0].set_xlim(α, β)
    title = f'$T$ = {(ex_h0_naive.fval-ex_h1_naive.fval):.2f}'
    ax[row, 0].legend(handles=[], title=title)

    ax[row, 1].plot(x, pdf(x, *ex_h1_opt.values), color='orange', linewidth=2)
    ax[row, 1].plot(x, pdf(x, *ex_h0_opt.values), linestyle='--', color='tab:green', linewidth=2)
    ax[row, 1].hist(ex_sample, bins=100, density=True, color='k', histtype='step', alpha=0.35)
    ax[row, 1].set_title('Optimised')
    ax[row, 1].set_xlim(α, β)
    title = f'$T$ = {(ex_h0_opt.fval-ex_h1_opt.fval):.2f}'
    ax[row, 1].legend(handles=[], title=title)



# Load the sample data
with open('src/utils/fit_optimisation.pkl', 'rb') as f:
    ex1_sample, ex2_sample = pickle.load(f)

# Perform the fits
ex1_h0_opt, ex1_h1_opt = minuit_fit_optimised(ex1_sample, binned=True)
ex1_h0_naive, ex1_h1_naive = minuit_fit_naive(ex1_sample, binned=True)

ex2_h0_opt, ex2_h1_opt = minuit_fit_optimised(ex2_sample, binned=True)
ex2_h0_naive, ex2_h1_naive = minuit_fit_naive(ex2_sample, binned=True)

# Create the subplots
fig, ax = plt.subplots(2, 2, figsize=(11, 9), sharey=True, sharex=True)

# Call the plot function for each subplot
plot_function(ax, 0, ex1_sample, ex1_h0_naive, ex1_h1_naive, ex1_h0_opt, ex1_h1_opt)
plot_function(ax, 1, ex2_sample, ex2_h0_naive, ex2_h1_naive, ex2_h0_opt, ex2_h1_opt)

# Trick for displaying separate legend
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[0:1]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='center right', ncol=3, fontsize=15)

# Add text in figure coordinates
plt.suptitle('Sample 1', fontsize=15, x=0.52, y=0.96)
ax[1, 1].set_title('\n\n\nOptimised')
plt.figtext(0.52, 0.45, 'Sample 2', ha='center', va='center', fontsize=15)
plt.savefig('report/figures/fit_optimisation.png')
