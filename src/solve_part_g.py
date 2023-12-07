from utils.distributions import numbaImplementation
from utils.tools import simulation_study
from iminuit.cost import UnbinnedNLL, BinnedNLL
from iminuit import Minuit
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
import matplotlib.gridspec as gs
import pickle
plt.style.use("src/utils/mphil.mplstyle")

# *************************************************************************************************************
# ************************************* Initialise Parameters *************************************************
# *************************************************************************************************************

# set random seed
np.random.seed(0)

# set boundaries
α = 5
β = 5.6
xlim = [α, β]

# create instance
numba = numbaImplementation(*xlim)

# define parameters
f1 = 0.1
f2 = 0.05
lam = 0.5
mu1 = 5.28
mu2 = 5.35
sg = 0.018
theta = [f1, f2, lam, mu1, mu2, sg]

# *************************************************************************************************************
# ************************************* Define Hypothesis Test ************************************************
# *************************************************************************************************************


# fit function: fits h0 and h1 to the sample and returns the Minuit instances
def two_peak_test(sample, binned=False):
    """
    This function carries out the hypothesis test for two peaks and returns
    the Minuit instances for H0 and H1.

    Methodology is discussed in section 3.3.1 of the report.

    Parameters
    ----------
    sample : array_like
        The sample to be tested.
    binned : bool, optional
        Whether to carry out a binned fit or not. The default is False.

    Returns
    -------
    mi_h0 : Minuit
        Minuit instance for H0.
    mi_h1 : Minuit
        Minuit instance for H1.
    """

    if binned:
        hist, bin_edges = np.histogram(sample, bins=int(len(sample) ** 0.5), range=xlim)
        nll = BinnedNLL(hist, bin_edges, numba.two_signal_cdf)
    else:
        nll = UnbinnedNLL(sample, numba.two_signal_pdf)

    # H0: f2 = 0
    mi_h0 = Minuit(nll, f1=0.05, f2=0, lam=lam, mu1=5.3, mu2=5.3, sg=0.02)
    mi_h0.limits = [(0, 1), (0, 1), (0, None), (α, β), (α, β), (0.009, 0.06)]
    mi_h0.fixed["f2"] = True
    mi_h0.fixed["mu2"] = True
    mi_h0.simplex()
    mi_h0.migrad()
    mi_h0.hesse()

    # H1: f != 0
    # the means wont be the same in the fit because finding a second peak will always increase the likelihood
    # setting the means to the average of the two peaks forces the fit to find seperate ones
    mi_h1 = Minuit(nll, f1=mi_h0.values['f1'], f2=0.05, lam=mi_h0.values['lam'], mu1=5.2, mu2=5.4, sg=mi_h0.values['sg'])
    mi_h1.limits = [(0, 1), (0, 1), (0, None), (α, β), (α, β), (0.009, 0.06)]
    mi_h1.simplex()
    mi_h1.migrad()
    mi_h1.hesse()

    return mi_h0, mi_h1


# *************************************************************************************************************
# ******************************************** Run Simulation *************************************************
# *************************************************************************************************************


# define sample sizes
sample_sizes = np.linspace(1500, 4000, 8, dtype=int)

# run simulation study
sim = simulation_study(
    sample_sizes=sample_sizes,
    repeats=10_000,
    pdf=numba.two_signal_pdf,
    pdf_params=theta,
    xlim=xlim,
    fit=two_peak_test,
    binned=True,
)

# run for H0 at a single sample size for plotting T0
H0_sim = simulation_study(
    sample_sizes=[1000],
    repeats=10_000,
    pdf=numba.pdf,
    pdf_params=[f1, lam, mu1, sg],
    xlim=xlim,
    fit=two_peak_test,
    binned=True,
)

# discard T < 0 in H0_sim as these represent invalid fits - see report for discussion.
H0_sim = H0_sim[H0_sim > 0]

# save results
with open("results/part_g_results.pkl", "wb") as f:
    pickle.dump([sim, H0_sim], f)

# *************************************************************************************************************
# ********************************* Find DOF for H0 & Calculate P values **************************************
# *************************************************************************************************************


# define chi2 distribution for fitting
def chi2_fit(x, dof):
    return chi2.pdf(x, dof)

# fit the chi2 distribution
nll = UnbinnedNLL(H0_sim, chi2_fit)
mi = Minuit(nll, dof=3)
mi.simplex()
mi.migrad()
mi.hesse()

# extract the dof
dof = mi.values["dof"]

# calculate p-values
pvals = chi2.sf(sim, dof)
avg_pvals = np.nanmedian(pvals, axis=1)
std_pvals = np.nanstd(pvals, axis=1)
q_90_pvals = np.nanquantile(pvals, 0.9, axis=1)
q_10_pvals = np.nanquantile(pvals, 0.1, axis=1)

# *************************************************************************************************************
# ****************************************** Plot results *****************************************************
# *************************************************************************************************************

# plot 1: T distribution under H0
plt.figure(figsize=(8, 6))
plt.hist(H0_sim, bins=70, density=True, label="$P(T|H_0)$", histtype="step")
plt.plot(np.linspace(0, 40, 200), chi2.pdf(np.linspace(0, 40, 200), dof), label=f"χ2 fit, dof = {dof:.2f}", color="k")
plt.axvline(chi2.ppf(1 - 2.9e-7, dof), linestyle="--", label="$T_c$ (χ2 fit at σ = 5)", color="k")
plt.xlabel("T")
plt.ylabel("p(T)")
plt.legend(loc="upper center")
plt.savefig('report/figures/part_g_H0_distribution.png')
plt.show()

# plot 2: T distribution under H1
plt.figure(figsize=(12, 5))
for dist in sim:
    plt.hist(dist, bins=75, density=True, alpha=0.35)
plt.axvline(chi2.ppf(1 - 2.9e-7, dof), linestyle="--", label="$T_c$ (χ2 fit at σ = 5)", color="k", linewidth=1)
plt.xlabel("T")
plt.ylabel("$P(T|H_1)$")
plt.ylim(0, 0.08)
plt.xlim(0, 100)
plt.legend()
plt.savefig('report/figures/part_g_H1_distribution.png')
plt.show()

# plots 3 & 4: p-value and power vs sample size
# create figure
fig = plt.figure(figsize=(13, 5))
grid = gs.GridSpec(1, 2)
ax1 = plt.subplot(grid[0, 0])
ax2 = plt.subplot(grid[0, 1])

# plot 1: median p-value vs Sample Size
ax1.plot(sample_sizes, avg_pvals, "-", label="Median p-value")
ax1.plot(sample_sizes, q_90_pvals, "--", label="90% quantile")
ax1.plot(sample_sizes, q_10_pvals, "--", label="10% quantile")
ax1.set_yscale("log")
ax1.axhline(2.9e-7, color="k", linestyle="--", label="σ = 5")
ax1.set_xlabel("Sample Size")
ax1.set_ylabel("p-value")
ax1.legend()

# plot 2: Power vs Sample Size
y = np.mean(pvals < 2.9e-7, axis=1)
y_e = np.sqrt(
    y * (1 - y) / len(pvals[0])
)  # sample error of a proportion is binomial error
ax2.errorbar(sample_sizes, y, y_e, label="Data", fmt="o", ms=0, capsize=5, capthick=0.5, elinewidth=1)

# fit a sigmoid to y
def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d

popt, pcov = curve_fit(sigmoid, sample_sizes, y, p0=[0.5, 0.01, 1500, 0.5], method="lm", sigma=y_e)
x = np.linspace(sample_sizes[0], sample_sizes[-1], 100)

# plot sigmoid fit
ax2.plot(x, sigmoid(x, *popt), label="Sigmoid fit", linestyle="--", color="green")
ax2.axhline(0.9, color="k", linestyle="--", label="90% power")

# calculate intercept and error and plot on the graph
x90 = np.interp(0.9, sigmoid(x, *popt), x)
x90_e = np.sqrt(np.diag(pcov))[2]  # error on the intercept is the error on the x value at y = 0.9
ax2.axvspan(x90 - x90_e, x90 + x90_e, alpha=0.2, color="green", label="{:.0f} ± {:.0f} samples".format(x90, x90_e))

ax2.legend(framealpha=1)
ax2.set_xlabel("Sample Size")
ax2.set_ylabel("Power")

plt.savefig('report/figures/part_g_power.png')
plt.show()
