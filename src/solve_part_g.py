from utils.distributions import numbaImplementation, analyticalImplementation
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL, BinnedNLL
from tqdm import tqdm
from scipy.stats import chi2
import matplotlib.gridspec as gs
import pickle

plt.style.use('src/utils/mphil.mplstyle')

# set random seed
np.random.seed(0)

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

theta = [f1, f2, lam, mu1, mu2, sg]


# *************************************************************************************************************
# ************************************ Set Up Simulation Study ************************************************
# *************************************************************************************************************

# we use the same method as before, with modified h0 and h1

# inverse cdf method to generate two peak samples
def two_signal_inverse_cdf_generator(f1, f2, samples):
    u = np.random.random(samples)
    n = np.random.random(samples)
    events = np.empty(samples)
    events = [nf.signal_inv_cdf(u[i], mu1, sg) if n[i] < f1
              else nf.signal_inv_cdf(u[i], mu2, sg) if n[i] < f1 + f2
              else nf.background_inv_cdf(u[i], lam) for i in range(samples)]
    return events


# fit function: fits h0 and h1 to the sample and returns the Minuit instances
def minuit_fit(sample, binned=False):

    if binned:
        hist, bin_edges = np.histogram(sample, bins=int(len(sample)**0.5), range=(α, β))
        nll = BinnedNLL(hist, bin_edges, nf.two_signal_cdf)
    else:
        nll = UnbinnedNLL(sample, nf.two_signal_pdf)

    # H0: f1 = 0 or f2 = 0 - set the mean to the average of the two peaks
    mi_h0 = Minuit(nll, f1=f1, f2=0, lam=lam, mu1=5.32, mu2=5.32, sg=sg)
    mi_h0.limits = [(0.001, 1), (0.001, 1), (0.01, None), (α, β), (α, β), (0.0001, None)]
    mi_h0.fixed["f2"] = True
    mi_h0.fixed["mu2"] = True
    mi_h0.errordef = Minuit.LIKELIHOOD
    mi_h0.migrad()

    # H1: f1 != 0 and f2 != 0
    # the means wont be the same in the fit because finding a second peak will always increase the likelihood
    # setting the means to the average of the two peaks forces the fit to find seperate ones
    mi_h1 = Minuit(nll, f1=f1, f2=f2, lam=lam, mu1=5.32, mu2=5.32, sg=sg)
    mi_h1.limits = [(0.001, 1), (0.001, 1), (0.01, None), (α, β), (α, β), (0.0001, None)]
    mi_h1.errordef = Minuit.LIKELIHOOD
    mi_h1.migrad()

    return mi_h0, mi_h1


# function to run simulation study
def simulation_study(sample_sizes, repeats, H0=False, binned=False):
    output = []

    # for each sample size
    for s in sample_sizes:
        delta_Ls = []

        # run r repeats
        with tqdm(total=repeats, desc=f"Sample Size: {s}") as pbar:
            for _ in range(repeats):
                # generate a sample
                if H0:
                    sample = two_signal_inverse_cdf_generator(0.1, 0, s)
                else:
                    sample = two_signal_inverse_cdf_generator(0.1, 0.05, s)

                # fit the sample
                mi_h0, mi_h1 = minuit_fit(sample, binned=binned)

                # if fit is valid, append the log likelihood difference
                if mi_h0.valid and mi_h1.valid:
                    T = mi_h0.fval - mi_h1.fval
                    delta_Ls.append(T)
                    pbar.update(1)

                # otherwise, append nan
                else:
                    pbar.update(1)
                    delta_Ls.append(np.nan)

        output.append(delta_Ls)

    return output


# *************************************************************************************************************
# ******************************************** Run Simulation *************************************************
# *************************************************************************************************************

# set binned or unbinned
binned = True

# define sample sizes
sample_sizes = np.linspace(1000, 3500, 6, dtype=int)

# run simulation study
sim = simulation_study(sample_sizes, repeats=50_000, binned=binned)

# run for H0 at a single sample size for plotting T0
H0_sim = simulation_study([1000], repeats=50_000, H0=True, binned=binned)

# save results
if binned:
    with open("results/part_g_results_binned.pkl", "wb") as f: pickle.dump([sim, H0_sim], f)
else:
    with open("results/part_g_results.pkl", "wb") as f: pickle.dump([sim, H0_sim], f)

# load results
#with open('results/part_f_results.pkl', 'rb') as f: sim, H0_sim = pickle.load(f)


# *************************************************************************************************************
# ****************************************** Plot results *****************************************************
# *************************************************************************************************************

# calculate p-values
pvals = chi2.sf(sim, 1)
avg_pvals = np.nanmedian(pvals, axis=1)
std_pvals = np.nanstd(pvals, axis=1)
q_90_pvals = np.nanquantile(pvals, 0.9, axis=1)
q_10_pvals = np.nanquantile(pvals, 0.1, axis=1)

# create figure
fig = plt.figure(figsize=(14, 8))
grid = gs.GridSpec(2, 3)

# create axis objects for each plot
ax1 = plt.subplot(grid[0, 0])
ax2 = plt.subplot(grid[1, 0])
ax3 = plt.subplot(grid[0, 1:3])
ax4 = plt.subplot(grid[1, 1:3])

# plot 1: median p-value vs Sample Size
ax1.plot(sample_sizes, avg_pvals, "-", label="Median p-value")
ax1.plot(sample_sizes, q_90_pvals, "--", label="90% quantile")
ax1.plot(sample_sizes, q_10_pvals, "--", label="10% quantile")
ax1.set_yscale("log")
ax1.axhline(2.9e-7, color="k", linestyle="--", label="5 Sigma")
ax1.set_xlabel("Sample Size")
ax1.set_ylabel("p-value")
ax1.set_title("Median p-value vs Sample Size")
ax1.legend()

# plot 2: Fraction with p-value < 2.9e-7 vs Sample Size
y = np.mean(pvals < 2.9e-7, axis=1)
y_e = np.sqrt(y * (1 - y) / len(pvals[0]))
ax2.errorbar(sample_sizes, y, y_e, label="Data", fmt="o", ms=3, capsize=3, capthick=1, elinewidth=1, linestyle="-")
ax2.set_xlabel("Sample Size")
ax2.set_title("Fraction of p-values < 2.9e-7")
ax2.set_ylabel("Fraction")
ax2.axhline(0.9, color="k", linestyle="--")

# plot 3: T distribution under H1
for dist in sim:
    ax3.hist(dist, bins=80, density=True, alpha=0.4)
ax3.axvline(chi2.ppf(1 - 2.9e-7, 1), color="k", linestyle="--", label="T0: 5 sigma (analytical)")
ax3.set_xlabel("T")
ax3.set_ylabel("p(T)")
ax3.set_ylim(0, 0.08)
ax3.set_title("Test statistic distribution under H1 for sample sizes 1000 - 3500")
ax3.set_xlim(0, 120)
ax3.legend()

# plot 4: T distribution under H0
ax4.hist(H0_sim, bins=100, density=True, label="H0")
ax4.axvline(np.nanquantile(H0_sim, 1 - 2.9e-7), linestyle="--", label="T0 (simulated)")
ax4.hist(sim[1], bins=80, density=True, label="s=1500", histtype="step")
ax4.axvline(np.nanquantile(sim[1], 0.1), linestyle="--", label="10% quantile for s=1500", color="orange")
ax4.hist(sim[3], bins=80, density=True, label="s=2500", alpha=0.5)
ax4.axvline(np.nanquantile(sim[3], 0.1), linestyle="--", label="10% quantile for s=2500", color="limegreen")
ax4.set_ylim(0, 0.08)
ax4.set_xlim(0, 100)
ax4.annotate(
    "T0 (analytical)",
    xy=(chi2.ppf(1 - 2.9e-7, 1), 0),
    xytext=(chi2.ppf(1 - 2.9e-7, 1), -0.01),
    arrowprops=dict(width=0.1, headwidth=0, facecolor="black", shrink=0.0001),
)
ax4.set_title("Test statistic distrbutions under H0 and H1 with relevant quantiles")
ax4.set_xlabel("T")
ax4.set_ylabel("p(T)")
ax4.legend()

plt.tight_layout()

# save figure
plt.savefig("results/part_g_results_binned.png") if binned else plt.savefig("results/part_g_results.png")

plt.show()
