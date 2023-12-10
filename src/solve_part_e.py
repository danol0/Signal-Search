from utils.tools import accept_reject
import matplotlib.pyplot as plt
import numpy as np
from utils.distributions import numbaImplementation
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
import matplotlib.gridspec as gs
from numba_stats import norm

plt.style.use("src/utils/mphil.mplstyle")

# *************************************************************************************************************
# ******************************************** Generate the sample ********************************************
# *************************************************************************************************************

# set random seed
np.random.seed(1)

# set boundaries
α = 5
β = 5.6

# define parameters
f = 0.1
lam = 0.5
mu = 5.28
sg = 0.018
theta = [f, lam, mu, sg]

# create pdf instance
numba = numbaImplementation(α, β)

# generate the sample
sample = accept_reject(f=lambda x: numba.pdf(x, *theta), xlim=[α, β], samples=100_000)

# *************************************************************************************************************
# *********************************************** Fit the sample **********************************************
# *************************************************************************************************************

# define the negative log likelihood function
nll = UnbinnedNLL(sample, numba.pdf)

# create Minuit instance - use slightly different starting values to the true values to check convergence
mi = Minuit(nll, f=0.5, lam=1, mu=5.3, sg=0.02)

# set error definition
mi.errordef = Minuit.LIKELIHOOD

# add limits
mi.limits = [(0, 1), (0, None), (α, β), (0, None)]

# assign errors
mi.errors = [0.1, 0.1, 0.1, 0.1]

# minimize
mi.migrad()

# extract parameter estimates
est = mi.values

# *************************************************************************************************************
# ****************************************** Plot the sample and fit ******************************************
# *************************************************************************************************************

# to plot the sample as points with error bars, we need to bin the data
bins = 90
y, x = np.histogram(sample, bins=bins, range=[α, β])
bin_width = x[1] - x[0]

# shift x to bin centres
x = np.mean(np.vstack([x[1:], x[:-1]]), axis=0)

# normalise y & error
y_n = y / (bin_width * np.sum(y))
y_error = (y_n / (len(sample) * bin_width)) ** 0.5

# calculate the residuals and pulls
residual = y_n - numba.pdf(x, *est)
pull = residual / y_error

# define the figure with gridspec
fig = plt.figure(figsize=(9, 9))
grid = gs.GridSpec(
    3, 2, hspace=0, wspace=0, height_ratios=(3, 1, 1), width_ratios=(7, 1)
)
ax_main = plt.subplot(grid[0, 0])
ax_residuals = plt.subplot(grid[1, 0], sharex=ax_main)
ax_pulls = plt.subplot(grid[2, 0], sharex=ax_main)
ax_empty = plt.subplot(grid[0, 1])
ax_empty.axis("off")
ax_residuals_dist = plt.subplot(grid[1, 1], sharey=ax_residuals)
ax_pull_dist = plt.subplot(grid[2, 1], sharey=ax_pulls)

# main plot: sample and fit
ax_main.errorbar(
    x,
    y_n,
    y_error,
    label="Sample",
    fmt="o",
    ms=1,
    capsize=4,
    capthick=1,
    elinewidth=1,
    color="k",
)
# total pdf
ax_main.plot(x, numba.pdf(x, *est), label="Fitted distribution", color="tab:blue")
# signal pdf
ax_main.plot(
    x,
    est["f"] * numba.signal_pdf(x, est["mu"], est["sg"]),
    label="Signal (scaled)",
    linestyle=":",
    color="g",
)
# background pdf
ax_main.plot(
    x,
    (1 - est["f"]) * numba.background_pdf(x, est["lam"]),
    label="Background (scaled)",
    linestyle="--",
    color="tab:orange",
    linewidth=1,
)
ax_main.set_xlabel("M")
ax_main.set_ylabel("Density")

# add parameter estimates to the legend
legend_text = (
    f"Estimated values:\n"
    f"f = {est['f']:.2f} ± {mi.errors['f']:.1}\n"
    f"λ = {est['lam']:.2f} ± {mi.errors['lam']:.1}\n"
    f"μ = {est['mu']:.2f} ± {mi.errors['mu']:.1}\n"
    f"σ = {est['sg']:.3f} ± {mi.errors['sg']:.1}"
)
# workaround to add text to legend
ax_main.plot([], [], " ", label=legend_text)
ax_main.legend()

# residual plot
ax_residuals.errorbar(
    x, residual, y_error, fmt="o", ms=2, capsize=4, capthick=1, elinewidth=1, color="k"
)
ax_residuals.axhline(0, color="tab:blue", linewidth=1)
ax_residuals.set_ylabel("Residuals")
# ax_residuals.set_ylim(-0.19, 0.19)

# residual distribution plot
ax_residuals_dist.hist(
    residual,
    bins=10,
    density=True,
    alpha=0.5,
    orientation="horizontal",
)
ax_residuals_dist.xaxis.set_visible(False)
ax_residuals_dist.spines[["top", "bottom", "right"]].set_visible(False)
ax_residuals_dist.tick_params(
    which="both", direction="in", axis="y", right=False, labelcolor="none"
)

# pull plot
ax_pulls.errorbar(
    x,
    pull,
    np.ones_like(x),
    fmt="o",
    ms=2,
    capsize=4,
    capthick=1,
    elinewidth=1,
    color="k",
)
ax_pulls.axhline(0, color="tab:blue", linewidth=1)
ax_pulls.set_xlabel("$M$")
ax_pulls.set_ylabel("Pulls")

# pull distribution plot
ax_pull_dist.hist(
    pull, bins=10, density=True, alpha=0.5, orientation="horizontal"
)
ax_pull_dist.xaxis.set_visible(False)
ax_pull_dist.spines[["top", "bottom", "right"]].set_visible(False)
ax_pull_dist.tick_params(
    which="both", direction="in", axis="y", right=False, labelcolor="none"
)
xp = np.linspace(-3.5, 3.5, 100)
ax_pull_dist.plot(norm.pdf(xp, loc=0, scale=1), xp, "r-", alpha=0.5)

fig.align_ylabels()
plt.savefig("report/figures/part_e_result.png")
