from utils.tools import accept_reject
import matplotlib.pyplot as plt
import numpy as np
from utils.distributions import numbaImplementation
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
import matplotlib.gridspec as gs
from numba_stats import norm
plt.style.use("src/utils/signal.mplstyle")

# *************************************************************************************************************
# ******************************************* Initialise Parameters *******************************************
# *************************************************************************************************************

# set random seed
np.random.seed(1)

# set boundaries
α = 5
β = 5.6

# define parameters
f1 = 0.1
f2 = 0.05
lam = 0.5
mu1 = 5.28
mu2 = 5.35
sg = 0.018

theta = [f1, lam, mu1, sg]
theta2 = [f1, f2, lam, mu1, mu2, sg]

# create pdf instance
numba = numbaImplementation(α, β)

# generate the samples
single_sample = accept_reject(f=lambda x: numba.pdf(x, *theta), xlim=[α, β], samples=100_000)
double_sample = accept_reject(f=lambda x: numba.two_signal_pdf(x, *theta2), xlim=[α, β], samples=100_000)

# *************************************************************************************************************
# *********************************************** Fit the sample **********************************************
# *************************************************************************************************************

for sample,pdf,name in zip([single_sample,double_sample],[numba.pdf, numba.two_signal_pdf],["single","double"]):

    # define the negative log likelihood function
    nll = UnbinnedNLL(sample, pdf)

    # create Minuit instance - use slightly different starting values to the true values to check convergence
    if name == "single":
        mi = Minuit(nll, f=0.5, lam=1, mu=5.3, sg=0.02)
        mi.limits = [(0, 1), (0, None), (α, β), (0, None)]
    else:
        mi = Minuit(nll, f1=0.5, f2=0.5, lam=1, mu1=5.3, mu2=5.3, sg=0.02)
        mi.limits = [(0, 1), (0, 1), (0, None), (α, β), (α, β), (0, None)]

    # minimize
    mi.errordef = Minuit.LIKELIHOOD
    mi.migrad()

    # extract parameter estimates
    est = mi.values

    # *************************************************************************************************************
    # ****************************************** Plot the sample and fit ******************************************
    # *************************************************************************************************************

    # define some colours
    main_colour = '#2A788EFF'
    signal_colour = 'tab:green'
    background_colour = 'tab:orange'

    # to plot the sample as points with error bars, we need to bin the data
    bins = 90
    y, x = np.histogram(sample, bins=bins, range=[α, β])
    bin_width = x[1] - x[0]

    # shift x to bin centres
    x = np.mean(np.vstack([x[1:], x[:-1]]), axis=0)

    # normalise y & error
    scale_factor = 1 / np.sum(y) / bin_width
    y_n = y * scale_factor
    y_error = scale_factor * y ** 0.5

    # calculate the residuals and pulls
    residual = y_n - pdf(x, *est)
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
    ax_main.plot(x, pdf(x, *est), label="Fitted distribution", color=main_colour)
    # signal pdf
    ax_main.plot(
        x,
        est["f"] * numba.signal_pdf(x, est["mu"], est["sg"]),
        label="Signal (scaled)",
        linestyle=":",
        color=signal_colour,
    )
    # background pdf
    if name == "single":
        ax_main.plot(
            x,
            (1 - est["f"]) * numba.background_pdf(x, est["lam"]),
            label="Background (scaled)",
            linestyle="--",
            color=background_colour,
        )
    else:
        
    ax_main.plot(
        x,
        (1 - est["f"]) * numba.background_pdf(x, est["lam"]),
        label="Background (scaled)",
        linestyle="--",
        color=background_colour,
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
    ax_residuals.axhline(0, color=main_colour, linewidth=1)
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
    ax_pulls.axhline(0, color=main_colour, linewidth=1)
    ax_pulls.set_xlabel("$M$")
    ax_pulls.set_ylabel("Pulls")

    # pull distribution plot
    ax_pull_dist.hist(
        pull, bins=10, density=True, alpha=0.5, orientation="horizontal",
    )
    ax_pull_dist.xaxis.set_visible(False)
    ax_pull_dist.spines[["top", "bottom", "right"]].set_visible(False)
    ax_pull_dist.tick_params(
        which="both", direction="in", axis="y", right=False, labelcolor="none"
    )
    xp = np.linspace(-3.5, 3.5, 100)
    ax_pull_dist.plot(norm.pdf(xp, loc=0, scale=1), xp, "r-", alpha=0.5)

    fig.align_ylabels()
    plt.savefig("figures/" + name + "_fit.png", bbox_inches="tight")