from scipy.optimize import brute, minimize
from tqdm import tqdm
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy.optimize import curve_fit
from iminuit.cost import BinnedNLL
from iminuit import Minuit
plt.style.use("src/utils/mphil.mplstyle")


def accept_reject(f, xlim, samples):
    """
    This function generates samples from a distribution using an optimised accept-reject method.

    Parameters
    ----------
    f : function
        The distribution to sample from.
    xlim : array_like
        The boundaries of the distribution.
    samples : int
        The number of samples to generate.

    Returns
    -------
    events : array_like
        The generated samples.
    """

    # find maximum of f
    argmax = brute(lambda x: -f(x), [xlim], disp=False)[0]
    argmax = minimize(lambda x: -f(x), argmax, bounds=[xlim]).x[0]
    fmax = f(argmax)

    events = np.empty(0)

    while len(events) < samples:
        # generate x and y
        x = np.random.uniform(*xlim, samples)
        y = np.random.uniform(0, fmax, samples)

        # accept-reject
        events = np.append(events, x[y < f(x)])

        # check for an fmax violation and reset the sample if necessary
        if np.any(f(x) > fmax):
            fmax = np.max(f(x))
            events = np.empty(0)

    return events[:samples]


def simulation_study(sample_sizes, repeats, pdf, pdf_params, xlim, fit, binned=False):
    """
    This function carries out a simulation study for a given hypothesis test, list of sample sizes
    and number of repeats, and returns the calculated test statistics.

    Parameters
    ----------
    sample_sizes : array_like
        The sample sizes to use.
    repeats : int
        The number of repeats to carry out.
    pdf : function
        The distribution to sample from.
    pdf_params : array_like
        The parameters to use in the distribution.
    xlim : array_like
        The boundaries of the distribution.
    fit : function
        The hypothesis test to fit the sample.
    binned : bool, optional
        Whether to carry out a binned fit or not. The default is False.

    Returns
    -------
    output : array_like
        A repeats x len(sample_sizes) array containing the calculated test statistics.
    """

    sim = []

    # for each sample size
    for s in sample_sizes:
        test_statistics = []

        # run r repeats
        with tqdm(total=repeats, desc=f"Sample Size: {s}") as pbar:
            while len(test_statistics) < repeats:
                # generate a sample
                sample = accept_reject(
                    f=lambda x: pdf(x, *pdf_params), xlim=xlim, samples=s
                )

                # fit the sample
                mi_h0, mi_h1 = fit(sample, binned=binned)

                # if fit is valid, calculate the test statistic
                if mi_h0.valid and mi_h1.valid:
                    # Note that factor of 2 is included in the iminuit definition of fval
                    T = mi_h0.fval - mi_h1.fval
                    # Assign negative test statistics to 0: source https://arxiv.org/abs/1007.1727 section 2.2
                    if T < 0:
                        T = 0
                    test_statistics.append(T)
                    pbar.update(1)

        sim.append(test_statistics)

    return sim


def fit_chi2_dof(H0_sim):
    """
    This function fits the chi2 distribution to the test statistics under H0 and returns the
    degrees of freedom.
    
    Parameters
    ----------
    H0_sim : array_like
        The test statistics under H0.
    """

    # define chi2 distribution for fitting
    def chi2_fit(x, dof):
        return chi2.cdf(x, dof)

    y, x = np.histogram(H0_sim, bins=100, density=True)
    nll = BinnedNLL(y, x, chi2_fit)

    mi = Minuit(nll, dof=2)
    mi.limits = [(0, 10)]
    mi.migrad()

    return mi.values["dof"]


def plot_simulation_study(H0_sim, sim, sample_sizes, dof, file_name):
    """
    This function plots the results of the simulation study and saves the plots.

    Parameters
    ----------
    H0_sim : array_like
        The test statistics under H0.
    sim : array_like
        The test statistics under H1.
    sample_sizes : array_like
        The sample sizes used.
    dof : float
        The degrees of freedom.
    file_name : str
        The file name to save the plots as.
    """

    # calculate p-values
    pvals = chi2.sf(sim, dof)
    avg_pvals = np.nanmedian(pvals, axis=1)
    q_90_pvals = np.nanquantile(pvals, 0.9, axis=1)
    q_10_pvals = np.nanquantile(pvals, 0.1, axis=1)

    # critical value for 5σ
    T_c = chi2.ppf(1 - 2.9e-7, dof)

    # plot 1: T distribution under H0
    bins = 100 if dof > 2 else 50   # reduce bins for small dof to avoid distorting near 0
    plt.figure(figsize=(8, 6))
    plt.hist(H0_sim, bins=bins, density=True, label="$P(T|H_0)$", histtype="step")
    plt.plot(
        np.linspace(0, int(T_c*1.1), 200),
        chi2.pdf(np.linspace(0, int(T_c*1.1), 200), dof),
        label=f"χ2 fit, dof = {dof:.2f}",
        color="k",
    )
    plt.axvline(
        T_c,
        linestyle="--",
        label="$T_c$ (χ2 fit at σ = 5)",
        color="k",
        linewidth=1,
    )
    plt.xlabel("T")
    plt.ylabel("p(T)")
    plt.legend(loc="upper center")
    plt.savefig("report/figures/"+ file_name + "_H0_distribution.png")

    # plot 2: T distribution under H1
    plt.figure(figsize=(12, 5))
    for dist in sim:
        plt.hist(dist, bins=90, density=True, alpha=0.35)
    plt.axvline(
        T_c,
        linestyle="--",
        label="$T_c$ (χ2 fit at σ = 5)",
        color="k",
        linewidth=1,
    )
    plt.xlabel("T")
    plt.ylabel("$P(T|H_1)$")
    plt.ylim(0, 0.08)
    plt.xlim(0, 100)
    plt.legend()
    plt.savefig("report/figures/"+ file_name + "_H1_distribution.png")

    # plots 3 & 4: p-value and power vs sample size
    # create figure
    plt.figure(figsize=(13, 5))
    grid = gs.GridSpec(1, 2)
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[0, 1])

    # plot 1: median p-value vs Sample Size
    ax1.plot(sample_sizes, avg_pvals, "-", label="Median p-value")
    ax1.plot(sample_sizes, q_90_pvals, "--", label="90% quantile")
    ax1.plot(sample_sizes, q_10_pvals, "--", label="10% quantile")
    ax1.set_yscale("log")
    ax1.axhline(2.9e-7, color="k", linestyle="--", label="σ = 5", linewidth=1)
    ax1.set_xlabel("Sample Size")
    ax1.set_ylabel("p-value")
    ax1.legend()

    # plot 2: Power vs Sample Size
    y = np.mean(pvals < 2.9e-7, axis=1)
    y_e = np.sqrt(
        y * (1 - y) / len(pvals[0])
    )  # sample error of a proportion is binomial error - ref: https://pdp.sjsu.edu/faculty/gerstman/StatPrimer/conf-prop.htm#:~:text=The%20standard%20error%20of%20a,symbol%20is%20called%20a%20hat.
    ax2.errorbar(
        sample_sizes,
        y,
        y_e,
        label="Data",
        fmt="o",
        ms=3,
        capsize=5,
        capthick=1,
        elinewidth=1,
    )

    # fit a sigmoid to y
    def sigmoid(x, a, b, c, d):
        return a / (1 + np.exp(-b * (x - c))) + d

    popt, pcov = curve_fit(
        sigmoid,
        sample_sizes,
        y,
        p0=[0.5, 0.01, np.median(sample_sizes), 0.5],
        method="lm",
        sigma=y_e,
    )
    x = np.linspace(sample_sizes[0], sample_sizes[-1], 100)

    # plot sigmoid fit
    ax2.plot(x, sigmoid(x, *popt), label="Sigmoid fit", linestyle="--", color="green")
    ax2.axhline(0.9, color="k", linestyle="--", label="90% power", linewidth=1)

    # calculate intercept and error and plot on the graph
    x90 = np.interp(0.9, sigmoid(x, *popt), x)
    x90_e = np.sqrt(np.diag(pcov))[
        2
    ]  # error on the intercept is the error on the x value at y = 0.9
    ax2.axvspan(
        x90 - x90_e,
        x90 + x90_e,
        alpha=0.2,
        color="green",
        label="{:.0f} ± {:.0f} samples".format(x90, x90_e),
    )

    ax2.legend(framealpha=1)
    ax2.set_xlabel("Sample Size")
    ax2.set_ylabel("Power")

    plt.savefig("report/figures/"+ file_name + "_power.png")
