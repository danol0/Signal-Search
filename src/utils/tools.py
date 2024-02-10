from scipy.optimize import brute, minimize
from tqdm import tqdm
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from iminuit.cost import BinnedNLL
from iminuit import Minuit
from jacobi import propagate
plt.style.use("src/utils/signal.mplstyle")


def accept_reject(f, xlim, samples):
    """
    This function generates samples from a distribution using an accept-reject method.

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

    # find maximum of f for upper bound
    argmax = brute(lambda x: -f(x), [xlim], disp=False)[0]
    argmax = minimize(lambda x: -f(x), argmax, bounds=[xlim]).x[0]
    fmax = f(argmax)

    events = np.empty(0)

    while len(events) < samples:

        x = np.random.uniform(*xlim, samples)
        y = np.random.uniform(0, fmax, samples)

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
    sim : array_like
        A repeats x len(sample_sizes) array containing the calculated test statistics.
    """

    sim = []

    for s in sample_sizes:
        test_statistics = []

        with tqdm(total=repeats, desc=f"Sample Size: {s}") as pbar:
            while len(test_statistics) < repeats:

                sample = accept_reject(
                    f=lambda x: pdf(x, *pdf_params), xlim=xlim, samples=s
                )

                mi_h0, mi_h1 = fit(sample, binned=binned)


                if mi_h0.valid and mi_h1.valid:
                    # Note that factor of 2 is included in the iminuit definition of fval
                    T = mi_h0.fval - mi_h1.fval
                    # Assign negative test statistics to 0, ref https://arxiv.org/abs/1007.1727 section 2.2
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

    def chi2_fit(x, dof):
        return chi2.cdf(x, dof)

    # workaround for calculating bins: pickle loads as nx1 array, so we need to check the shape
    bins = max(int(len(H0_sim)**0.5), int(len(H0_sim[0])**0.5))

    y, x = np.histogram(H0_sim, bins=bins, density=True)
    nll = BinnedNLL(y, x, chi2_fit)

    mi = Minuit(nll, dof=2)
    mi.limits = [(0, 10)]
    mi.errordef = Minuit.LIKELIHOOD  # set the errordef parameter to 0.5 for NLL to extract correct 1σ errors
    mi.migrad()
    mi.hesse()

    if not mi.valid:
        print("Warning: Invalid Chi2 Fit")

    return mi.values["dof"], mi.errors['dof']


def sigmoid(x, a, b, c, d):
    """Define a sigmoid for plots & fits."""
    return a / (1 + np.exp(-b * (x - c))) + d


def analyse_power(sim, sample_sizes, dof, dof_e):
    """
    This function analyses the power of the hypothesis test and returns the power, power error,
    sample size for 90% power, sample size error, and the critical value at 5σ.

    Parameters
    ----------
    sim : array_like
        The test statistics under H1.
    sample_sizes : array_like
        The sample sizes used.
    dof : float
        The degrees of freedom.
    dof_e : float
        The error on the degrees of freedom.

    Returns
    -------
    power : array_like
        The power for each sample size.
    power_e : array_like
        The error on the power for each sample size.
    x90 : float
        The sample size for 90% power.
    x90_e : float
        The error on the sample size for 90% power.
    T_c : float
        The critical value at 5σ.
    """

    # define critical value at 5σ for plotting
    T_c = chi2.ppf(1 - 2.9e-7, dof)

    # function for estimating power per sample size for error propagation
    # defining the critical value again in this function allows jacobi to propagate the error
    def estimate_power(dof):
        critical_value = chi2.ppf(1 - 2.9e-7, dof)
        power = np.mean(sim > critical_value, axis=1)
        return power

    # estimate power and error
    power, power_var = propagate(estimate_power, dof, dof_e**2, diagonal=True)  # use jacobi to propagate errors
    power_e = np.sqrt(power_var)  # take square root of variance to get standard error

    # add statistical (binomial) error to power error in quadrature
    power_e = np.sqrt(power_e ** 2 + power * (1 - power) / len(sim[0]))

    # To estimate the sample size for 90% power, fit a sigmoid to the power vs sample size and interpolate
    # the sample size needed for 90% power. Can again propagate the error on this with jacobi

    # define x values for sigmoid fit
    x = np.linspace(sample_sizes[0], sample_sizes[-1], 300)

    # function for estimating 90% power from sigmoid fit for error propagation
    def estimate_intercept(power):
        popt, _ = curve_fit(sigmoid, sample_sizes, power, p0=[1, 0, np.median(sample_sizes), 0], maxfev=10000)
        return np.interp(0.9, sigmoid(x, *popt), x)

    x90, x90_var = propagate(estimate_intercept, power, power_var)
    x90_e = np.sqrt(x90_var)

    return power, power_e, x90, x90_e, T_c


def plot_simulation_study(H0_sim, sim, sample_sizes, dof, dof_e, file_name):
    """
    This function plots the results of the simulation study.

    The plots are:
        1. The T distribution under H0.
        2. The T distribution under H1.
        3. The power vs sample size.

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

    x = np.linspace(sample_sizes[0], sample_sizes[-1], 300)

    power, power_e, x90, x90_e, T_c = analyse_power(H0_sim, sim, sample_sizes, dof, dof_e)

    # -------- Plot 1: T distribution under H0 and χ2 fit

    blue = '#2A788EFF'

    bins = 100 if dof > 2 else 50   # reduce bins for small dof to avoid distorting near 0
    plt.figure(figsize=(8, 6))
    plt.hist(H0_sim, bins=bins, density=True, label="$P(T|H_0)$", histtype="step", color=blue)
    # show T samples on x axis
    plt.scatter(H0_sim, np.zeros_like(H0_sim), alpha=0.5, marker=2, color=blue)
    # add a single T sample for legend
    plt.scatter(0, 0, alpha=0.5, marker=2, color=blue, label="T samples")
    plt.plot(
        np.linspace(0, int(T_c * 1.1), 200),
        chi2.pdf(np.linspace(0, int(T_c * 1.1), 200), dof),
        label=f"χ2 fit, dof = {dof:.2f} ± {dof_e:.2f}",
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
    plt.savefig("figures/" + file_name + "_H0_distribution.png", bbox_inches="tight")

    # -------- Plot 2: T distribution under H1

    plt.figure(figsize=(11, 5))
    # plot the T distribution for each sample size
    plot = []  # list to store the plots for the separate legends
    for i, dist in enumerate(sim):
        p = plt.hist(dist, bins=100, density=True, alpha=0.35, label=sample_sizes[i])
        plot.append(p)

    # plot the critical value
    Tc = plt.axvline(
        T_c,
        linestyle="--",
        label="$T_c$ (χ2 fit at σ = 5)",
        color="k",
        linewidth=1.2,
    )

    # add H0 distribution for comparison
    H0 = plt.hist(H0_sim, bins=100, density=True, label="$P(T|H_0)$", histtype="step", color="k")

    legend1 = plt.legend(handles=[H0[2][0], Tc], loc="center right")
    plt.gca().add_artist(legend1)
    plt.xlabel("T")
    plt.ylabel("$P(T)$")
    plt.ylim(0, 0.08)
    plt.xlim(0, 100)
    plt.legend(handles=[p[2][0] for p in plot], title='$P(T|H_1)$ Sample Size', ncols=2)
    plt.savefig("figures/" + file_name + "_H1_distribution.png", bbox_inches="tight")

    # -------- Plot 3: Power vs Sample Size

    plt.figure(figsize=(8, 6))

    # plot the power with propagated errors
    plt.errorbar(
        sample_sizes,
        power,
        power_e,
        label="Power",
        fmt="o",
        ms=3,
        capsize=3,
        capthick=1.2,
        elinewidth=1,
        color="k",
    )

    # fit sigmoid for plotting
    sigmoid_fit, _ = curve_fit(
        sigmoid,
        sample_sizes,
        power,
        p0=[1, 0, np.median(sample_sizes), 0],
    )

    # plot sigmoid fit
    plt.plot(x, sigmoid(x, *sigmoid_fit), label="Sigmoid fit", linestyle="-", color="#2A788EFF")

    # add line for 90% power
    plt.axhline(0.9, color="k", linestyle="--", label="0.9 Power", linewidth=1)

    # add sample size for 90% power and error
    plt.axvline(x90, color="#2A788EFF", linestyle=":", label="{:.0f} ± {:.0f} samples".format(x90, x90_e))
    plt.axvspan(
        x90 - x90_e,
        x90 + x90_e,
        alpha=0.1,
        color="#2A788EFF",
    )

    plt.legend(framealpha=1)
    plt.xlabel("Sample Size")
    plt.ylabel("Power")

    plt.savefig("figures/" + file_name + "_power.png", bbox_inches="tight")
