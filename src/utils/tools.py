from scipy.optimize import brute, minimize
from tqdm import tqdm
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy.optimize import curve_fit
from iminuit.cost import BinnedNLL
from iminuit import Minuit
from jacobi import propagate
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

    # define chi2 distribution for fitting
    def chi2_fit(x, dof):
        return chi2.cdf(x, dof)

    y, x = np.histogram(H0_sim, bins=300, density=True)
    nll = BinnedNLL(y, x, chi2_fit)

    mi = Minuit(nll, dof=2)
    mi.limits = [(0, 10)]
    mi.errordef = Minuit.LIKELIHOOD  # set the errordef parameter to 0.5 for NLL to extract correct 1σ errors
    mi.migrad()
    mi.hesse()

    return mi.values["dof"], mi.errors['dof']


def plot_simulation_study(H0_sim, sim, sample_sizes, dof, dof_e, file_name):
    """
    This function plots the results of the simulation study.

    The plots are:
        1. The T distribution under H0.
        2. The T distribution under H1.
        3. The power vs sample size.

    The calculations error propagation is carried out in the top of the function.

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

    # -------- Calculate p-values and power, with propagated errors

    # critical value for 5σ
    T_c = chi2.ppf(1 - 2.9e-7, dof)

    # function for estimating power per sample size for error propagation
    def estimate_power(dof):
        pvals = chi2.sf(sim, dof)
        power = np.mean(pvals < 2.9e-7, axis=1)
        return power

    # estimate power and error
    power, power_var = propagate(estimate_power, dof, dof_e**2, diagonal=True)  # use jacobi to propagate errors
    power_e = np.sqrt(power_var)  # take square root of variance to get standard error

    # add statistical (binomial) error to power error
    power_e += np.sqrt(power * (1 - power) / len(sim[0]))

    # To estimate the sample size for 90% power, we fit a sigmoid to the power vs sample size and
    # interpolate the sample size needed for 90% power. We can again propagate the errors using jacobi

    # define sigmoid function
    def sigmoid(x, a, b, c, d):
        return a / (1 + np.exp(-b * (x - c))) + d

    # define x values for sigmoid fit and plotting
    x = np.linspace(sample_sizes[0], sample_sizes[-1], 300)

    # function for estimating 90% power from sigmoid fit for error propagation
    def estimate_intercept(power):
        popt, _ = curve_fit(sigmoid, sample_sizes, power, p0=[1, 0, np.median(sample_sizes), 0], maxfev=10000)
        return np.interp(0.9, sigmoid(x, *popt), x)

    # estimate 90% and error
    x90, x90_var = propagate(estimate_intercept, power, power_var)
    x90_e = np.sqrt(x90_var)

    # -------- Plot 1: T distribution under H0 and χ2 fit

    bins = 100 if dof > 2 else 50   # reduce bins for small dof to avoid distorting near 0
    plt.figure(figsize=(8, 6))
    plt.hist(H0_sim, bins=bins, density=True, label="$P(T|H_0)$", histtype="step")
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
    plt.savefig("report/figures/" + file_name + "_H0_distribution.png")

    # -------- Plot 2: T distribution under H1

    plt.figure(figsize=(11, 5))
    for i, dist in enumerate(sim):
        plt.hist(dist, bins=90, density=True, alpha=0.35, label=sample_sizes[i])
    plt.axvline(
        T_c,
        linestyle="--",
        label="$T_c$ (χ2 fit at σ = 5)",
        color="k",
        linewidth=1.2,
    )
    plt.xlabel("T")
    plt.ylabel("$P(T|H_1)$")
    plt.ylim(0, 0.08)
    plt.xlim(0, 100)
    plt.legend(title='Sample Size', ncols=2)
    plt.savefig("report/figures/" + file_name + "_H1_distribution.png")

    # -------- Plot 3: Power vs Sample Size

    plt.figure(figsize=(8, 6))

    # plot the power with propagated errors
    plt.errorbar(
        sample_sizes,
        power,
        power_e,
        label="Power with propagated error",
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
    plt.plot(x, sigmoid(x, *sigmoid_fit), label="Sigmoid fit", linestyle="-", color="tab:blue")

    # add line for critical value
    plt.axhline(0.9, color="k", linestyle="--", label="90% power", linewidth=1)

    # add sample size for 90% power and error
    plt.axvline(x90, color="tab:blue", linestyle=":", label="{:.0f} ± {:.0f} samples".format(x90, x90_e))
    plt.axvspan(
        x90 - x90_e,
        x90 + x90_e,
        alpha=0.1,
        color="tab:blue",
    )

    plt.legend(framealpha=1)
    plt.xlabel("Sample Size")
    plt.ylabel("Power")

    plt.savefig("report/figures/" + file_name + "_power.png")
