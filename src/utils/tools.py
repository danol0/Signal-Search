from scipy.optimize import brute, minimize
from tqdm import tqdm
import numpy as np


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
                    test_statistics.append(T)
                    pbar.update(1)

        sim.append(test_statistics)

    return sim
