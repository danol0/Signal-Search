from utils.distributions import numbaImplementation
from utils.tools import simulation_study, plot_simulation_study, fit_chi2_dof
from iminuit.cost import UnbinnedNLL, BinnedNLL
from iminuit import Minuit
import numpy as np
from scipy.stats import chi2
import pickle

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


def two_peak_test(sample, binned=False):
    """
    This function carries out the hypothesis test for two peaks and returns
    the Minuit instances for H0 and H1.

    Methodology is discussed in section 3.2 of the report.

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


# *************************************************************************************************************
# ******************************************** Run Simulation *************************************************
# *************************************************************************************************************

# set to True to run the simulation study, False to load results from file
run = False

# define sample sizes
sample_sizes = np.linspace(1500, 4000, 8, dtype=int)

if run:
    # run simulation study
    sim = simulation_study(
        sample_sizes=sample_sizes,
        repeats=30_000,
        pdf=numba.two_signal_pdf,
        pdf_params=theta,
        xlim=xlim,
        fit=two_peak_test,
        binned=True,
    )

    # run for H0 at a single sample size for plotting T0
    H0_sim = simulation_study(
        sample_sizes=[2000],
        repeats=30_000,
        pdf=numba.pdf,
        pdf_params=[f1, lam, mu1, sg],
        xlim=xlim,
        fit=two_peak_test,
        binned=True,
    )

    # save results
    with open("results/part_g_results.pkl", "wb") as f:
        pickle.dump([sim, H0_sim], f)

else:
    # load results
    with open("results/part_g_results.pkl", "rb") as f:
        sim, H0_sim = pickle.load(f)

# *************************************************************************************************************
# ******************************** Find DOF for H0 and Plot Results *******************************************
# *************************************************************************************************************

dof = fit_chi2_dof(H0_sim)

plot_simulation_study(
    H0_sim=H0_sim, sim=sim, sample_sizes=sample_sizes, dof=dof, file_name="part_g"
)
