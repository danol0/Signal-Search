from utils.distributions import numbaImplementation
from utils.tools import simulation_study, plot_simulation_study, fit_chi2_dof
from iminuit.cost import UnbinnedNLL, BinnedNLL
from iminuit import Minuit
import matplotlib.pyplot as plt
import numpy as np
import pickle
plt.style.use("src/utils/signal.mplstyle")

# Set to True to run the simulation study, or False to load from file
run = False

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
f = 0.1
lam = 0.5
mu = 5.28
sg = 0.018
theta = [f, lam, mu, sg]

# *************************************************************************************************************
# ************************************* Define Hypothesis Test ************************************************
# *************************************************************************************************************


def single_peak_test(sample, binned=False):
    """
    This function carries out the hypothesis test for a single peak and returns
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
        nll = BinnedNLL(hist, bin_edges, numba.cdf)
    else:
        nll = UnbinnedNLL(sample, numba.pdf)

    # H0: f = 0
    mi_h0 = Minuit(nll, f=0, lam=0.8, mu=5.3, sg=0.02)
    mi_h0.limits = [(0, 1), (0, None), (α, β), (0.009, 0.04)]
    mi_h0.fixed["f"] = True
    mi_h0.fixed["mu"] = True
    mi_h0.fixed["sg"] = True
    mi_h0.migrad()
    mi_h0.hesse()

    # H1: f != 0
    mi_h1 = Minuit(nll, *mi_h0.values)
    mi_h1.limits = [(0, 1), (0, None), (α, β), (0.009, 0.04)]
    mi_h1.simplex()
    mi_h1.migrad()
    mi_h1.hesse()

    return mi_h0, mi_h1


# *************************************************************************************************************
# ******************************************** Run Simulation *************************************************
# *************************************************************************************************************

# define sample sizes
sample_sizes = np.linspace(100, 1000, 10, dtype=int)

if run:
    print("Running simulation study...")
    # run simulation study
    # sim is a list of lists, each list contains the test statistics for a given sample size
    sim = simulation_study(
        sample_sizes=sample_sizes,
        repeats=100_000,
        pdf=numba.pdf,
        pdf_params=theta,
        xlim=xlim,
        fit=single_peak_test,
        binned=False,
    )

    # run for H0 at a single sample size for plotting T0
    # H0_sim is a list of test statistics for H0
    H0_sim = simulation_study(
        sample_sizes=[500],
        repeats=100_000,
        pdf=numba.background_pdf,
        pdf_params=[lam],
        xlim=xlim,
        fit=single_peak_test,
        binned=False,
    )

    # save results
    with open("results/part_f_results.pkl", "wb") as f:
        pickle.dump([sim, H0_sim], f)

else:
    print("File running from saved results...")
    # load results
    with open("results/part_f_results.pkl", "rb") as f:
        sim, H0_sim = pickle.load(f)

# *************************************************************************************************************
# ******************************** Find DOF for H0 and Plot Results *******************************************
# *************************************************************************************************************

dof, dof_e = fit_chi2_dof(H0_sim)

plot_simulation_study(
    H0_sim=H0_sim, sim=sim, sample_sizes=sample_sizes, dof=dof, dof_e=dof_e, file_name="single_peak"
)
