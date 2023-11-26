################################################################################################################
# Assuming these true values, generate a single high-statistics sample                                         #
# (containing 100K events) from this probability distribution. Use an estimation                               #
# method to obtain an estimate for the parameters of the model, along with                                     #
# uncertainties on those estimates, using the generated sample. Make a plot which                              #
# shows the generated sample, along with the estimates of the signal, background                               #
# and total probability all overlaid.                                                                          #
################################################################################################################

from utils.distributions import numbaImplementation, analyticalImplementation
from utils.tools import accept_reject
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL

# *************************************************************************************************************
# ******************************************** Generate the sample ********************************************
# *************************************************************************************************************

# set boundaries
α = 5
β = 5.6

# create instances
af = analyticalImplementation(α, β)
nf = numbaImplementation(α, β)

# define parameters
f = 0.1
lam = 0.5
mu = 5.28
sg = 0.018

theta = [f, lam, mu, sg]

# generate sample - note that using the analytical pdf is fastest
sample = accept_reject(f=lambda x: af.pdf(x, *theta), xlim=[α, β], samples=100000)

# *************************************************************************************************************
# *********************************************** Fit the sample **********************************************
# *************************************************************************************************************

# define the negative log likelihood function - numba implimented pdf is fastest
nll = UnbinnedNLL(sample, nf.pdf)

# create Minuit instance - use slightly different starting values to the true values to check convergence
mi = Minuit(nll, f=0.11, lam=0.55, mu=5.25, sg=0.02)

# set error definition
mi_h1.errordef = Minuit.LIKELIHOOD

# add limits: 0 < f < 1, 0 < lam, a < mu < b, 0 < sg
mi.limits = [(0, 1), (0, None), (α, β), (0, None)]

# assign errors
mi.errors = [0.1, 0.1, 0.1, 0.1]

# minimise
mi.migrad()

# extract parameter estimates
est = mi.values

# *************************************************************************************************************
# ****************************************** Plot the sample and fit ******************************************
# *************************************************************************************************************

# create histogram
bins = 80
y, x = np.histogram(sample, bins=bins, range=[α, β])
bin_width = x[1] - x[0]

# shift x to bin centres
x = np.mean(np.vstack([x[1:], x[:-1]]), axis=0)

# normalise y & error
y_n = y / (bin_width * np.sum(y))
y_error = y_n * (1 / y + 1 / (np.sum(y)))**0.5

# plot the sample and fit
plt.errorbar(x, y_n, y_error, label='Data', fmt='o', ms=3, capsize=4, capthick=1, elinewidth=1)
plt.plot(x, nf.pdf(x, *est), label='Fit')
plt.plot(x, est['f'] * nf.signal_pdf(x, est['mu'], est['sg']), label='Signal (scaled)', linestyle='--')
plt.plot(x, (1 - est['f']) * nf.background_pdf(x, est['lam']), label='Background (scaled)', linestyle='--')

# add labels and legend
plt.title('100k sample with fitted distribution')
plt.xlabel('M')
plt.ylabel('Normalised counts/PDF')

# Add parameter estimates to the legend
legend_text = f"Estimated values:\n"\
              f"f = {est['f']:.2f} ± {mi.errors['f']:.1}\n" \
              f"λ = {est['lam']:.2f} ± {mi.errors['lam']:.1}\n" \
              f"μ = {est['mu']:.2f} ± {mi.errors['mu']:.1}\n" \
              f"σ = {est['sg']:.3f} ± {mi.errors['sg']:.1}"
plt.plot([], [], ' ', label=legend_text)

plt.legend()
plt.show()

# save the figure
plt.savefig('results/part_e.png', bbox_inches='tight')