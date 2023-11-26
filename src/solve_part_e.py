from utils.distributions import numbaImplementation, analyticalImplementation
from utils.tools import accept_reject
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
plt.style.use('src/utils/mphil.mplstyle')

# *************************************************************************************************************
# ******************************************** Generate the sample ********************************************
# *************************************************************************************************************

# set random seed
np.random.seed(0)

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
mi.errordef = Minuit.LIKELIHOOD

# add limits
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
fig = plt.figure(figsize=(8, 6))

plt.errorbar(x, y_n, y_error, label='Sample', fmt='o', ms=3, capsize=4, capthick=1, elinewidth=1)
plt.plot(x, nf.pdf(x, *est), label='Fitted distribution')
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

# save the figure
plt.savefig('results/part_e_result.png')

plt.show()
