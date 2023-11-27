from utils.distributions import numbaImplementation, analyticalImplementation
from utils.tools import accept_reject
import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
plt.style.use('src/utils/mphil.mplstyle')
import matplotlib.gridspec as gs
from numba_stats import norm

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
bins = 85
y, x = np.histogram(sample, bins=bins, range=[α, β])
bin_width = x[1] - x[0]

# shift x to bin centres
x = np.mean(np.vstack([x[1:], x[:-1]]), axis=0)

# normalise y & error
y_n = y / (bin_width * np.sum(y))
y_error = (y_n / (len(sample) * bin_width))**0.5

# calcluate the pulls
pull = (y_n - nf.pdf(x, *est)) / y_error


# plot the sample and fit
fig = plt.figure(figsize=(8, 8))
grid = gs.GridSpec(2, 2, hspace=0, wspace=0, height_ratios=(3, 1), width_ratios=(7, 1))

ax_main = plt.subplot(grid[0, 0])
ax_pulls = plt.subplot(grid[1, 0], sharex=ax_main)
ax_empty = plt.subplot(grid[0, 1])
ax_empty.set_visible(False)
ax_pull_dist = plt.subplot(grid[1, 1], sharey=ax_pulls)

# main plot
ax_main.errorbar(x, y_n, y_error, label='Sample', fmt='o', ms=3, capsize=4, capthick=1, elinewidth=1)
ax_main.plot(x, nf.pdf(x, *est), label='Fitted distribution')
ax_main.plot(x, est['f'] * nf.signal_pdf(x, est['mu'], est['sg']), label='Signal (scaled)', linestyle='--')
ax_main.plot(x, (1 - est['f']) * nf.background_pdf(x, est['lam']), label='Background (scaled)', linestyle='--')
ax_main.set_title('Normalised sample & fitted distribution for 100k events')
ax_main.set_xlabel('M')
ax_main.set_ylabel('Density')

# Add parameter estimates to the legend
legend_text = f"Estimated values:\n"\
              f"f = {est['f']:.2f} ± {mi.errors['f']:.1}\n" \
              f"λ = {est['lam']:.2f} ± {mi.errors['lam']:.1}\n" \
              f"μ = {est['mu']:.2f} ± {mi.errors['mu']:.1}\n" \
              f"σ = {est['sg']:.3f} ± {mi.errors['sg']:.1}"
ax_main.plot([], [], ' ', label=legend_text)
ax_main.legend()

# plot the pulls
ax_pulls.errorbar(x, pull, np.ones_like(x), fmt='o', ms=3, capsize=4, capthick=1, elinewidth=1)
ax_pulls.axhline(0, color='k', linestyle='--')
ax_pulls.set_xlabel('$X$')
ax_pulls.set_ylabel('Pull')

# plot the pull distribution
ax_pull_dist.hist(pull, bins=10, range=(-3, 3), density=True, alpha=0.5, orientation='horizontal')
ax_pull_dist.xaxis.set_visible(False)
ax_pull_dist.spines[['top', 'bottom', 'right']].set_visible(False)
ax_pull_dist.tick_params(which='both', direction='in', axis='y', right=False, labelcolor='none')
xp = np.linspace(-3, 3, 100)
ax_pull_dist.plot(norm.pdf(xp, loc=0, scale=1), xp, 'r-', alpha=0.5)

fig.align_ylabels()

# save the figure
plt.savefig('results/part_e_result.png')
plt.show()
