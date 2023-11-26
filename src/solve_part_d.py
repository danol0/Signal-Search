###################################################################################################
#   Assuming that the true values of the parameters take the following values                     #
#   f = 0.1, 𝜆 = 0.5, 𝜇 = 5.28, 𝜎 = 0.018,                                                        #
#   make a plot of the true signal, background and total probability distributions all overlaid.  #
###################################################################################################

from utils.distributions import numbaImplementation
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('src/utils/mphil.mplstyle')

# Set boundaries
α = 5
β = 5.6

# Create instances
nf = numbaImplementation(α, β)

# Define parameters
f = 0.1
lam = 0.5
mu = 5.28
sg = 0.018

theta = [f, lam, mu, sg]

# Plot the distributions
fig = plt.figure(figsize=(8, 6))

x = np.linspace(α, β, 200, endpoint=False)
plt.plot(x, nf.pdf(x, *theta), label="Total")
plt.plot(x, f * nf.signal_pdf(x, mu, sg), label="Signal (scaled by f)", linestyle="--")
plt.plot(x, (1 - f) * nf.background_pdf(x, lam), label="Background (scaled by 1-f)", linestyle="--")
plt.title("Total, signal & background distributions")
plt.xlabel("M")
plt.ylabel("Probability")
plt.legend()

# save the figure
plt.savefig('results/part_d_result.png')

plt.show()
