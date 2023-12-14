from utils.distributions import numbaImplementation
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('src/utils/mphil.mplstyle')

# Set boundaries
α = 5
β = 5.6

# Create pdf instance
numba = numbaImplementation(α, β)

# Define parameters
f = 0.1
lam = 0.5
mu = 5.28
sg = 0.018
theta = [f, lam, mu, sg]

# Plot the distributions
fig = plt.figure(figsize=(7, 5))

# Set colours
main_colour = '#2A788EFF'
signal_colour = 'tab:orange'
background_colour = '#7AD151FF'

x = np.linspace(α, β, 200, endpoint=False)
plt.plot(x, numba.pdf(x, *theta), label="Total", color=main_colour)
plt.plot(x, f * numba.signal_pdf(x, mu, sg), label="Signal (scaled by f)", linestyle=":", color=signal_colour)
plt.plot(x, (1 - f) * numba.background_pdf(x, lam), label="Background (scaled by 1-f)", linestyle="--", color=background_colour)
plt.xlabel("M")
plt.ylabel("P(M)")
plt.legend()

# save the figure
plt.savefig('report/figures/part_d_result.png')
