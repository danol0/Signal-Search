from utils.distributions import analyticalImplementation, numbaImplementation, scipyImplementation
from scipy.integrate import quad
import time
import numpy as np
import matplotlib.pyplot as plt

# *************************************************************************************************************
# ************************************** Show the PDFs are Normalised *****************************************
# *************************************************************************************************************

# Set boundaries
α = 5
β = 5.6

# Create pdf instances
analytical = analyticalImplementation(α, β)
numba = numbaImplementation(α, β)
scipy = scipyImplementation(α, β)

# Define some parameters
f = 0.2
lam = 0.5
mu = 5.3
sg = 1

theta = [f, lam, mu, sg]

# Show that the integrals are equal to one
print("Analytical implementation: ", quad(lambda x: analytical.pdf(x, *theta), α, β))
print("Numba implementation: ", quad(lambda x: numba.pdf(x, *theta), α, β))
print("Scipy implementation: ", quad(lambda x: scipy.pdf(x, *theta), α, β))

# *************************************************************************************************************
# ****************************************** Test the relative speed ******************************************
# *************************************************************************************************************
print("Evaluating the speed of the various implementations...")

# Seed, although results will vary slightly due to compute differences
np.random.seed(42)

# Evaluate the execution time for a range of sample sizes
numba_times = []
analytical_times = []
scipy_times = []

lists = [numba_times, analytical_times, scipy_times]
pdfs = [numba.pdf, analytical.pdf, scipy.pdf]
xs = np.linspace(1, 1e7, 6, dtype=int)

for i in xs:
    x = np.random.uniform(α, β, i) if i > 1 else np.random.uniform(α, β)
    for list, pdf in zip(lists, pdfs):
        times = []
        # Do three repeats and average
        for _ in range(3):
            start = time.time()
            pdf(x, *theta)
            end = time.time()
            times.append(end - start)
        list.append(np.mean(times))

# plot the results
plt.figure(figsize=(8, 6))
plt.style.use('src/utils/mphil.mplstyle')

# Set colours
blue = '#2A788EFF'
purple = 'tab:orange'
green = 'tab:green'

plt.plot(xs, np.array(analytical_times) / np.array(analytical_times), label='Analytical', color=purple, linestyle='--')
plt.plot(xs, np.array(analytical_times) / np.array(scipy_times), label='SciPy', color=blue)
plt.plot(xs, np.array(analytical_times) / np.array(numba_times), label='Numba', color=green)
plt.xlabel('Input Array Size')
plt.ylabel('Evaluation Speed (relative to analytical)')
plt.legend()
plt.savefig('report/figures/part_c_eval_time.png', bbox_inches='tight')
