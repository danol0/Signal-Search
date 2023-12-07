from utils.distributions import analyticalImplementation, numbaImplementation, scipyImplementation
from scipy.integrate import quad

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

print("Analytical implementation: ", quad(lambda x: analytical.pdf(x, *theta), α, β))
print("Numba implementation: ", quad(lambda x: numba.pdf(x, *theta), α, β))
print("Scipy implementation: ", quad(lambda x: scipy.pdf(x, *theta), α, β))
