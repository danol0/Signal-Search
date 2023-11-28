from utils.distributions import analyticalImplementation, numbaImplementation, scipyImplementation
from scipy.integrate import quad

# Set boundaries
α = 5
β = 5.6

# Create instances
af = analyticalImplementation(α, β)
nf = numbaImplementation(α, β)
sf = scipyImplementation(α, β)

# Define some parameters
f = 0.2
lam = 0.5
mu = 5.3
sg = 1

theta = [f, lam, mu, sg]

print("Analytical implimentation: ", quad(lambda x: af.pdf(x, *theta), α, β))
print("Numba implimentation: ", quad(lambda x: nf.pdf(x, *theta), α, β))
print("Scipy implimentation: ", quad(lambda x: sf.pdf(x, *theta), α, β))
