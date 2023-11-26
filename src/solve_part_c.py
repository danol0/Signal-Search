####################################################################################################
# Code these expressions for the probability density in python, in any way you so                  #
# desire. Plugging in some different values for the parameters 𝜽, use a numerical                  #
# integration to convince yourself that the total probability density integrates to unity          #
# in the range 𝑀 ∈ [5, 5.6].                                                                       #
####################################################################################################

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
sg = 0.5

theta = [f, lam, mu, sg]

print("Analytical integration: ", quad(lambda x: af.pdf(x, *theta), α, β))
print("Numba integration: ", quad(lambda x: nf.pdf(x, *theta), α, β))
print("Scipy integration: ", quad(lambda x: sf.pdf(x, *theta), α, β))
