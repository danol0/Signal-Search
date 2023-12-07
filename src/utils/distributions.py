from numba_stats import truncexpon, truncnorm
from scipy.stats import truncexpon as texpon
from scipy.stats import truncnorm as tnorm
from scipy.special import erf
from math import e, pi

# This module contains the analytical, numba and scipy implementations of the various 
# distribution functions. They have been separated into different classes for ease of use

# Aside from the section comparing their efficiencies, the project has been completed 
# exclusively using the numba implementation
class numbaImplementation:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def background_pdf(self, M, lam):
        return truncexpon.pdf(M, xmin=self.a, xmax=self.b, loc=self.a, scale=1 / lam)

    def signal_pdf(self, M, mu, sg):
        return truncnorm.pdf(M, xmin=self.a, xmax=self.b, loc=mu, scale=sg)

    def pdf(self, M, f, lam, mu, sg):
        return f * self.signal_pdf(M, mu, sg) + (1 - f) * self.background_pdf(M, lam)

    def two_signal_pdf(self, M, f1, f2, lam, mu1, mu2, sg):
        return (
            f1 * self.signal_pdf(M, mu1, sg)
            + f2 * self.signal_pdf(M, mu2, sg)
            + (1 - f1 - f2) * self.background_pdf(M, lam)
        )

    # CDFs for binned fits

    def cdf(self, M, f, lam, mu, sg):
        low, high = (self.a - mu) / sg, (self.b - mu) / sg
        return (
            f * truncnorm.cdf(M, xmin=low, xmax=high, loc=mu, scale=sg) 
            + (1 - f) * truncexpon.cdf(M, xmin=self.a, xmax=self.b, loc=self.a, scale=1 / lam)
        )

    def two_signal_cdf(self, M, f1, f2, lam, mu1, mu2, sg):
        return (
            f1 * truncnorm.cdf(M, xmin=self.a, xmax=self.b, loc=mu1, scale=sg)
            + f2 * truncnorm.cdf(M, xmin=self.a, xmax=self.b, loc=mu2, scale=sg)
            + (1 - f1 - f2) * truncexpon.cdf(M, xmin=self.a, xmax=self.b, loc=self.a, scale=1 / lam)
        )


class analyticalImplementation:

    def __init__(self, a, b):
        # bounds for the truncated distribution are set when the class is initialised
        self.a = a
        self.b = b

    def background_pdf(self, M, lam):
        return lam * e ** (-lam * M)

    def background_cdf(self, x, lam):
        return 1 - e ** (-lam * x)

    def background_pdf_trunc(self, M, lam):
        return self.background_pdf(M, lam) / (
            self.background_cdf(self.b, lam) - self.background_cdf(self.a, lam)
        )

    def signal_pdf(self, M, mu, sg):
        return 1 / (sg * (2 * pi) ** 0.5) * e ** (-0.5 * ((M - mu) / sg) ** 2)

    def signal_cdf(self, x, mu, sg):
        return 0.5 * (1 + erf((x - mu) / (sg * 2**0.5)))

    def signal_pdf_trunc(self, M, mu, sg):
        return self.signal_pdf(M, mu, sg) / (
            self.signal_cdf(self.b, mu, sg) - self.signal_cdf(self.a, mu, sg)
        )

    def pdf(self, M, f, lam, mu, sg):
        return f * self.signal_pdf_trunc(M, mu, sg) + (1 - f) * self.background_pdf_trunc(M, lam)


class scipyImplementation:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def background_pdf(self, M, lam):
        return texpon.pdf(M, b=(self.b - self.a) * lam, loc=self.a, scale=1 / lam)

    def signal_pdf(self, M, mu, sg):
        low, high = (self.a - mu) / sg, (self.b - mu) / sg
        return tnorm.pdf(M, low, high, loc=mu, scale=sg)

    def pdf(self, M, f, lam, mu, sg):
        return f * self.signal_pdf(M, mu, sg) + (1 - f) * self.background_pdf(M, lam)
