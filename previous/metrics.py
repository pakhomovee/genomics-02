import numpy as np
import matplotlib.pyplot as plt
import math

import scipy
from scipy.integrate import quad

# Define population sizes and divergence times
population_sizes = {
    "N1": 10000,  # European population
    "N2": 500,   # Neanderthal population, admixture time
    "N3": 3400,   # Neanderthal population
    "N4": 18500,  # Ancestral population
    "N5": 10000,  # Ancestral population, long time ago
}

divergence_times = {
    "N1": 1800,
    "N2": 10000,
    "N3": 18965,
    "N4": 100000,
}

mu = 1.2 * 10 ** -8

# Define the coalescent rate function
def coalescent_rate(population_size):
    return 1 / (population_size)

def t_mrca_pdf(t):
    if t < divergence_times["N1"]:
        # Coalescent in N1
        rate = coalescent_rate(population_sizes["N1"])
        return rate * np.exp(-rate * t)
    elif t < divergence_times["N2"]:
        # Coalescent in N2, conditioned on non-coalescence in N1
        rate1 = coalescent_rate(population_sizes["N1"])
        rate2 = coalescent_rate(population_sizes["N2"])
        return rate2 * np.exp(-rate2 * (t - divergence_times["N1"])) * np.exp(-rate1 * divergence_times["N1"])
    elif t < divergence_times["N3"]:
        # Coalescent in N3, conditioned on non-coalescence in N1 and N2
        rate1 = coalescent_rate(population_sizes["N1"])
        rate2 = coalescent_rate(population_sizes["N2"])
        rate3 = coalescent_rate(population_sizes["N3"])
        return (
            rate3 * np.exp(-rate3 * (t - divergence_times["N2"])) *
            np.exp(-rate1 * divergence_times["N1"]) *
            np.exp(-rate2 * (divergence_times["N2"] - divergence_times["N1"]))
        )
    elif t < divergence_times["N4"]:
        # Coalescent in N4, conditioned on non-coalescence in N1, N2, and N3
        rate1 = coalescent_rate(population_sizes["N1"])
        rate2 = coalescent_rate(population_sizes["N2"])
        rate3 = coalescent_rate(population_sizes["N3"])
        rate4 = coalescent_rate(population_sizes["N4"])
        return (
            rate4 * np.exp(-rate4 * (t - divergence_times["N3"])) *
            np.exp(-rate1 * divergence_times["N1"]) *
            np.exp(-rate2 * (divergence_times["N2"] - divergence_times["N1"])) *
            np.exp(-rate3 * (divergence_times["N3"] - divergence_times["N2"]))
        )
    else:
        # Coalescent in N5, conditioned on non-coalescence in N1, N2, N3, and N4
        rate1 = coalescent_rate(population_sizes["N1"])
        rate2 = coalescent_rate(population_sizes["N2"])
        rate3 = coalescent_rate(population_sizes["N3"])
        rate4 = coalescent_rate(population_sizes["N4"])
        rate5 = coalescent_rate(population_sizes["N5"])
        return (
            rate5 * np.exp(-rate5 * (t - divergence_times["N4"])) *
            np.exp(-rate1 * divergence_times["N1"]) *
            np.exp(-rate2 * (divergence_times["N2"] - divergence_times["N1"])) *
            np.exp(-rate3 * (divergence_times["N3"] - divergence_times["N2"])) *
            np.exp(-rate4 * (divergence_times["N4"] - divergence_times["N3"]))
        )

def Pois(k, len, t):
    L = mu * t * 2 * len
    return np.exp(-L) * np.power(L, k) / math.factorial(k)

def estimate(count: int, length: int) -> float:
    def f(t):
        return Pois(count, length, t) * t_mrca_pdf(t)
    print(divergence_times)
    print(population_sizes)
    return quad(f, 0, np.inf)[0]

def precise_estimate(count: int, length: int, normalization = True) -> float:
    def f(t):
        return Pois(count, length, t) * t_mrca_pdf(t)

    from scipy.special import gammaincc, gammaln

    def expn_negative_k(k, x, info):
        """
        Compute the generalized exponential integral E_{-k}(x) for positive k.

        Parameters:
            k (float): The positive order of the exponential integral (k > 0).
            x (float): The argument of the exponential integral (x >= 0).

        Returns:
            float: The value of E_{-k}(x).
        """
        if k <= 0:
            raise ValueError("k must be positive.")
        if x < 0:
            raise ValueError("x must be non-negative.")

        # Handle the special case x = 0
        if x == 0:
            if k > 1:
                return 1 / (k - 1)
            else:
                raise ValueError("E_{-k}(0) is undefined for k <= 1.")

        # Compute E_{-k}(x) using the incomplete gamma function
        a = 1 + k  # Parameter for the incomplete gamma function
        gamma_part = gammaincc(a, x)  # Upper incomplete gamma function
        result = (x / info) ** (-k) / x * gamma_part  # Include the gamma function normalization
        return result

    def F(t0, N, t):
        if count == 0:
            return -(np.exp(-(t-t0)/N - mu * length * t)) / (length * N * mu + 1)
        #print(t0, N, t)
        res = -np.exp(t0 / N) * expn_negative_k(count, (N * length * mu + 1) * t / N, mu * length * t)
        return res * t / N
        #return -np.exp(t0 / N) * expn_negative_k(count, (N * length * mu + 1) * t / N) * (mu * length * t) ** count / (N * math.factorial(count))

    rate1 = coalescent_rate(population_sizes["N1"])
    rate2 = coalescent_rate(population_sizes["N2"])
    rate3 = coalescent_rate(population_sizes["N3"])
    rate4 = coalescent_rate(population_sizes["N4"])
    quotients = [
        1,
        np.exp(-rate1 * divergence_times["N1"]),
        np.exp(-rate1 * divergence_times["N1"]) *
        np.exp(-rate2 * (divergence_times["N2"] - divergence_times["N1"])),
        np.exp(-rate1 * divergence_times["N1"]) *
        np.exp(-rate2 * (divergence_times["N2"] - divergence_times["N1"])) *
        np.exp(-rate3 * (divergence_times["N3"] - divergence_times["N2"])),
        np.exp(-rate1 * divergence_times["N1"]) *
        np.exp(-rate2 * (divergence_times["N2"] - divergence_times["N1"])) *
        np.exp(-rate3 * (divergence_times["N3"] - divergence_times["N2"])) *
        np.exp(-rate4 * (divergence_times["N4"] - divergence_times["N3"]))
    ]
    borders = [
        1,
        divergence_times["N1"],
        divergence_times["N2"],
        divergence_times["N3"],
        divergence_times["N4"],
        10 ** 18
    ]
    res = 0
    for i in range(5):
        #print('========')
        #print('precise', (F(borders[i], population_sizes[f"N{i+1}"], borders[i + 1]) - F(borders[i], population_sizes[f"N{i+1}"], borders[i])))
        #print(quad(f, 2000, 15000))
        res += (F(borders[i], population_sizes[f"N{i+1}"], borders[i + 1]) - F(borders[i], population_sizes[f"N{i+1}"], borders[i])) * quotients[i]

    def normalize(res):
        if res == np.inf:
            return 0
        return min(1, res * 1000)
    if normalization:
        res = normalize(res)
    return res