from scipy.special import gammaincc, gammaln
import numpy as np

class Predictor:
    def __init__(self, PARAMS):
        self.PARAMS = PARAMS
    
    def coalescent_rate(self, population_size):
        '''
        Computes coalescent_rate for specified effective population size and ploidy as in PARAMS

        Parameters:
            population_size (float): effective population size

        Returns:
            float: coalescent_rate
        '''
        return 1 / (population_size * self.PARAMS["ploidy"])
    
    def precise_estimate(self, population_sizes, divergence_times, count: int, length: int, normalization = True) -> float:
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
                return -(np.exp(-(t-t0)/N - self.PARAMS["mu"] * length * t)) / (length * N * self.PARAMS["mu"] + 1)
            res = -np.exp(t0 / N) * expn_negative_k(count, (N * length * self.PARAMS["mu"] + 1) * t / N, self.PARAMS["mu"] * length * t)
            return res * t / N
            #return -np.exp(t0 / N) * expn_negative_k(count, (N * length * mu + 1) * t / N) * (mu * length * t) ** count / (N * math.factorial(count))

        rate1 = self.coalescent_rate(population_sizes["N1"])
        rate2 = self.coalescent_rate(population_sizes["N2"])
        rate3 = self.coalescent_rate(population_sizes["N3"])
        rate4 = self.coalescent_rate(population_sizes["N4"])
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
            res += (F(borders[i], population_sizes[f"N{i+1}"] * self.PARAMS["ploidy"], borders[i + 1]) - F(borders[i], population_sizes[f"N{i+1}"] * self.PARAMS["ploidy"], borders[i])) * quotients[i]

        def normalize(res):
            if res == np.inf:
                return 0
            return min(1, res * 1000)
        if normalization:
            res = normalize(res)
        return np.log(res)