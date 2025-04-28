import numpy as np
def coalescent_rate(population_size):
    return 1 / (population_size)

for N2 in [300, 500, 700, 900, 1100, 1300, 1500, 1700]:
    population_sizes = {
        "N1": 10000,  # European population
        "N2": N2,   # Neanderthal population, admixture time
        "N3": 3400,   # Neanderthal population
        "N4": 18500,  # Ancestral population
        "N5": 10000,  # Ancestral population, long time ago
    }

    divergence_times = {
        "N1": 1800,
        "N2": 5000,
        "N3": 18965,
        "N4": 100000,
    }
    gen_population_sizes = {
        "N1": 10000,  # European population
        "N2": N2,   # Neanderthal population
        "N3": 3400,   # Earlier population
        "N4": 18500,   # Earlier population
        "N5": 10000,  # Earliest population
    }

    gen_divergence_times = {
        "N1": 1800,    # N1 diverged from N2 at 2000 generations ago
        "N2": 5000,   # N2 diverged from N3 at 15000 generations ago
        "N3": 18965,   # N3 diverged from N4 at 50000 generations ago
        "N4": 100000,  # N4 diverged from N5 at 100000 generations ago
    }

    rate1 = coalescent_rate(population_sizes["N1"])
    rate2 = coalescent_rate(population_sizes["N2"])
    p = (1 - np.exp(-rate1 * divergence_times["N1"])) * np.exp(-rate2 * (divergence_times["N2"] - divergence_times["N1"])) + np.exp(-rate1 * divergence_times["N1"])
    p = (1 - p)
    print(f'for N2={N2} dropped {p * 2}')