import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
import tqdm
import demes
import msprime
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt

# Define population sizes and divergence times
mu = 2 * 1.2 * 10 ** -8

from scipy.special import gammaincc, gammaln

def coalescent_rate(population_size):
    return 1 / (population_size)

def precise_estimate(population_sizes, divergence_times, count: int, length: int, normalization = True) -> float:

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
    return np.log(res)


# Define the demographic model
def create_deme_graph(gen_population_sizes, gen_divergence_times):
    b = demes.Builder(
        description="Neanderthal Admixture Model",
        time_units="generations",
    )

    # Define the demes
    b.add_deme("N5", epochs=[dict(start_size=gen_population_sizes["N5"], end_time=gen_divergence_times["N4"], end_size=gen_population_sizes["N5"])])
    b.add_deme("N4", ancestors=["N5"], epochs=[dict(start_size=gen_population_sizes["N4"], end_time=gen_divergence_times["N3"], end_size=gen_population_sizes["N4"])])
    b.add_deme("N3", ancestors=["N4"], epochs=[dict(start_size=gen_population_sizes["N3"], end_time=gen_divergence_times["N2"], end_size=gen_population_sizes["N3"])])
    b.add_deme("N2", ancestors=["N3"], epochs=[dict(start_size=gen_population_sizes["N2"], end_time=gen_divergence_times["N1"], end_size=gen_population_sizes["N2"])])
    b.add_deme("N1", ancestors=["N2"], epochs=[dict(start_size=gen_population_sizes["N1"], end_time=0, end_size=gen_population_sizes["N1"])])
    # Build the graph
    graph = b.resolve()
    return graph

# Convert the demes graph to an msprime demographic model
def convert_to_msprime(graph):
    demography = msprime.Demography.from_demes(graph)
    return demography

# Simulate a genome for two European individuals
def simulate_genome(demography, seed=42, k=2):
    length = 100_000
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(k, ploidy=2, population="N1")],  # Two individuals from the European population
        demography=demography,
        sequence_length=length,  # Simulate a 100 Mb genome
        random_seed=seed,  # For reproducibility
        #recombination_rate=1e-8
    )
    return ts

# Simulate mutations on the tree sequence
def add_mutations(ts, mutation_rate=1.2*1e-8):
    mutated_ts = msprime.sim_mutations(
        ts,
        rate=mutation_rate,  # Mutation rate per base pair per generation
        random_seed=np.random.randint(1, 10 ** 9)
    )
    return mutated_ts

def calculate_t_mrca(ts, x):
    # Get the two sample nodes
    sample_nodes = [0 + x * 2, 1 + x * 2]  # First two nodes correspond to the two samples

    # Calculate T_MRCA for each tree and average
    t_mrca_times = []
    for tree in ts.trees():
        # Find the MRCA of the two samples
        mrca_node = tree.mrca(sample_nodes[0], sample_nodes[1])
        # Get the time of the MRCA
        t_mrca = tree.time(mrca_node)
        t_mrca_times.append(t_mrca)

    # Return the average T_MRCA
    return np.mean(t_mrca_times)


# Compare two individuals and count differences in overlapping segments
def count_differences_in_overlapping_segments(ts, x, k=2):
    # Get the genotype matrix for all individuals
    genotype_matrix = ts.genotype_matrix()

    # The first two columns are for the first individual, the next two for the second individual

    # Count the number of differences
    res = []
    for i in range(k):
        for j in range(i+1, k):
            ind1_genotypes = genotype_matrix[:, i * 2 + x * k * 2]  # First individual (two chromosomes)
            ind2_genotypes = genotype_matrix[:, j * 2 + x * k * 2]  # Second individual (two chromosomes)
            differences = 0
            for site_idx in range(ts.num_sites):
                # Get the alleles for each individual
                ind1_alleles = ind1_genotypes[site_idx]  # Alleles for the first individual
                ind2_alleles = ind2_genotypes[site_idx]  # Alleles for the second individual

                # Check if the alleles differ
                if ind1_alleles != ind2_alleles:
                    differences += 1
            res.append(differences)

    return res

class Test:
    def __init__(self, diff, len, t):
        self.t_mrca = t
        self.diff = diff
        self.len = len

# Main workflow
def generate_test(tp, gen_sizes, gen_times, k=2, seed=42):
    # Create the demographic model
    graph = create_deme_graph(gen_sizes, gen_times)

    # Convert to msprime demography
    demography = convert_to_msprime(graph)

    # Simulate a genome for two individuals
    ts = simulate_genome(demography, seed, k)

    # Add mutations to the tree sequence
    mutation_rate = 1.2e-8  # Mutation rate per base pair per generation
    mutated_ts = add_mutations(ts, mutation_rate)

    # Compare the two individuals and count differences in overlapping segments
    differences = count_differences_in_overlapping_segments(mutated_ts, tp, k)
    length = mutated_ts.sequence_length
    return Test(differences, length, calculate_t_mrca(ts, tp))


''' TESTING '''

from concurrent.futures import ThreadPoolExecutor

TESTCOUNT = 5000
K = 2

def task(N2):
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

    result = []
    for step in range(10):
        samples = []
        print("step #", step)
        all = []
        for i in range(TESTCOUNT):
            samples.append([])
            test = generate_test(0, gen_population_sizes, gen_divergence_times, K, np.random.randint(1, 10000))
            for j in range(K * (K - 1) // 2):
                samples[-1].append((test.diff[j], test.len))
                all.append(test.diff[j])
        

        def loss(params, samples):
            N2, N3 = params
            population_sizes["N2"] = N2

            total_log = 0
            for sample in samples:
                for diff, len_ in sample:
                    p = precise_estimate(population_sizes, divergence_times, diff, len_, normalization=False)
                    total_log += p
            return -total_log
        optimized_N2 = 100
        fine = np.inf
        for N2_pred in tqdm.tqdm(range(100, 4000, 20)):
            x = loss((N2_pred, N2_pred), samples)
            if fine > x:
                #print(f"New opt: {N2}, fine: {x}, prev: {fine}")
                fine = x
                optimized_N2 = N2_pred
        result.append(optimized_N2 // 2)
        #print(optimized_N2)
    avg = np.average(result)
    return (N2, result, avg, abs(N2 - avg) / N2)

final_info = []
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(task, c) for c in [i * 100 for i in range(1, 21, 2)]]
    for future in futures:
        final_info.append(future.result())
for entry in final_info:
    c, result, avg, s = entry
    print(f"N2={c}, predictions: {result}, average: {avg}, error: {abs(c - avg) / c}")