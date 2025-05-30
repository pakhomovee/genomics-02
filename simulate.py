import demes
import msprime
import numpy as np

class Test:
    def __init__(self, diff, len, t):
        self.t_mrca = t
        self.diff = diff
        self.len = len

class Simulator:
    def __init__(self, PARAMS):
        self.PARAMS = PARAMS

    # Define the demographic model
    def create_deme_graph(self, gen_population_sizes, gen_divergence_times):
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

    def convert_to_msprime(self, graph):
        demography = msprime.Demography.from_demes(graph)
        return demography

    def simulate_genome(self, demography, seed=42, k=2):
        length = 100_000
        ts = msprime.sim_ancestry(
            samples=[msprime.SampleSet(k, ploidy=2, population="N1")],
            demography=demography,
            sequence_length=length,
            random_seed=seed,
            recombination_rate=self.PARAMS["rr"]
        )
        return ts

    # Simulate mutations on the tree sequence
    def add_mutations(self, ts, mutation_rate=None):
        if mutation_rate is None:
            mutation_rate = self.PARAMS["mu"] / 2
        mutated_ts = msprime.sim_mutations(
            ts,
            rate=mutation_rate,
            random_seed=np.random.randint(1, 10 ** 9)
        )
        return mutated_ts

    def calculate_t_mrca(self, ts, x):
        sample_nodes = [0 + x * 2, 1 + x * 2]
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
    def count_differences_in_overlapping_segments(self, ts, x, k=2):
        genotype_matrix = ts.genotype_matrix()
        
        # Count the number of differences
        res = []
        for i in range(k):
            for j in range(i+1, k):
                ind1_genotypes = genotype_matrix[:, i * 2 + x * k * 2]  # First individual
                ind2_genotypes = genotype_matrix[:, j * 2 + x * k * 2]  # Second individual
                differences = 0
                for site_idx in range(ts.num_sites):
                    ind1_alleles = ind1_genotypes[site_idx]
                    ind2_alleles = ind2_genotypes[site_idx]
                    # Check if the alleles differ
                    if ind1_alleles != ind2_alleles:
                        differences += 1
                res.append(differences)

        return res

    # Main workflow
    def generate_test(self, tp, gen_sizes, gen_times, k=2, seed=42):
        # Create the demographic model
        graph = self.create_deme_graph(gen_sizes, gen_times)

        # Convert to msprime demography
        demography = self.convert_to_msprime(graph)

        # Simulate a genome for two individuals
        ts = self.simulate_genome(demography, seed, k)

        # Add mutations to the tree sequence
        mutation_rate = self.PARAMS["mu"] / 2
        mutated_ts = self.add_mutations(ts, mutation_rate)

        # Compare the two individuals and count differences in overlapping segments
        differences = self.count_differences_in_overlapping_segments(mutated_ts, tp, k)
        length = mutated_ts.sequence_length
        return Test(differences, length, self.calculate_t_mrca(ts, tp))