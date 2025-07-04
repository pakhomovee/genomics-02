from pathos.multiprocessing import ProcessingPool
from simulate import Simulator
from model import Predictor
import numpy as np
import tqdm
from params import GLOBAL_PARAMS


class Tester:
    def __init__(self, PARAMS, workers=10):
        self.PARAMS = PARAMS
        self.workers = workers
        self.simulator = Simulator(PARAMS)
        self.predictor = Predictor(PARAMS)
    def run(self, sizes):
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

            TESTCOUNT = self.PARAMS["TESTCOUNT"]
            K = self.PARAMS["K"]

            result = []
            for step in range(self.PARAMS["iters"]):
                samples = []
                print("step #", step)
                all = []
                for i in range(TESTCOUNT):
                    samples.append([])
                    test = self.simulator.generate_test(0, gen_population_sizes, gen_divergence_times, K, np.random.randint(1, 10000))
                    for j in range(K * (K - 1) // 2):
                        samples[-1].append((test.diff[j], test.len))
                        all.append(test.diff[j])
                

                def loss(params, samples):
                    N2, N3 = params
                    population_sizes["N2"] = N2

                    total_log = 0
                    for sample in samples:
                        for diff, len_ in sample:
                            p = self.predictor.precise_estimate(population_sizes, divergence_times, diff, len_, normalization=False)
                            total_log += p
                    return -total_log
                optimized_N2 = 100
                fine = np.inf
                for N2_pred in tqdm.tqdm(range(self.PARAMS["start"], self.PARAMS["end"], self.PARAMS["step"])):
                    x = loss((N2_pred, N2_pred), samples)
                    if fine > x:
                        fine = x
                        optimized_N2 = N2_pred
                result.append(optimized_N2)
            avg = np.average(result)
            return (N2, result, avg, abs(N2 - avg) / N2)

        final_info = []
        with ProcessingPool(self.workers) as executor:
            for i in range(0, len(sizes), self.workers):
                results = executor.map(task, sizes[i:min(i+self.workers, len(sizes))])
                final_info.extend(results)
        f = open("runs/tmp.txt", "w")
        for entry in final_info:
            c, result, avg, s = entry
            f.write(f"N2={c}, predictions: {result}, average: {avg}, error: {abs(c - avg) / c}\n")
        f.close()


if __name__ == "__main__":
    tester = Tester(GLOBAL_PARAMS, workers=7)
    tester.run([900, 1100, 1300, 1500, 1700, 1900, 2100])