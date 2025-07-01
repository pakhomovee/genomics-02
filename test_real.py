from utilities import Parser
from model import Predictor
import numpy as np
import tqdm
from params import GLOBAL_PARAMS
import sys
from pathos.multiprocessing import ProcessingPool

class Tester:
    def __init__(self, PARAMS, filepath, workers=10):
        self.PARAMS = PARAMS
        self.parser = Parser(PARAMS, filepath)
        self.predictor = Predictor(PARAMS)
        self.workers = workers
    def run(self):
        population_sizes = {
            "N1": 10000,  # European population
            "N2": 100,   # Neanderthal population, admixture time
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

        samples = self.parser.parse()
        print(f"WORKING WITH {len(samples)} samples", file=sys.stderr)

        def loss(params, samples):
            N2, N3 = params
            population_sizes["N2"] = N2

            total_log = 0
            for diff, len_ in samples:
                p = self.predictor.precise_estimate(population_sizes, divergence_times, diff, len_, normalization=False)
                total_log += p
            return -total_log
        
        def task(i):
            optimized_N2 = 100
            fine = np.inf
            k = self.workers
            for N2_pred in tqdm.tqdm(range(self.PARAMS["start"] + self.PARAMS["step"] * i, self.PARAMS["end"], self.PARAMS["step"] * k)):
                x = loss((N2_pred, N2_pred), samples)
                if fine > x:
                    fine = x
                    optimized_N2 = N2_pred
            return (optimized_N2, fine)
        with ProcessingPool(self.workers) as executor:
            results = executor.map(task, list(range(self.workers)))
            results.sort(key=lambda x: x[1])
        print(results)
        return results[0][0]

t = Tester(GLOBAL_PARAMS, 'data/task2_data.txt')
print(t.run())