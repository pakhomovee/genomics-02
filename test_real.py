from utilities import Parser
from model import Predictor
import numpy as np
import tqdm
from params import GLOBAL_PARAMS
import sys
from pathos.multiprocessing import ProcessingPool
from scipy.optimize import minimize, differential_evolution

class Tester:
    def __init__(self, PARAMS, filepath, workers=10):
        self.PARAMS = PARAMS
        self.parser = Parser(PARAMS, filepath)
        self.predictor = Predictor(PARAMS)
        self.workers = workers
    def run(self):
        population_sizes = {
            "N1": 10000,    # European
            "N2": 500,    # Early post-bottleneck (~3–5 kya)
            "N3": 1800,   # Mid-time (~13–17 kya)
            "N4": 3400,   # Prehistoric growth (~25–40 kya)
            "N5": 18500,   # Ancient plateau (~60–90+ kya)
        }

        divergence_times = {
            "N1": 1500,    # Recent event (possibly shaded region)
            "N2": 4000,    # Admixture/transition
            "N3": 10000,   # Intermediate shift
            "N4": 20000,   # Large expansion
        }

        samples = self.parser.parse()
        print(f"WORKING WITH {len(samples)} samples", file=sys.stderr)
        batch_size = len(samples) // self.workers

        def loss(params, samples):
            N2, t2 = params
            population_sizes["N2"] = int(N2)
            divergence_times["N2"] = int(t2)

            total_log = 0
            for diff, len_ in samples:
                p = self.predictor.precise_estimate(population_sizes, divergence_times, diff, len_, normalization=False)
                total_log += p
            return -total_log
        
        def task(i):
            optimized_N2 = 100
            fine = np.inf
            initial_guess = [100, 2000]
            bounds = [(100, 2500),  # x bounds
                        (2000, 10000)]   # y bounds
            data = samples[i * batch_size:min(len(samples), (i + 1) * batch_size)]
            #res = minimize(loss, initial_guess, args=samples, method='L-BFGS-B', bounds=bounds)
            res = differential_evolution(loss, bounds, args=(data,),
                                        strategy='best1bin',
                                        popsize=50,
                                        mutation=(0.3, 1),
                                        recombination=0.5,
                                        tol=1e-8,            # Tighter tolerance
                                        maxiter=2000,        # Increased maximum iterations
                                        polish=True,         # Final local optimization
            )
            print(res.x)
            def critical_area(params):
                # check if params are too close to borders
                if params[0] < bounds[0][0] + 100:
                    return True
                if params[0] > bounds[0][1] - 100:
                    return True
                if params[1] < bounds[1][0] + 1000:
                    return True
                if params[1] > bounds[1][1] - 1000:
                    return True
                return False

            if critical_area(res.x):
                step -= 1
                print("REJECTING PREDICITON: ONE OF PARAMS IS CRITICAL")
                return
            return res.x[0]

        with ProcessingPool(self.workers) as executor:
            results = executor.map(task, list(range(self.workers)))
            results.sort(key=lambda x: x[1])
        print(results)
        return results

t = Tester(GLOBAL_PARAMS, 'data/task2_data.txt', workers=7)
print(t.run())