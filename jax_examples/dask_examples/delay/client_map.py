import time
from dask.distributed import Client
import numpy as np


class Fitness:

    def __init__(self, delay=0.2):
        self.delay = delay

    def call(self, params):
        time.sleep(self.delay)  # simulate real work
        return float(np.sum(params))


# 2) Process‐fitness is also at module scope (so dask can pickle it easily)
def process_fitness(params: np.ndarray, fitness_future) -> float:
    # convert incoming numpy→device array, call the jitted fn,
    # and return a Python float
    time.sleep(0.2)
    fitnss = fitness_future
    return fitnss.call(np.array(params))


def main():

    n_batch = 20
    n_dim = 19

    for n_workers in [1, 4, 8]:

        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
        print("Dashboard:", client.dashboard_link)

        fitness = Fitness()

        func = client.scatter(fitness, broadcast=True)

        points = np.random.rand(n_batch, n_dim)

        import time as t

        start = t.time()

        # client.map can take multiple iterables; we repeat the jitted_future
        futures = client.map(
            process_fitness,
            points,  # each element is a (19,) array
            [func] * n_batch,  # broadcast the same future
        )
        results = client.gather(futures)

        end = t.time()

        print("Results:", results)
        print(f"Time: for n workers = {n_workers} time = {end - start:.2f} sec")

        client.close()


if __name__ == "__main__":
    main()
