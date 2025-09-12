import time
from dask.distributed import Client
import numpy as np


class Fitness:
    def __init__(self, delay=0.2):
        self.delay = delay

    def call(self, params):
        time.sleep(self.delay)  # simulate real work
        return float(np.sum(params))


class FitnessActor(Fitness):
    pass


def main():

    n_batch = 20
    n_dim = 19

    for n_workers in [1, 4, 8]:

        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
        print("Dashboard:", client.dashboard_link)

        actors = [
            client.submit(FitnessActor, actor=True).result() for _ in range(n_workers)
        ]

        points = np.random.rand(n_batch, n_dim)

        import time as t

        start = t.time()

        # Submit all 100 jobs first
        futures = []

        for i, pt in enumerate(points):
            actor = actors[i % len(actors)]
            futures.append(actor.call(pt))  # returns ActorFuture

        # Wait for all 100 to finish together
        results = [f.result() for f in futures]

        end = t.time()

        print("Results:", results)
        print(f"Time: for n workers = {n_workers} time = {end - start:.2f} sec")

        client.close()


if __name__ == "__main__":
    main()
