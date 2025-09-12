import time
from dask.distributed import Client
import numpy as np


class Fitness:
    def __init__(self, delay=0.5):
        self.delay = delay

    def call(self, params):
        time.sleep(self.delay)  # simulate real work
        return float(np.sum(params))


class FitnessActor(Fitness):
    pass


def main():

    n_batch = 100
    n_dim = 19

    client = Client(n_workers=8, threads_per_worker=1, processes=True)
    print("Dashboard:", client.dashboard_link)

    actors = [client.submit(FitnessActor, actor=True).result() for _ in range(4)]

    points = np.random.rand(n_batch, n_dim)

    from itertools import cycle

    actor_cycle = cycle(actors)

    import time as t

    start = t.time()

    # from itertools import cycle
    # actor_cycle = cycle(actors)
    # futures = [actor.call(pt) for actor, pt in zip(actor_cycle, points)]
    # results = [f.result() for f in futures]

    # Submit all 100 jobs first
    futures = []

    for i, pt in enumerate(points):
        actor = actors[i % len(actors)]
        futures.append(actor.call_numpy_wrapper(pt))  # returns ActorFuture

    # Wait for all 100 to finish together
    results = [f.result() for f in futures]

    end = t.time()

    print("Results:", results[:5])
    print(f"Time: {end - start:.2f} sec")

    client.close()


if __name__ == "__main__":
    main()
