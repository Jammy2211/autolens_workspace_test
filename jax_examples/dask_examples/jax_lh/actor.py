from autoconf import cached_property
import time as t
from itertools import cycle
import jax
import jax.numpy as jnp
from dask.distributed import Client
import numpy as np


class Fitness:

    def __init__(self):
        pass

    def _sum(self, x: jnp.ndarray) -> jnp.ndarray:

        lh = 0.0

        for i in range(10000):

            lh += jnp.sum(x)

        return lh

    @cached_property
    def single_jit(self):
        print(">>> JAX compile happening now!")
        return jax.jit(self._sum)

    def call(self, params: np.ndarray) -> float:
        arr = np.array(params, dtype=np.float64)
        #  t.sleep(0.1)
        return float(self.single_jit(arr))


class FitnessActor(Fitness):
    pass


def main():

    time_dict = {}

    n_batch = 20
    n_dim = 19

    for n_workers in [1, 2, 4, 8]:

        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
        print("Dashboard:", client.dashboard_link)

        actors = [
            client.submit(FitnessActor, actor=True).result() for _ in range(n_workers)
        ]

        # Submit all 100 jobs first
        futures = []

        points = np.random.rand(n_batch, n_dim)

        for i, pt in enumerate(points):
            actor = actors[i % len(actors)]
            futures.append(actor.call(pt))  # returns ActorFuture

        # Wait for all 100 to finish together
        results = [f.result() for f in futures]

        start = t.time()

        print("STARTING")
        print()
        print()
        print()

        for i in range(50):

            points = np.random.rand(n_batch, n_dim)

            # Submit all 100 jobs first
            futures = []

            for i, pt in enumerate(points):
                actor = actors[i % len(actors)]
                futures.append(actor.call(pt))  # returns ActorFuture

            # Wait for all 100 to finish together
            results = [f.result() for f in futures]

        # Submit all 100 jobs first

        # for i in range(50):
        #
        #     points = np.random.rand(n_batch, n_dim)
        #
        #     actor_cycle = cycle(actors)
        #     futures = [actor.call(pt) for actor, pt in zip(actor_cycle, points)]
        #     results = [f.result() for f in futures]

        end = t.time()

        print()
        print()
        print()
        print("Results:", results)
        print(f"Time: for n workers = {n_workers} time = {end - start:.2f} sec")

        time_dict[n_workers] = end - start

        client.close()

    print(time_dict)


if __name__ == "__main__":
    main()
