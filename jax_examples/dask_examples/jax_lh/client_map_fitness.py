from autoconf import cached_property
import time as t
import jax
import jax.numpy as jnp
from dask.distributed import Client
import numpy as np


class Fitness:

    def __init__(self, delay=0.5):
        self.delay = delay

    def _sum(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x)

    @cached_property
    def single_jit(self):
        print(">>> JAX compile happening now!")
        return jax.jit(self._sum)

    def call(self, params: np.ndarray) -> float:
        arr = np.array(params, dtype=np.float64)
        return self.single_jit(arr)

    def __call__(self, params):
        """This allows the Fitness instance to be called like a function."""
        return self.call(params)

    def call_numpy_wrapper(self, parameters):

        figure_of_merit = self.__call__(np.array(parameters))

        return figure_of_merit.item()


def process_fitness(params, lookup_fitness) -> float:
    return lookup_fitness.call_numpy_wrapper(params)


def main():

    n_batch = 20
    n_dim = 100

    time_dict = {}

    for n_workers in [1, 2, 4, 8]:

        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)

        fitness = Fitness()

        jitted_future = client.scatter(fitness, broadcast=True)

        # client.map can take multiple iterables; we repeat the jitted_future

        points = np.random.rand(n_batch, n_dim)

        futures = client.map(
            process_fitness,
            points,  # each element is a (19,) array
            [jitted_future] * n_batch,  # broadcast the same future
        )
        results = client.gather(futures)

        print("STARTING")
        print()
        print()
        print()

        start = t.time()

        for i in range(50):

            points = jnp.asarray(np.random.rand(n_batch, n_dim), dtype=jnp.float64)

            # client.map can take multiple iterables; we repeat the jitted_future
            futures = client.map(
                process_fitness,
                points,  # each element is a (19,) array
                [jitted_future] * n_batch,  # broadcast the same future
            )
            results = client.gather(futures)
        #            print(results[0])

        end = t.time()

        print("Results:", results)
        print(f"Time: for n workers = {n_workers} time = {end - start:.2f} sec")

        print()
        print()
        print()

        time_dict[n_workers] = end - start

        client.close()

    print(time_dict)


if __name__ == "__main__":
    main()
