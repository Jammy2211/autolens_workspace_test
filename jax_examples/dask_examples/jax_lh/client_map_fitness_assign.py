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
        return float(self.single_jit(arr))

    def __call__(self, params):
        """This allows the Fitness instance to be called like a function."""
        return self.call(params)


# Module-level helper so dask can pickle it easily
def process_fitness(params: np.ndarray, fitness_obj: Fitness) -> float:
    """Run the fitness call. fitness_obj is expected to be the scattered object on the worker."""
    return float(fitness_obj(params))


def _compile_on_worker(fitness_obj: Fitness) -> bool:
    """
    Force the fitness object's `.single_jit` property to be accessed on the worker,
    which triggers compilation there. Returns True when done.
    """
    _ = fitness_obj.single_jit
    return True


def main():
    n_batch = 20
    n_dim = 100

    time_dict = {}

    for n_workers in [1, 2, 4, 8]:

        # Use processes=True so each worker has its own process / JAX compilation cache.
        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
        print("Started client:", client)

        # Identify workers' addresses
        worker_addrs = list(client.scheduler_info()["workers"].keys())
        print("Workers:", worker_addrs)

        # Create and scatter one Fitness instance to each worker (not broadcast)
        per_worker_futures = {}
        for addr in worker_addrs:
            # instantiate on driver but scatter to specific worker
            f_obj = Fitness()
            future = client.scatter(f_obj, workers=[addr], broadcast=False)
            per_worker_futures[addr] = future

        # Optional: force compile once per worker so the cost doesn't show up in the first measurement
        precompile_futures = [
            client.submit(_compile_on_worker, per_worker_futures[addr], workers=[addr])
            for addr in worker_addrs
        ]
        # Wait until all workers have performed the compilation
        client.gather(precompile_futures)
        print("Pre-compilation done on all workers.")

        # Warm-up single task (optional)
        points = np.random.rand(n_batch, n_dim).astype(np.float64)

        futures = []
        for i, pt in enumerate(points):
            addr = worker_addrs[i % len(worker_addrs)]
            # schedule process_fitness on the worker that holds the Fitness object
            futures.append(
                client.submit(
                    process_fitness, pt, per_worker_futures[addr], workers=[addr]
                )
            )

        print("STARTING")
        print()
        print()
        print()

        # Test run (measure timings)
        start = t.time()

        for i in range(50):

            points = np.random.rand(n_batch, n_dim).astype(np.float64)

            futures = []

            for i, pt in enumerate(points):
                addr = worker_addrs[i % len(worker_addrs)]
                # schedule process_fitness on the worker that holds the Fitness object
                futures.append(
                    client.submit(
                        process_fitness, pt, per_worker_futures[addr], workers=[addr]
                    )
                )
            results = client.gather(futures)

        print(results)
        end = t.time()
        print(f"Time for {n_workers} workers: {end - start:.2f} sec")
        print()
        print()
        print()
        time_dict[n_workers] = end - start

        client.close()

    print(time_dict)


if __name__ == "__main__":
    main()
