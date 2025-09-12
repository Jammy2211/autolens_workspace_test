import time as t
import jax
import jax.numpy as jnp
from dask.distributed import Client
import numpy as np


# 1) Define & JIT-compile your likelihood at module-scope:
@jax.jit
def log_likelihood(params: jnp.ndarray) -> jnp.ndarray:
    print(">>> JAX compile happening now!")
    return jnp.sum(params)


# 2) Process‐fitness is also at module scope (so dask can pickle it easily)
def process_fitness(params: np.ndarray, jitted_fn) -> float:
    # convert incoming numpy→device array, call the jitted fn,
    # and return a Python float
    return float(jitted_fn(jnp.array(params)))


def main():

    n_batch = 20
    n_dim = 19

    for n_workers in [1, 4, 8]:

        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)

        jitted_future = client.scatter(log_likelihood, broadcast=True)

        points = np.random.rand(n_batch, n_dim)

        # client.map can take multiple iterables; we repeat the jitted_future
        futures = client.map(
            process_fitness,
            points,  # each element is a (19,) array
            [jitted_future] * n_batch,  # broadcast the same future
        )
        results = client.gather(futures)

        start = t.time()

        for i in range(50):
            points = np.random.rand(n_batch, n_dim)

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

        client.close()


if __name__ == "__main__":
    main()
