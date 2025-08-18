# file: distributed_jax_example.py
import numpy as np
import jax
import jax.numpy as jnp
from dask.distributed import Client


# 1) Define & JIT-compile your likelihood at module-scope:
@jax.jit
def log_likelihood(params: jnp.ndarray) -> jnp.ndarray:
    # simple sum, but could be arbitrarily heavy
    return jnp.sum(params)


# 2) Process‐fitness is also at module scope (so dask can pickle it easily)
def process_fitness(params: np.ndarray, jitted_fn) -> float:
    # convert incoming numpy→device array, call the jitted fn,
    # and return a Python float
    return float(jitted_fn(jnp.array(params)))


def main():
    # 3) fire up your client
    client = Client(processes=True, n_workers=4, threads_per_worker=1)

    # 4) scatter the **compiled** function, once:
    jitted_future = client.scatter(log_likelihood, broadcast=True)

    # 5) generate your toy batch of 100×19
    n_dim, n_batch = 19, 100

    points = np.random.uniform(size=(n_batch, n_dim))

    # 6) loop a few times—Dask will reuse the same jitted_fn on each worker,
    #    so you only pay the compile cost exactly once.
    for _ in range(500):

        # client.map can take multiple iterables; we repeat the jitted_future
        futures = client.map(
            process_fitness,
            points,  # each element is a (19,) array
            [jitted_future] * n_batch,  # broadcast the same future
        )
        results = client.gather(futures)
        print(results[0])

    client.close()


if __name__ == "__main__":
    main()
