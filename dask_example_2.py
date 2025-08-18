# file: dask_example_single_vector_actor_sync_result.py

from autoconf import cached_property
import numpy as np

import jax
import jax.numpy as jnp

from dask.distributed import Client


class Fitness:
    def __init__(self, n_dim: int):
        self.n_dim = n_dim

    def _sum(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x)

    @cached_property
    def single_jit(self):
        print(">>> JAX compile happening now!")
        return jax.jit(self._sum)

    def call(self, params: np.ndarray) -> float:
        arr = jnp.array(params, dtype=jnp.float32)
        return float(self.single_jit(arr))


class FitnessActor(Fitness):
    pass


def main():
    # 1) Start your client
    client = Client(n_workers=4, threads_per_worker=1)
    print("Dask dashboard:", client.dashboard_link)

    # 2) Create and unwrap actors
    n_dim = 19
    actors = []
    for _ in range(4):
        afut = client.submit(FitnessActor, n_dim, actor=True)
        actor = afut.result()  # NOW you have the *proxy* with .call()
        actors.append(actor)

    # 3) Generate some test points
    n_batch = 100

    # 4) Run a few iterations, pulling each result with .result()
    for it in range(1, 25):

        points = np.random.rand(n_batch, n_dim).astype(np.float32)

        results = []
        for i, pt in enumerate(points):
            actor = actors[i % len(actors)]
            fut = actor.call(pt)  # This is a Dask Future
            val = fut.result()  # BLOCK until that Future yields a float
            results.append(val)

        # Now results is a list of floats
        print(
            f"Iter {it:2d} → first 5 fitness: {results[:5]} (types: {[type(r) for r in results[:5]]})"
        )

    client.close()


if __name__ == "__main__":
    main()
