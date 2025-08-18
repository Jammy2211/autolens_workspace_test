import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def logLlikelihood(parameters):
    return jnp.sum(parameters)


def prior_transform(cube):
    return cube + 1


class Sampler:

    def __init__(self, prior, likelihood):

        self.prior = prior
        self.likelihood = likelihood

    def evaluate_likelihood(self, points):

        args = self.prior(points)

        result = list(map(self.likelihood, args))

        return np.array(result)


def fit():

    n_dim = 19
    n_batch = 100

    sampler = Sampler(
        prior=prior_transform,
        likelihood=logLlikelihood,
        #   force_x1_cpu=True
    )

    points_in = np.array(
        [[np.random.uniform(0, 1) for _ in range(n_dim)] for _ in range(n_batch)]
    )

    for i in range(50):

        result = sampler.evaluate_likelihood(points_in)
        print(result)


if __name__ == "__main__":
    fit()
