from dask.distributed import Client
import time
import numpy as np
import jax.numpy as jnp
import jax
from os import path
from functools import partial

try:
    import h5py
except ImportError:
    pass
import numpy as np


from functools import partial
from multiprocessing import Pool
from pathlib import Path
from scipy.special import logsumexp
from shutil import get_terminal_size
from threadpoolctl import threadpool_limits
from time import time
from warnings import warn

from nautilus.bounds import UnitCube, NautilusBound
from nautilus.pool import initialize_worker, likelihood_worker, NautilusPool


import autofit as af
import autolens as al


def prior_transform(cube, model):
    return model.vector_from_unit_vector(unit_vector=cube)


class Sampler:

    def __init__(
        self,
        prior,
        model,
        likelihood,
        n_dim=None,
        n_live=2000,
        n_update=None,
        enlarge_per_dim=1.1,
        n_points_min=None,
        split_threshold=100,
        n_networks=4,
        neural_network_kwargs=dict(),
        prior_args=[],
        prior_kwargs=dict(),
        likelihood_args=[],
        likelihood_kwargs=dict(),
        n_batch=None,
        n_like_new_bound=None,
        vectorized=False,
        pass_dict=None,
        pool=None,
        seed=None,
        blobs_dtype=None,
        filepath=None,
        resume=True,
    ):

        self.prior = partial(prior, **{"model": model})
        self.model = model

        if callable(prior):
            if n_dim is None:
                raise ValueError(
                    "When passing a function as the 'prior' "
                    + "argument, 'n_dim' cannot be None."
                )
            self.n_dim = n_dim
            if pass_dict is None:
                pass_dict = False
        else:
            self.n_dim = prior.dimensionality()
            if pass_dict is None:
                pass_dict = True

        if self.n_dim <= 1:
            raise ValueError("Cannot run Nautilus with less than 2 parameters.")

        self.n_live = n_live

        if n_update is None:
            n_update = n_live
        self.n_update = n_update

        if n_like_new_bound is None:
            n_like_new_bound = 10 * n_live
        self.n_like_new_bound = n_like_new_bound

        self.enlarge_per_dim = enlarge_per_dim

        if n_points_min is None:
            n_points_min = self.n_dim + 50
        self.n_points_min = n_points_min

        self.split_threshold = split_threshold

        self.n_networks = n_networks

        self.neural_network_kwargs = neural_network_kwargs
        self.vectorized = vectorized
        self.pass_dict = pass_dict

        try:
            pool = list(pool)
        except TypeError:
            pool = [pool]

        for i in range(len(pool)):
            if pool[i] in [None, 1]:
                pool[i] = None
            elif i == 0 and isinstance(pool[i], int):
                pool[i] = NautilusPool(pool[i], likelihood=self.likelihood)
                self.likelihood = likelihood_worker
            else:
                pool[i] = NautilusPool(pool[i])
        #

        # self.pool_l = pool[0]

        number_of_cores = 4

        from dask.distributed import Client

        client = Client(processes=True, n_workers=number_of_cores, threads_per_worker=1)

        # 4) Define the worker function
        def process_fitness(params, lookup_fitness) -> float:
            return lookup_fitness(params)

        self.process_fitness = process_fitness

        self.fitness_future = client.scatter(likelihood, broadcast=True)
        self.client_l = client

        self.pool_s = pool[-1]

        if n_batch is None:
            s = number_of_cores
            n_batch = (100 // s + (100 % s != 0)) * s
        self.n_batch = n_batch

        self.rng = np.random.default_rng(seed)

        # The following variables carry the information about the run.
        self.n_like = 0
        self.explored = False
        self.bounds = []
        self.points = []
        self.log_l = []
        self.blobs = None
        self.blobs_dtype = blobs_dtype
        self._discard_exploration = False
        self.shell_n = np.zeros(0, dtype=int)
        self.shell_n_sample = np.zeros(0, dtype=int)
        self.shell_n_eff = np.zeros(0, dtype=float)
        self.shell_log_l_min = np.zeros(0, dtype=float)
        self.shell_log_l = np.zeros(0, dtype=float)
        self.shell_log_v = np.zeros(0, dtype=float)
        self.shell_n_sample_exp = np.zeros(0, dtype=int)
        self.shell_end_exp = np.zeros(0, dtype=int)
        self.points_t = np.zeros((0, self.n_dim))
        self.shell_t = np.zeros(0, dtype=int)
        self.log_l_t = np.zeros(0)
        self.blobs_t = None

        self.filepath = filepath
        if resume and filepath is not None and Path(filepath).exists():
            with h5py.File(filepath, "r") as fstream:

                group = fstream["sampler"]

                self.rng.bit_generator.state = dict(
                    bit_generator="PCG64",
                    state=dict(
                        state=int(group.attrs["rng_state"]),
                        inc=int(group.attrs["rng_inc"]),
                    ),
                    has_uint32=group.attrs["rng_has_uint32"],
                    uinteger=group.attrs["rng_uinteger"],
                )

                for key in [
                    "n_like",
                    "explored",
                    "_discard_exploration",
                    "shell_n",
                    "shell_n_sample",
                    "shell_n_eff",
                    "shell_log_l_min",
                    "shell_log_l",
                    "shell_log_v",
                    "shell_n_sample_exp",
                    "shell_end_exp",
                    "n_update_iter",
                    "n_like_iter",
                ]:
                    setattr(self, key, group.attrs[key])

                for shell in range(len(self.shell_n)):
                    self.points.append(np.array(group["points_{}".format(shell)]))
                    self.log_l.append(np.array(group["log_l_{}".format(shell)]))
                    if "blobs_{}".format(shell) in group:
                        if shell == 0:
                            self.blobs = []
                        self.blobs.append(np.array(group["blobs_{}".format(shell)]))
                        if shell == 0:
                            self.blobs_dtype = self.blobs[-1].dtype

                for key in ["shell_t", "points_t", "log_l_t", "blobs_t"]:
                    if key in group:
                        setattr(self, key, np.array(group[key]))

                self.bounds = [
                    UnitCube.read(fstream["bound_0"], rng=self.rng),
                ]
                for i in range(1, len(self.shell_n)):
                    self.bounds.append(
                        NautilusBound.read(fstream["bound_{}".format(i)], rng=self.rng)
                    )

    def evaluate_likelihood(self, points):

        args = list(map(self.prior, np.copy(points)))
        points = np.array(args)

        futures = [
            self.client_l.submit(self.process_fitness, params, self.fitness_future)
            for params in points
        ]

        result = self.client_l.gather(futures)
        print(result)

        return result

    def sample_shell(self, index, shell_t=None):
        """Sample a batch of points uniformly from a shell.

        The shell at index :math:`i` is defined as the volume enclosed by the
        bound of index :math:`i` and enclosed by not other bound of index
        :math:`k` with :math:`k > i`.

        Parameters
        ----------
        index : int
            Index of the shell.
        shell_t : np.ndarray or None, optional
            If not None, an array of shell associations of possible transfer
            points.

        Returns
        -------
        points : numpy.ndarray
            Array of shape (n_shell, n_dim) containing points sampled uniformly
            from the shell.
        n_bound : int
            Number of points drawn within the bound at index :math:`i`. Will
            be different from `n_shell` if there are bounds with index
            :math:`k` with :math:`k > i`.
        idx_t : np.ndarray, optional
            Indeces of the transfer candidates that should be transferred. Only
            returned if `shell_t` is not None.

        """
        if shell_t is not None and index not in [-1, len(self.bounds) - 1]:
            raise ValueError(
                "'shell_t' must be empty list if not sampling "
                + "from the last bound/shell."
            )

        n_bound = 0
        n_sample = 0
        idx_t = np.zeros(0, dtype=int)
        points_all = []

        with threadpool_limits(limits=1):
            while n_sample < self.n_batch:
                points = self.bounds[index].sample(
                    self.n_batch - n_sample, pool=self.pool_s
                )
                n_bound += self.n_batch - n_sample

                # Remove points that are actually in another shell.
                in_shell = np.ones(len(points), dtype=bool)
                for bound in self.bounds[index:][1:]:
                    in_shell = in_shell & ~bound.contains(points)
                    if np.all(~in_shell):
                        continue
                points = points[in_shell]

                # Replace points for which we can use transfer points.
                replace = np.zeros(len(points), dtype=bool)
                if shell_t is not None and len(shell_t) > 0:
                    shell_p = self.shell_association(points, n_max=len(self.bounds) - 1)
                    for shell in range(len(self.bounds) - 1):
                        idx_1 = np.flatnonzero(shell_t == shell)
                        idx_2 = np.flatnonzero(shell_p == shell)
                        n = min(len(idx_1), len(idx_2))
                        if n > 0:
                            idx_t = np.append(
                                idx_t, self.rng.choice(idx_1, size=n, replace=False)
                            )
                            shell_t[idx_t] = -1
                            replace[self.rng.choice(idx_2, size=n, replace=False)] = (
                                True
                            )

                points = points[~replace]

                if len(points) > 0:
                    points_all.append(points)
                    n_sample += len(points)

        points = np.concatenate(points_all)

        if shell_t is None:
            return points, n_bound
        else:
            return points, n_bound, idx_t

    def add_samples(self, shell, verbose=False):
        """Add samples to a shell.

        The number of new points added is always equal to the batch size.

        Parameters
        ----------
        shell : int
            The index of the shell for which to add points.
        verbose : bool, optional
            If True, print additional information. Default is False.

        Returns
        -------
        n_update : int
            Number of new samples with likelihood equal or higher than the
            likelihood threshold of the bound.

        """
        if verbose:
            self.print_status("Sampling", end="\r")

        if shell == -1 and len(self.shell_t) > 0:
            points, n_bound, idx_t = self.sample_shell(-1, self.shell_t)
            assert len(points) + len(idx_t) == n_bound
            if verbose:
                self.print_status("Computing", end="\r")
            if len(idx_t) > 0:
                self.points[-1] = np.concatenate(
                    (self.points[-1], self.points_t[idx_t])
                )
                self.log_l[-1] = np.concatenate((self.log_l[-1], self.log_l_t[idx_t]))
                if self.blobs is not None:
                    self.blobs[-1] = np.concatenate(
                        (self.blobs[-1], self.blobs_t[idx_t])
                    )
        else:
            points, n_bound = self.sample_shell(shell)
            if verbose:
                self.print_status("Computing", end="\r")

        self.shell_n_sample[shell] += n_bound

        args = list(map(self.prior, np.copy(points)))
        points = np.array(args)

        futures = [
            self.client_l.submit(self.process_fitness, params, self.fitness_future)
            for params in points
        ]

        result = self.client_l.gather(futures)
        print(result)

        return result

    def close(self):

        self.client_l.close()


def fit():

    dataset_name = "simple"
    dataset_path = path.join("dataset", "imaging", dataset_name)

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=0.1,
    )

    mask_2d = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    dataset = dataset.apply_mask(mask=mask_2d)

    dataset = dataset.apply_over_sampling(over_sample_size_lp=1)

    positions = al.Grid2DIrregular(
        al.from_json(file_path=path.join(dataset_path, "positions.json"))
    )

    # Lens:

    bulge = af.Model(al.lp_linear.Sersic)

    mass = af.Model(al.mp.Isothermal)

    shear = af.Model(al.mp.ExternalShear)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

    # Source:

    bulge = af.Model(al.lp_linear.Sersic)

    source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

    # Overall Lens Model:

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
    )

    from autofit.non_linear.fitness import Fitness

    search = af.Nautilus(
        name="imaging_lp_parallel_dask_custom",
        unique_tag=dataset_name,
        n_live=150,
        vectorized=False,
        iterations_per_update=1000,
        number_of_cores=4,
    )

    fitness = Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )

    sampler = Sampler(
        prior=prior_transform,
        model=model,
        likelihood=fitness.call_numpy_wrapper,
        n_dim=model.prior_count,
    )

    params_list = [
        model.random_vector_from_priors,
        model.random_vector_from_priors,
        model.random_vector_from_priors,
        model.random_vector_from_priors,
    ]

    for i in range(10):

        sampler.add_samples(params_list)

    # 7) Clean up
    sampler.close()


if __name__ == "__main__":
    fit()
