from autoconf import cached_property
import time as t
import jax
import jax.numpy as jn
from dask.distributed import Client
import numpy as np
from os import path

import autofit as af
import autolens as al


import time
from dask.distributed import Client
import numpy as np

from time import perf_counter


def profile_one_call(fitness_obj, params):
    """
    Run on a worker. fitness_obj is the (prebuilt) Fitness instance already on worker.
    This function returns timing breakdown for one params vector.
    """
    import numpy as np

    t0 = perf_counter()

    # Stage 1: prepare inputs (numpy -> device or any conversions)
    t_prep0 = perf_counter()
    params_np = np.asarray(params)
    prep_s = perf_counter() - t_prep0

    # Stage 2: call Python wrapper that may make small allocations / setup
    t_py0 = perf_counter()
    py_s = perf_counter() - t_py0

    # Stage 3: call the actual likelihood computation
    t_compute0 = perf_counter()
    out = fitness_obj.call(params_np)  # or whatever entrypoint you have
    # If returns JAX DeviceArray: do out.block_until_ready()
    try:
        out.block_until_ready()
    except Exception:
        pass
    compute_s = perf_counter() - t_compute0

    t_total = perf_counter() - t0
    return dict(prep_s=prep_s, py_s=py_s, compute_s=compute_s, total_s=t_total)


class Fitness:
    def __init__(self, delay=0.2, analysis=None, model=None):
        self.delay = delay
        self.analysis = analysis
        self.model = model

    def call(self, params):

        instance = self.model.instance_from_vector(vector=params)

        # Evaluate log likelihood (must be side-effect free and exception-free)
        return self.analysis.log_likelihood_function(instance=instance)


class FitnessActor(Fitness):
    pass


def main():

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
    )

    fitness = Fitness(delay=0.2, analysis=analysis, model=model)

    n_batch = 20
    n_dim = 19

    points = np.random.rand(n_dim)

    stuff = profile_one_call(fitness_obj=fitness, params=points)

    print(stuff)


if __name__ == "__main__":
    main()
