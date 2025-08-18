import time
from autoconf import cached_property
import time as t
import jax
import jax.numpy as jnp
from itertools import cycle
from dask.distributed import Client
import numpy as np
from os import path

import autofit as af
import autolens as al


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

    from autofit.non_linear.fitness import Fitness

    fitness = Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )

    n_batch = 100
    n_dim = 19

    points = np.random.rand(n_batch, n_dim).astype(np.float64)
    points = af.PriorVectorized(model)((points))

    result = list(map(fitness.call_numpy_wrapper, points))
    print(result)

    start = t.time()

    print("STARTING")
    print()
    print()
    print()

    for i in range(50):

        points = np.asarray(np.random.rand(n_batch, n_dim), dtype=np.float64)

        result = list(map(fitness.call_numpy_wrapper, points))

    end = t.time()

    print("Results:", result)
    print(f"Time: for x1 CPU time = {end - start:.2f} sec")


if __name__ == "__main__":
    main()
