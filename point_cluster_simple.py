"""
Func Grad: Light Parametric Operated
====================================

This script test if JAX can successfully compute the gradient of the log likelihood of an `Imaging` dataset with a
model which uses operated light profiles.

 __Operated Fitting__

It is common for galaxies to have point-source emission, for example bright emission right at their centre due to
an active galactic nuclei or very compact knot of star formation.

This point-source emission is subject to blurring during data accquisiton due to the telescope optics, and therefore
is not seen as a single pixel of light but spread over multiple pixels as a convolution with the telescope
Point Spread Function (PSF).

It is difficult to model this compact point source emission using a point-source light profile (or an extremely
compact Gaussian / Sersic profile). This is because when the model-image of a compact point source of light is
convolved with the PSF, the solution to this convolution is extremely sensitive to which pixel (and sub-pixel) the
compact model emission lands in.

Operated light profiles offer an alternative approach, whereby the light profile is assumed to have already been
convolved with the PSF. This operated light profile is then fitted directly to the point-source emission, which as
discussed above shows the PSF features.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import copy
import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
from autoconf import conf



"""
__Dataset__
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "point_source" / dataset_name

dataset = al.from_json(
    file_path=dataset_path / "point_dataset_positions_only.json",
)


"""
__Point Solver__
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
__Model__
"""
import pandas as pd

df = pd.read_csv("mass_profiles.csv")

lens_galaxies = {}

i = 0

for _, row in df.iterrows():

    mass = af.Model(al.mp.Isothermal)

    mass.centre.centre_0 = row["center_y(arcsec)"]
    mass.centre.centre_1 = row["center_x(arcsec)"]
    mass.ell_comps.ell_comps_0 = 0.01
    mass.ell_comps.ell_comps_1 = 0.01
    mass.einstein_radius = row["kappa_scale"] / (834.6912404)

    lens_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    lens_galaxies[f"lens_{i}"] = lens_galaxy

    i += 1

centre_list = [
    (0.01, 0.01),
    (0.02, 0.02),
]

points = {}

for i, centre in enumerate(centre_list):

    # Source

    point = af.Model(al.ps.Point)
    point.centre.centre_0 = af.UniformPrior(
        lower_limit=centre[0] - 0.01,
        upper_limit=centre[0] + 0.01,
    )
    point.centre.centre_1 = af.UniformPrior(
        lower_limit=centre[1] - 0.01,
        upper_limit=centre[1] + 0.01,
    )

    points[f"point_{i}"] = point

source_galaxies = {}

for i in range(len(centre_list)):

    source_galaxies[f"source_{i}"] = af.Model(
        al.Galaxy, redshift=1.0, **{f"point_{i}": points[f"point_{i}"]}
    )

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(**lens_galaxies, **source_galaxies))

"""
__Analysis__
"""
analysis_factor_list = []

for i in range(len(centre_list)):

    dataset_analysis = copy.copy(dataset)

    dataset_analysis.name = f"point_{i}"

    model_analysis = model.copy()
    analysis = al.AnalysisPoint(dataset=dataset_analysis, solver=solver)

    instance = model.instance_from_prior_medians()
    tracer = analysis.tracer_via_instance_from(instance=instance)

    positions = solver.solve(tracer=tracer, source_plane_coordinate=centre_list[i])

    print(positions)

    analysis_factor = af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)

    analysis_factor_list.append(analysis_factor)

factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
__Fitness__
"""
import time
from autofit.non_linear.fitness import Fitness

fitness = Fitness(
    model=factor_graph.global_prior_model,
    analysis=factor_graph,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

"""
We now test the JAX-ing of this LH function.
"""
param_vector = np.array(model.physical_values_from_prior_medians)
print(fitness(param_vector))

start = time.time()

print(fitness(param_vector))

print("JAX JIT LOOP Time taken:", time.time() - start)
