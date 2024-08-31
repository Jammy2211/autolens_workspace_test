"""
Modeling Features: No Lens Light
================================

"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)


"""
__Model__

"""
# Lens:
mass = af.Model(al.mp.PowerLaw)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

source.bulge.radius_break = 0.025
source.bulge.gamma = 0.25
source.bulge.alpha = 3.0

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Analysis__
"""
analysis = al.AnalysisImaging(dataset=dataset)

""""
__MLE__
"""

mass_centre_0 = 0.05  # 0.0
mass_centre_1 = 0.05  # 0.0
mass_ell_comps_0 = 0.05  # 0.0526315789473
mass_ell_comps_1 = 0.02  # 0.0
mass_einstein_radius = 1.7  # 1.6
mass_slope = 2.1  # 2.0

source_centre_0 = 0.05  # 0.0
source_centre_1 = 0.05  # 0.0
source_ell_comps_0 = 0.07  # 0.096
source_ell_comps_1 = -0.02  # -0.0555
source_effective_radius = 0.12  # 0.1
source_sersic_index = 0.9  # 1.0
source_intensity = 3.5  # 4.0

x0 = [
    mass_centre_0,
    mass_centre_1,
    mass_ell_comps_0,
    mass_ell_comps_1,
    mass_einstein_radius,
    mass_slope,
    source_centre_0,
    source_centre_1,
    source_ell_comps_0,
    source_ell_comps_1,
    source_effective_radius,
    source_sersic_index,
    #  source_intensity,
]

search = af.LBFGS(
    path_prefix=path.join("mle", "simple"),
    name="lbfgs",
    unique_tag=dataset_name,
    x0=x0,
    visualize=True,
)

# search = af.BFGS(
#     path_prefix=path.join("mle", "simple"),
#     name="bfgs",
#     unique_tag=dataset_name,
#     x0=x0,
#     visualize=True
# )

result = search.fit(model=model, analysis=analysis)

print(result.log_likelihood)

"""
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.

__Wrap Up__

This script shows how to fit a lens model to data where the lens galaxy's light is not present.

It was a straightforward extension to the modeling API illustrated in `start_here.ipynb`, where one simply removed
the light profiles from the lens galaxy's model.

Models where the source has no light, or other components of the model are omitted can also be easily composed using
the same API manipulation.
"""
