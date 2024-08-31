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
mass = af.Model(al.mp.Isothermal)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

# Source:

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.SersicCore)

source.bulge.radius_break = 0.025
source.bulge.gamma = 0.25
source.bulge.alpha = 3.0

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Analysis__
"""
analysis = al.AnalysisImaging(dataset=dataset)

"""
__Nested Sampler__
"""
search = af.Nautilus(
    path_prefix=path.join("mle", "simple"),
    name="nested",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=4,
    iterations_per_update=5000,
)

result = search.fit(model=model, analysis=analysis)


"""
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.

__Wrap Up__

This script shows how to fit a lens model to data where the lens galaxy's light is not present.

It was a straightforward extension to the modeling API illustrated in `start_here.ipynb`, where one simply removed
the light profiles from the lens galaxy's model.

Models where the source has no light, or other components of the model are omitted can also be easily composed using
the same API manipulation.
"""
