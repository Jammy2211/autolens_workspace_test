"""
Chaining: Parametric To Inversion
=================================
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
__Colors__
"""
color_list = ["g", "r"]

"""
__Pixel Scales__
"""
pixel_scales_list = [0.08, 0.12]

"""
__Dataset__ 
"""
dataset_type = "imaging"
dataset_label = "multi"
dataset_name = "mass_sie__source_sersic"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

imaging_list = [
    al.Imaging.from_fits(
        image_path=path.join(dataset_path, f"{color}_image.fits"),
        psf_path=path.join(dataset_path, f"{color}_psf.fits"),
        noise_map_path=path.join(dataset_path, f"{color}_noise_map.fits"),
        pixel_scales=pixel_scales,
    )
    for color, pixel_scales in zip(color_list, pixel_scales_list)
]

"""
__Masking__
"""
mask_list = [
    al.Mask2D.circular(
        shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
    )
    for imaging in imaging_list
]

imaging_list = [
    imaging.apply_mask(mask=mask) for imaging, mask in zip(imaging_list, mask_list)
]

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("multi", "chaining")

"""
__Analysis (Search 1)__

We create an `Analysis` object for every dataset.
"""
analysis_list = [al.AnalysisImaging(dataset=imaging) for imaging in imaging_list]
analysis = sum(analysis_list)
analysis.n_cores = 1

"""
__Model (Search 1)__

In search 1 we fit a lens model where the source's intensity varies across each analysis, giving N=15
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal, shear=al.mp.ExternalShear
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllSersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
We now make the intensity a free parameter across every analysis object.
"""
analysis = analysis.with_free_parameters(model.galaxies.source.bulge.intensity)

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[1]__parametric",
    unique_tag=dataset_name,
    nlive=50,
)

result_1_list = search.fit(model=model, analysis=analysis)

"""
__Analysis + Positions (Search 2)__
"""
settings_lens = al.SettingsLens(
    threshold=result_1_list.positions_threshold_from(factor=3.0, minimum_threshold=0.2)
)

analysis_list = [
    al.AnalysisImaging(
        dataset=imaging,
        positions=result_1_list[0].image_plane_multiple_image_positions,
        settings_lens=settings_lens,
    )
    for imaging in imaging_list
]
analysis = sum(analysis_list)
analysis.n_cores = 1

"""
__Model (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2.

We switch to pixelized sources, where the regularization coefficient of each source is free to vary across wavelength.
"""
lens = result_1_list[0].model.galaxies.lens
source = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=al.pix.DelaunayMagnification,
    regularization=al.reg.Constant,
)
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = analysis.with_free_parameters(
    model.galaxies.source.regularization.coefficient
)

"""
__Search + Model-Fit__

We now create the non-linear search and perform the model-fit using this model.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[2]__inversion",
    unique_tag=dataset_name,
    nlive=40,
)

result_2_list = search.fit(model=model, analysis=analysis)

"""
__Wrap Up__

In this example, we passed used prior passing to initialize an `EllIsothermal` + `ExternalShear` lens mass model 
using a parametric source and pass this model to a second search which modeled the source using an `Inversion`. 

This was more computationally efficient than just fitting the `Inversion` by itself and helped to ensure that the 
`Inversion` did not go to an unphysical mass model solution which reconstructs the source as a demagnified version
of the lensed image.

__Pipelines__

Advanced search chaining uses `pipelines` that chain together multiple searches to perform complex lens modeling 
in a robust and efficient way. 

The following example pipelines fits an inversion, using the same approach demonstrated in this script of first fitting 
a parametric source:

 `autolens_workspace/imaging/chaining/pipelines/no_lens_light/mass_total__source_pixelized.py`

 __SLaM (Source, Light and Mass)__

An even more advanced approach which uses search chaining are the SLaM pipelines, which break the lens modeling 
processing into a series of fits that first perfect the source model, then the lens light model and finally the lens
mass model. 

The SLaM pipelines begin with a parametric Source pipeline, which then switches to an inversion Source pipeline, 
exploiting the chaining technique demonstrated in this example.
"""
