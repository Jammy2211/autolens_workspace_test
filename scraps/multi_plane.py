"""
Modeling: Point-Source Position + Fluxes
========================================

In this script, we fit a `PointSourceDataset` with a strong lens model where:

 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear`.
 - The source `Galaxy` is a `PointSource`.
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

Load the strong lens dataset `mass_sie__source_point`, which is the dataset we will use to perform lens modeling.

We begin by loading an image of the dataset. Although we are performing point-source modeling and will not use this
data in the model-fit, it is useful to load it for visualization. By passing this dataset to the model-fit at the
end of the script it will be used when visualizing the results. However, the use of an image in this way is entirely
optional, and if it were not included in the model-fit visualization would simple be performed using grids without
the image.
"""
# dataset_name = "mass_sie__source_point"
# dataset_path = path.join("dataset", "point_source", dataset_name)
#
# image = al.Array2D.from_fits(
#     file_path=path.join(dataset_path, "image.fits"), pixel_scales=0.05
# )
pixel_scale = 0.05

"""
We now load the positions we will fit using point source modeling. We load them as a `Grid2DIrregular` data 
structure, which groups different sets of positions to a common source. This is used, for example, when there are 
multiple source galaxy's in the source plane. For this simple example, we assume there is just one source and just one 
group.
"""
positions_0 = al.Grid2DIrregular(
    grid=[
        (1.62072652, 0.11527572),
        (-1.26041904, 0.32485744),
        (-0.69765808, 1.08274092),
        (-0.52066857, -1.11514952),
    ]
)
positions_1 = al.Grid2DIrregular(
    grid=[
        (-2.34163794, -0.29739571),
        (1.95629793, -0.56458603),
        (0.73609466, -1.85197399),
        (0.49242136, 1.77971741),
    ]
)

print(positions_0.in_list)
print(positions_1.in_list)

"""
We also load the observed fluxes of the point source at every one of these position. We load them as 
a `ValuesIrregular` data  structure, which groups different sets of positions to a common source. This is used, 
for example, when there are  multiple source galaxy's in the source plane. For this simple example, we assume there 
is just one source and just one group.
"""
fluxes_0 = al.ValuesIrregular(values=[4.18575419, 13.03704258, 9.04411441, 4.89700567])
fluxes_1 = al.ValuesIrregular(values=[4.71145604, 9.70135635, 7.05226917, 3.98374201])

print(fluxes_0.in_list)
print(fluxes_1.in_list)

"""
We can now plot our positions dataset over the observed image.
"""
visuals_2d = aplt.Visuals2D(positions=[positions_0, positions_1])

# array_plotter = aplt.Array2DPlotter(array=image, visuals_2d=visuals_2d)
# array_plotter.figure()

"""
We can also just plot the positions, omitting the image.
"""
grid_plotter = aplt.Grid2DPlotter(grid=positions_0)
grid_plotter.figure()

"""
For point-source modeling, we also need the noise of every measured position. This is simply the pixel-scale of our
observed dataset, which in this case is 0.05".

The `position_noise_map` should have the same structure as the `Grid2DIrregular`. In this example, the positions
are a single group of 4 (y,x) coordinates, therefore their noise map should be a single group of 4 floats. We can
make this noise-map by creating a `ValuesIrregular` structure from the `Grid2DIrregular`.

We also create the noise map of fluxes, which for simplicity here I have entered manually.
"""
positions_0_noise_map = positions_0.values_from_value(value=pixel_scale)
positions_1_noise_map = positions_1.values_from_value(value=pixel_scale)

print(positions_0_noise_map)
print(positions_1_noise_map)

fluxes_noise_map_0 = al.ValuesIrregular(values=[1.0, 1.0, 1.0, 1.0])
fluxes_noise_map_1 = al.ValuesIrregular(values=[1.0, 1.0, 1.0, 1.0])

"""
__PointSourceDataset__

We next create a `PointSourceDataset` which contains the positions, fluxes and their noise-maps. 

It also names the the dataset. This `name` pairs the dataset to the `PointSource` in the model below. Specifically, 
because we name the dataset `point_0`, there must be a corresponding `PointSource` in the model below with the name 
`point_0` for the model-fit to be possible.

In this example, where there is just one source, named pairing appears uncessary. However, point-source datasets may
have many source galaxies in them, and name pairing ensures every point source in the model is compared against its
point source dataset.
"""
point_source_dataset_0 = al.PointSourceDataset(
    name="point_0",
    positions=positions_0,
    positions_noise_map=positions_0_noise_map,
    fluxes=fluxes_0,
    fluxes_noise_map=fluxes_noise_map_0,
)
point_source_dataset_1 = al.PointSourceDataset(
    name="point_1",
    positions=positions_1,
    positions_noise_map=positions_1_noise_map,
    fluxes=fluxes_1,
    fluxes_noise_map=fluxes_noise_map_1,
)

"""
We now create the `PointSourceDict`, which is a dictionary of every `PointSourceDataset`. Again, because we only have 
one dataset the use of this class seems unecessary, but it is important for model-fits containing many point sources.
"""
point_source_dict = al.PointSourceDict(
    point_source_dataset_list=[point_source_dataset_0, point_source_dataset_1]
)

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a lens model where:

 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear` [7 parameters].
 - The source galaxy's light is a parametric `EllSersic` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.

NOTE: 

**PyAutoLens** assumes that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/notebooks/preprocess`). 
 - Manually override the lens model priors (`autolens_workspace/notebooks/imaging/modeling/customize/priors.py`).
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal)
source_0 = af.Model(
    al.Galaxy, redshift=1.0, mass=al.mp.EllIsothermal, point_0=al.ps.PointSourceFlux
)
source_1 = af.Model(al.Galaxy, redshift=2.0, point_1=al.ps.PointSourceFlux)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source_0=source_0, source_1=source_1)
)

"""
__PositionsSolver__

For point-source modeling we also need to define our `PositionsSolver`. This object determines the multiple-images of 
a mass model for a point source at location (y,x) in the source plane, by iteratively ray-tracing light rays to the 
source-plane. 

Checkout the script ? for a complete description of this object, we will use the default `PositionSolver` in this 
exampl
"""
grid = al.Grid2D.uniform(
    shape_native=(150, 150), pixel_scales=(pixel_scale, pixel_scale)
)

positions_solver = al.PositionsSolver(grid=grid, pixel_scale_precision=0.02)

"""
__Search__

The lens model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Dynesty (https://dynesty.readthedocs.io/en/latest/).

The folder `autolens_workspace/notebooks/imaging/modeling/customize/non_linear_searches` gives an overview of the 
non-linear searches **PyAutoLens** supports. If you are unclear of what a non-linear search is, checkout chapter 2 of 
the **HowToLens** lectures.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autolens_workspace/output/imaging/mass_sie__source_sersic/mass[sie]_source[bulge]`.
"""
search = af.DynestyStatic(
    path_prefix=path.join("point_source", "multi_plane"),
    name="mass[sie]_source[point_flux]",
    nlive=50,
)

"""
__Analysis__

The `AnalysisPointSource` object defines the `log_likelihood_function` used by the non-linear search to fit the model 
to the `PointSourceDataset`.
"""
analysis = al.AnalysisPointSource(
    point_source_dict=point_source_dict, solver=positions_solver
)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the folder `autolens_workspace/output/point_source/mass_sie__source_point/mass[sie]_source[point_flux]` for 
live outputs  of the results of the fit, including on-the-fly visualization of the best fit model!
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which includes: 

 - The lens model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Tracer` object.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grid
)
tracer_plotter.subplot_tracer()

"""
Checkout `autolens_workspace/notebooks/modeling/results.py` for a full description of the result object.
"""
