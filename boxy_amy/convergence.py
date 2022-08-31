"""
Modeling: Quantity
==================

Ordinary model-fits in **PyAutoLens** fit a lens model to a dataset (e.g. `Imaging`, `Interferometer`). The inferred
lens model then tells us about the properties of the lens galaxy, for example its convergence, potential and
deflection angles.

This script instead fits a lens model directly to a quantity of lens galaxy, which could be its convergence,
potential, deflection angles or another of its quantities.

This fit allows us to fit a quantity of a certain mass profile (e.g. the convergence of an `EllNFW` mass profile) to
the same quantity of a different mass profile (e.g. the convergence of a `EllPowerLaw`). This provides parameters
describing how to translate between two mass profiles as closely as possible, and to understand how similar or
different the mass profiles are.

This script fits a `DatasetQuantity` dataset of a 'galaxy-scale' strong lens with a model. The `DatasetQuantity` is the
convergence map of a `EllNFW` mass model which is fitted by a lens model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `EllPowerLaw`.
 - There is no source galaxy.
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

dataset_path = path.join("dataset", "for_amy", "manga_lens_data_50", "1068_7962-6103")

convergence_2d = al.Array2D.from_fits(file_path=path.join(dataset_path, "true_kappa.fits"), pixel_scales=0.05)

"""
__Grid__

Define the 2D grid the quantity (in this example, the convergence) is evaluated using.
"""
grid_2d = al.Grid2D.uniform(shape_native=convergence_2d.shape_native, pixel_scales=convergence_2d.pixel_scales)

dataset = al.DatasetQuantity(
    data=convergence_2d,
    noise_map=al.Array2D.full(
        fill_value=0.01,
        shape_native=convergence_2d.shape_native,
        pixel_scales=convergence_2d.pixel_scales,
    ),
)

"""
__Masking__

The model-fit requires a `Mask2D` defining the regions of the convergence we fit, which we define and apply to the 
`DatasetQuantity` object.
"""
mask_2d = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=1.0
)

dataset = dataset.apply_mask(mask=mask_2d)

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a lens model where:

 - The lens galaxy's total mass distribution is an `EllPowerLaw` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
"""
mass = af.Model(al.mp.EllIsothermal)
multipole = af.Model(al.mp.MultipoleIsothermalM4)

mass.centre = multipole.centre

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, multipole=multipole)

model = af.Collection(galaxies=af.Collection(lens=lens))

"""
__Search__


"""
search = af.DynestyStatic(
    path_prefix=path.join("manga_for_amy"),
    name="quantity",
    nlive=50,
    number_of_cores=4,
)

"""
__Analysis__

The `AnalysisQuantity` object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `DatasetQuantity` dataset.

This includes a `func_str` input which defines what quantity is fitted. It corresponds to the function of the 
model `Tracer` objects that are called to create the model quantity. For example, if `func_str="convergence_2d_from"`, 
the convergence is computed from each model `Tracer`.
"""
analysis = al.AnalysisQuantity(dataset=dataset, func_str="convergence_2d_from")

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the output folder for live outputs of the results of the fit, including on-the-fly visualization of the best 
fit model!
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format:
"""
print(result.info)

"""
The `Result` object also contains:

 - The model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Plane` and `FitImaging` objects.
 - Information on the posterior as estimated by the `Dynesty` non-linear search. 
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grid
)
tracer_plotter.subplot_tracer()

fit_quantity_plotter = aplt.FitQuantityPlotter(fit=result.max_log_likelihood_fit)
fit_quantity_plotter.subplot_fit_quantity()

dynesty_plotter = aplt.DynestyPlotter(samples=result.samples)
dynesty_plotter.cornerplot()

"""
Checkout `autolens_workspace/*/imaging/modeling/results.py` for a full description of the result object.
"""
