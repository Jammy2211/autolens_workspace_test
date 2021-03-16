"""
__Example: Chaining Phases API__

In the `beginner` examples all model-fits were performed using one phase, which composed the lens model using one
parametrization and performed the model-fit using one non-linear search. In the `chaining` examples we break the
model-fitting procedure down into multiple phases, chaining the results of the initial phases to subsequent phases.
This allows us to guide the model-fitting procedure as to where it should look in parameter space for the
highest log-likelihood models.

When chaining phases:

 - The earlier phases fit simpler model parameterizations than the later phases, providing them with a less complex
   non-linear parameter space that can be sampled more efficiently and with a reduced chance of inferring an
   incorrect local maxima solution.

 - The earlier phases may use `NonLinearSearch` techniques that only seek to maximize the log likelihood and do not
   precisely quantify the errors on every parameter, whereas the latter phases do. Alternative, they may use a
   `NonLinearSearch` which does compute errors, but with settings that make sampling faster or omit accurately
   quantifying the errors.

This means we can `initialize` a model-fit very quickly and only spend more computational time estimating errors
in the final phase when we actually require them.

 - The earlier phases can use the `SettingsPhaseImaging` object to augment the data or alter the fitting-procedure
   in ways that speed up the computational run time. These may impact the quality of the model-fit overall, but they
   can be reverted to the more accurate but more computationally expense setting in the final phases.

This script gives an overview of the API for phase chaining, a description of how priors are chained and tools for
customizing prior chaining. The other scripts in the `model/chaining` folder give examples of when, for lens modeling,
it is beneficial to chain priors, often changing the model between the two phases.

Prior chaining is crucial for using the PyAutoLens pipelines found in the folder `autolens_workspace/pipelines`. This
example provide a conceptual overview of why prior chaining is used and an introduction to the API used to do so.

More details on prior chaining can be found in Chapter 2 of the HowToLens lectures, specifically
`tutorial_5_chaining_phases.py`.

This example scripts show a simple example of prior chaining, where we fit `Imaging` of a strong lens system where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `EllipticalIsothermal`.
 - The source galaxy's light is a parametric `EllipticalSersic`.

As discussed below, the first phase is set up to provide as fast a model-fit as possible without accurately quantifying
the errors on every parameter, whereas the second phase sacrifices this run-speed for accuracy. 
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
As per usual, load the `Imaging` data, create the `Mask2D` and plot them. In this strong lensing dataset:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `EllipticalIsothermal`.
 - The source galaxy's `LightProfile` is an `EllipticalExponential`.
"""
dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging_plotter = aplt.ImagingPlotter(
    imaging=imaging, visuals_2d=aplt.Visuals2D(mask=mask)
)
imaging_plotter.subplot_imaging()

"""
__Dataset__

Next, we set up the `MaskedImaging` dataset we fit with the lens model. This includes settings describing how the model 
is fitted to the data in the log likelihood function. 

Below, we use the `Mask2D` created above and specify:

 - That a regular `Grid2D` is used to fit create the model-image when fitting the data  (see 
   `autolens_workspace/examples/grids.py` for a description of grids).
 - The sub-grid size of this grid.
"""
masked_imaging = al.MaskedImaging(
    imaging=imaging,
    mask=mask,
    settings=al.SettingsMaskedImaging(grid_class=al.Grid2D, sub_size=2),
)

"""
__Model__

We compose our lens model using `GalaxyModel` objects, which represent the galaxies we fit to our data. In this 
example our lens model is:

 - An `EllipticalIsothermal` `MassProfile` for the lens galaxy's mass (5 parameters).
 - An `EllipticalSersic` `LightProfile` for the source galaxy's light (6 parameters).

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""
lens = al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal)
source = al.GalaxyModel(redshift=1.0, bulge=al.lp.EllipticalSersic)

model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(lens=lens, source=source)
)

"""
__Search__

You should be familiar with non-linear searches from other example scripts. If not checkout the beginner examples
and `autolens_workspace/notebooks/imaging/modeling/customize/non_linear_searches.py`.

For search 1, we again use `Dynesty` however we set a new input parameter the `evidence_tolerance`. This is essentially
the stopping criteria of `Dynesty`, where high values means that it stops sampling earlier, at the expense of less
robust parameter estimates and larger inferred parameter errors. Given we want phase 1 to be fast, we do not mind
either of these things happening. 
    
You should also note the `PriorPasser` object input into the search. We will describe this in a moment, but you
should run the script and model-fit first.

The `name` and `path_prefix` below specify the path where results are stored in the output folder:  

 `/autolens_workspace/output/examples/chaining/api/mass_sie__source_sersic/phase[1]`.
"""
search = af.DynestyStatic(
    path_prefix=path.join("imaging", "chaining", "api"),
    name="search[1]",
    n_live_points=50,
    evidence_tolerance=20.0,
    prior_passer=af.PriorPasser(sigma=5.0, use_widths=True, use_errors=False),
)

"""
__Analyis__

We next create an `AnalysisImaging` object, which is passed the `MaskedImaging` that we created above.
"""
analysis = al.AnalysisImaging(dataset=masked_imaging)

"""
__Model Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find the model-fits with the highest likelihoods.

The fit outputs visualization on-the-fly, so checkout the path `autolens_workspace/output/examples/mass[sie]_source[bulge]` 
to see how your fit is doing!
"""
result = search.fit(model=model, analysis=analysis)

"""
Before reading on to search 2, you may wish to inspect the results of the search 1 model-fit to ensure the fast
non-linear search has provided a reasonably accurate lens model.
"""
print(result)

"""
__Model Chaining__

We use the results of search 1 to create the `GalaxyModel` components that we fit in phase 2.

The term `model` below tells PyAutoLens to pass the lens and source models as model-components that are to be fitted
for by the non-linear search. In other chaining examples, we'll see other ways to pass prior results.
"""
lens = result.model.galaxies.lens
source = result.model.galaxies.source

model = af.CollectionPriorModel(
    galaxies=af.CollectionPriorModel(lens=lens, source=source)
)

"""
__Search__

In search 2, we again use the nested sampling algorithm `Dynesty`. However, we now remove the `evidence_tolerance`,
given we want an accurate lens model and precise errors!

The `name` and `path_prefix` below specify the path where results are stored in the output folder:  

 `/autolens_workspace/output/examples/chaining/api/mass_sie__source_sersic/search[2]`.
"""
search = af.DynestyStatic(
    path_prefix=path.join("imaging", "chaining", "api"),
    name="search[2]",
    n_live_points=50,
)

"""
__Model Fit__

We now perform a second model-fit, reusing the `Analsyis` class we created above but passing the new model which was
created via the results of search 1.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Prior Passing__

Once phase 2 is running, you should checkout its `model.info` file. The parameters do not use the default priors we saw
in the beginner turorials or phase 1 (which are typically broad UniformPriors). Instead, it uses GaussianPrior`s where:

 - The mean values are the median PDF results of every parameter in phase 1.
 - Many sigma values are the errors computed at 3.0 sigma confidence of every parameter in phase 1.
 - Other sigma values are higher than the errors computed at 3.0 sigma confidence. These instead use the value 
      specified in the `width_modifier` field of the `Profile`'s entry in the `json_config` files.

The `width_modifier` is used instead of the errors computed from phase 1 when the errors values estimated are smaller 
than the width modifier value. This ensure that the sigma values used for priors in phase 2 do not assume extremely 
small values (e.g. a value of < 0.01 for an einstein_radius) if the error estimates in phase 1 are very small, which
may occur when using a fast `NonLinearSearch` or fitting an overly simplified model.
    
Thus, phase 2 used the results of phase 1 to inform it where to search non-linear parameter space! 

The PriorPasser customizes how priors are passed from phase 1 as follows:

 - sigma: The sigma value that the errors passed to use as the sigma values in phase 1 are estimated at.
 - use_widths: If False, the "width_modifier" values in the json_prior configs are not used to override a passed
                  error value.
 - use_errors: If False, errors are not passed from phase 1 to set up the priors and only the "width" modifier
                  entries in the configs are used.  

For the interested read a complete description of prior passing is given in chapter 2, tutorial 5 of HowToLens. Below
is an extract of the full prior passing description.

__HowToLens Prior Passing__

Lets say I chain two parameters as follows:

    mass.einstein_radius = phase1_result.model.galaxies.lens.mass.einstein_radius

By invoking the `model` attribute, the prioris passed following 3 rules:

    1) The new parameter, in this case the einstein radius, uses a GaussianPrior. A GaussianPrior is ideal, as the 1D 
       pdf results we compute at the end of a phase are easily summarized as a Gaussian.

    2) The mean of the GaussianPrior is the median PDF value of the parameter estimated in phase 1.

      This ensures that the initial sampling of the new phase`s non-linear starts by searching the region of non-linear 
      parameter space that correspond to highest log likelihood solutions in the previous phase. Thus, we`re setting 
      our priors to look in the `correct` regions of parameter space.

    3) The sigma of the Gaussian will use the maximum of two values: 

            (i) the 1D error of the parameter computed at an input sigma value (default sigma=3.0).
            (ii) The value specified for the profile in the `config/priors/*.json` config file`s `width_modifer` 
                 field (check these files out now).

       The idea here is simple. We want a value of sigma that gives a GaussianPrior wide enough to search a broad 
       region of parameter space, so that the lens model can change if a better solution is nearby. However, we want it 
       to be narrow enough that we don't search too much of parameter space, as this will be slow or risk leading us 
       into an incorrect solution! A natural choice is the errors of the parameter from the previous phase.

       Unfortunately, this doesn`t always work. Lens modeling is prone to an effect called `over-fitting` where we 
       underestimate the errors on our lens model parameters. This is especially true when we take the shortcuts in 
       early phases - fast `NonLinearSearch` settings, simplified lens models, etc.

       Therefore, the `width_modifier` in the json config files are our fallback. If the error on a parameter is 
       suspiciously small, we instead use the value specified in the widths file. These values are chosen based on 
       our experience as being a good balance broadly sampling parameter space but not being so narrow important 
       solutions are missed. 

There are two ways a value is specified using the priors/width file:

    1) Absolute: In this case, the error assumed on the parameter is the value given in the config file. 
       For example, if for the width on centre_0 of a `LightProfile`, the width modifier reads "Absolute" with a value 
       0.05. This means if the error on the parameter centre_0 was less than 0.05 in the previous phase, the sigma of 
       its GaussianPrior in this phase will be 0.05.

    2) Relative: In this case, the error assumed on the parameter is the % of the value of the 
       estimate value given in the config file. For example, if the intensity estimated in the previous phase was 2.0, 
       and the relative error in the config file reads "Relative" with a value 0.5, then the sigma of the GaussianPrior 
       will be 50% of this value, i.e. sigma = 0.5 * 2.0 = 1.0.

We use absolute and relative values for different parameters, depending on their properties. For example, using the 
relative value of a parameter like the `Profile` centre makes no sense. If our lens galaxy is centred at (0.0, 0.0), 
the relative error will always be tiny and thus poorly defined. Therefore, the default configs in PyAutoLens use 
absolute errors on the centre.

However, there are parameters where using an absolute value does not make sense. Intensity is a good example of this. 
The intensity of an image depends on its unit_label, S/N, galaxy brightness, etc. There is no single absolute value 
that one can use to generically chain the intensity of any two proflies. Thus, it makes more sense to chain them using 
the relative value from a previous phase.

We can customize how priors are passed from the results of a phase and `NonLinearSearch` by inputting to the search 
a PriorPasser object:
"""
search = af.DynestyStatic(
    prior_passer=af.PriorPasser(sigma=2.0, use_widths=False, use_errors=True)
)

"""
The PriorPasser allows us to customize at what sigma the error values the model results are computed at to compute
the passed sigma values and customizes whether the widths in the config file, these computed errors, or both, 
are used to set the sigma values of the passed priors.

The default values of the PriorPasser are found in the config file of every non-linear search, in the [prior_passer]
section. All non-linear searches by default use a sigma value of 3.0, use_width=True and use_errors=True. We anticipate
you should not need to change these values to get lens modeling to work proficiently!

__EXAMPLE__

Lets go through an example using a real parameter. Lets say in phase 1 we fit the lens galaxy's light with an 
elliptical Sersic profile, and we estimate that its sersic index is equal to 4.0 +- 2.0 where the error value of 2.0 
was computed at 3.0 sigma confidence. To pass this as a prior to phase 2, we would write:

    lens.bulge.sersic_index = phase1.result.model.lens.bulge.sersic_index

The prior on the lens galaxy's sersic `LightProfile` in phase 2 would thus be a GaussianPrior, with mean=4.0 and 
sigma=2.0. If we had used a sigma value of 1.0 to compute the error, which reduced the estimate from 4.0 +- 2.0 to 
4.0 +- 1.0, the sigma of the Gaussian prior would instead be 1.0. 

If the error on the Sersic index in phase 1 had been really small, lets say, 0.01, we would instead use the value of the 
Sersic index width in the priors config file to set sigma instead. In this case, the prior config file specifies 
that we use an "Absolute" value of 0.8 to chain this prior. Thus, the GaussianPrior in phase 2 would have a mean=4.0 and 
sigma=0.8.

If the prior config file had specified that we use an relative value of 0.8, the GaussianPrior in phase 2 would have a 
mean=4.0 and sigma=3.2.

And with that, we`re done. Chaining priors is a bit of an art form, but one that tends to work really well. Its true to 
say that things can go wrong - maybe we `trim` out the solution we`re looking for, or underestimate our errors a bit 
due to making our priors too narrow. However, in general, things are okay, and the example pipelines in 
`autolens_workspace/pipelines` have been thoroughly tested to ensure prior chaining works effectively.
"""