"""
__Example: Modeling__

To fit a CTI model to a dataset, we must perform CTI modeling, which uses a `NonLinearSearch` to fit many
different CTI models to the dataset.

Model-fitting is handled by our project **PyAutoFit**, a probabilistic programming language for non-linear model
fitting. The setting up on configuration files is performed by our project **PyAutoConf**. We'll need to import
both to perform the model-fit.
"""


"""
In this script, we will fit the warm pixel data converted to charge injection imaging in the script 
`line_collection_to_ci.py`, where:

 - The CTI model consists of two parallel `Trap` species.
 - The `CCD` volume fill parameterization is a simple form with just a `well_fill_power` parameter.
 - The `ImagingCI` is warm-pixel data from ACS.

"""


"""
Load the `ImagingCIs` from the .pickle file create in `line_collection_to_ci.py`, which is the dataset we will
use to perform CTI modeling.
"""


import autofit as af
import autocti as ac
import autocti.plot as aplt
from os import path
import pickle

dataset_name = "jc0a01h8q_imaging_cis"
dataset_path = path.join("dataset", "examples", "acs", "lines")

with open(path.join(dataset_path, dataset_name, ".pickle"), "rb") as f:
    imaging_list = pickle.load(f)


"""Lets plot the first `ImagingCI`"""
imaging_plotter = aplt.ImagingCIPlotter(imaging=imaging_list[0])
imaging_plotter.subplot_imaging()


"""
The `Clocker` models the CCD read-out, including CTI. 

For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
"""


clocker = ac.Clocker(parallel_express=2)


"""
__Phase__

To perform lens modeling, we create a *PhaseImagingCI* object, which comprises:

   - The `Trap`'s and `CCD` models used to fit the data.
   - The `SettingsPhase` which customize how the model is fitted to the data.
   - The `NonLinearSearch` used to sample parameter space.

Once we have create the phase, we 'run' it by passing it the data and mask.

__Model__

We compose our lens model using `Trap` and `CCD` objects, which are what add CTI to our images during clocking and 
read out. In this example our CTI model is:

 - Two parallel `TrapInstantCapture`'s which capture electrons during clocking instantly in the parallel direction.
 - A simple `CCD` volume beta parametrization.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.
"""


parallel_trap_0 = af.Model(ac.TrapInstantCapture)
parallel_trap_1 = af.Model(ac.TrapInstantCapture)
parallel_traps = [parallel_trap_0, parallel_trap_1]
parallel_ccd = af.Model(ac.CCD)
# parallel_ccd.well_notch_depth = 0.0
# parallel_ccd.full_well_depth = 84700


"""
__Settings__

Next, we specify the *SettingsPhaseImagingCI*, which describes how the model is fitted to the data in the 
log likelihood function. Below, we specify:

Different `SettingsPhase` are used in different example model scripts and a full description of all `SettingsPhase`
can be found in the example script 'autoccti_workspace/examples/model/customize/settings.py' and the following 
link -> <link>
"""


settings = ac.SettingsPhaseImagingCI(
    settings_ci_mask=ac.ci.SettingsMask2DCI(parallel_front_edge_rows=(0, 1))
)


"""
__Search__

The lens model is fitted to the data using a `NonLinearSearch`. In this example, we use the
nested sampling algorithm Dynesty (https://dynesty.readthedocs.io/en/latest/).

The script 'autocti_workspace/examples/model/customize/non_linear_searches.py' gives a description of the types of
non-linear searches that can be used with **PyAutoCTI**. If you do not know what a `NonLinearSearch` is or how it 
operates, checkout chapters 1 and 2 of the HowToCTI lecture series.
"""


search = af.DynestyStatic(
    path_prefix=path.join("acs", dataset_name), name="phase_parallel[x2]", nlive=50
)


"""
__Phase__

We can now combine the model, settings and `NonLinearSearch` above to create and run a phase, fitting our data with
the lens model.

The name and folders inputs below specify the path where results are stored in the output folder:  

 '/autolens_workspace/output/examples/beginner/mass_sie__source_bulge/phase__mass_sie__source_bulge'.
"""


phase = ac.PhaseImagingCI(
    parallel_traps=parallel_traps,
    parallel_ccd=parallel_ccd,
    settings=settings,
    search=search,
)


"""
We can now begin the fit by passing the dataset and mask to the phase, which will use the `NonLinearSearch` to fit
the model to the data. 

The fit outputs visualization on-the-fly, so checkout the path 
'/path/to/autolens_workspace/output/examples/phase__mass_sie__source_bulge' to see how your fit is doing!
"""


result = phase.run(dataset_list=imaging_list, clocker=clocker)


"""
The phase above returned a result, which, for example, includes the lens model corresponding to the maximum
log likelihood solution in parameter space.
"""


print(result.max_log_likelihood_instance)


"""
It also contains instances of the maximum log likelihood Tracer and FitImaging, which can be used to visualize
the fit.
"""


fit_ci_plotter = aplt.FitImagingCIPlotter(fit=result.max_log_likelihood_fit)
fit_ci_plotter.subplot_fit_imaging()


"""
Checkout '/path/to/autocti_workspace/examples/model/results.py' for a full description of the result object.
"""
