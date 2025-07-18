import autofit as af
import autolens as al
import autolens.plot as aplt

from . import subhalo_util

import numpy as np
import os
from typing import Optional, List, Tuple, Union

"""
__Simulate Function Class__

We now write the `simulate_cls`, which takes the `simulation_instance` of our model (defined above) and uses it to 
simulate a strong lens dataset, which include a dark matter subhalo, which is subsequently fitted.

Additional attributes required to simulate the data (mask, PSF) can be passed to the `__init__` method, and the 
simulation is  performed in the `__call__` method.

When this dataset is simulated, the quantity `instance.perturb` is used in `__call__`. This is an instance 
of the `NFWMCRLudlowSph`, and it is different every time the `simulate_cls` is called based on the value of sensitivity 
being computed. 

In this example, this `instance.perturb` corresponds to two different subhalos with values of `mass_at_200` of 
1e6 MSun and 1e13 MSun.
"""


class SimulateImaging:
    def __init__(self, mask, psf, add_poisson_noise_to_data: bool):
        """
        Class used to simulate the strong lens imaging used for sensitivity mapping.

        Parameters
        ----------
        mask
            The mask applied to the real image data, which is applied to every simulated imaging.
        psf
           The PSF of the real image data, which is applied to every simulated imaging and used for each fit.
        """
        self.mask = mask
        self.psf = psf
        self.add_poisson_noise_to_data = add_poisson_noise_to_data

    def __call__(self, instance: af.ModelInstance, simulate_path: str):
        """
        The `simulate_function` called by the `Sensitivity` class which simulates each strong lens image fitted
        by the sensitivity mapper.

        The simulation procedure is as follows:

        1) Use the input galaxies of the sensitivity `instance` to set up a tracer, which generates the image-plane
           image of the strong lens system.

        2) Simulate this image using the input dataset noise (Poisson) and PSF.

        3) Apply the mask used in the analysis of the real image to the simulated image.

        4) Output information about the simulation to hard-disk.

        The `subhalo` in the sensitivity `instance` changes for every iteration of the sensitivity mapping, ensuring
        that we map out the sensitivity of the analysis to the subhalo properties (centre, mass, etc.).

        Parameters
        ----------
        instance
            The sensitivity instance, which includes the galaxies whose parameters are varied to perform sensitivity.
            The subhalo in this instance changes for every iteration of the sensitivity mapping.
        simulate_path
            The path where the simulated dataset is output, contained within each sub-folder of the sensitivity
            mapping.

        Returns
        -------
        A simulated image of a strong lens, which id input into the fits of the sensitivity mapper.
        """

        """
        __Resume Fit__

        If sensitivity mapping already began on this grid cell, the dataset will have been simulated already and we
        do not want to resimulate it and change its noise properties. 

        We therefore load it from the `simulate_path` instead.
        """
        try:

            dataset = al.Imaging.from_fits(
                data_path=f"{simulate_path}/data.fits",
                psf_path=f"{simulate_path}/psf.fits",
                noise_map_path=f"{simulate_path}/noise_map.fits",
                pixel_scales=self.mask.pixel_scales,
                check_noise_map=False,
            )

            dataset = dataset.apply_mask(mask=self.mask)

            tracer = al.Tracer(
                galaxies=[
                    instance.galaxies.lens,
                    instance.perturb,
                    instance.galaxies.source,
                ]
            )

            traced_grid = tracer.traced_grid_2d_list_from(
                grid=dataset.grid,
            )[-1]

            source_centre = instance.galaxies.source.bulge.centre

            over_sample_size = (
                al.util.over_sample.over_sample_size_via_radial_bins_from(
                    grid=traced_grid,
                    sub_size_list=[16, 8, 2],
                    radial_list=[0.1, 0.3],
                    centre_list=[source_centre],
                )
            )

            over_sample_size_lens = (
                al.util.over_sample.over_sample_size_via_radial_bins_from(
                    grid=dataset.grid,
                    sub_size_list=[8, 4, 1],
                    radial_list=[0.3, 0.6],
                    centre_list=[(0.0, 0.0)],
                )
            )

            over_sample_size = np.where(
                over_sample_size > over_sample_size_lens,
                over_sample_size,
                over_sample_size_lens,
            )
            over_sample_size = al.Array2D(values=over_sample_size, mask=self.mask)

            return dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        except FileNotFoundError:
            pass

        """
        __Ray Tracing__

        Set up the `Tracer` which is used to simulate the strong lens imaging, which may include the subhalo in
        addition to the lens and source galaxy.
        """
        tracer = al.Tracer(
            galaxies=[
                instance.galaxies.lens,
                instance.perturb,
                instance.galaxies.source,
            ]
        )

        """
        __Simulate__

        Set up the grid, PSF and simulator settings used to simulate imaging of the strong lens. These should be 
        tuned to match the S/N and noise properties of the observed data you are performing sensitivity mapping on.
        """
        grid = al.Grid2D.uniform(
            shape_native=self.mask.shape_native,
            pixel_scales=self.mask.pixel_scales,
        )

        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=grid,
            sub_size_list=[32, 8, 2],
            radial_list=[0.3, 0.6],
            centre_list=[(0.0, 0.0)],
        )

        grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

        simulator = al.SimulatorImaging(
            exposure_time=1000.0,
            psf=self.psf,
            background_sky_level=0.1,
            add_poisson_noise_to_data=self.add_poisson_noise_to_data,
            noise_seed=1,
        )

        dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

        """
        __Masking__

        The data generated by the simulate function is what is ultimately fitted.

        Therefore, we also apply the mask for the analysis before we return the simulated data.
        """
        dataset = dataset.apply_mask(mask=self.mask)

        traced_grid = tracer.traced_grid_2d_list_from(
            grid=dataset.grid,
        )[-1]

        source_centre = instance.galaxies.source.bulge.centre

        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=traced_grid,
            sub_size_list=[16, 8, 2],
            radial_list=[0.1, 0.3],
            centre_list=[source_centre],
        )

        over_sample_size_lens = (
            al.util.over_sample.over_sample_size_via_radial_bins_from(
                grid=dataset.grid,
                sub_size_list=[8, 4, 1],
                radial_list=[0.3, 0.6],
                centre_list=[(0.0, 0.0)],
            )
        )

        over_sample_size = np.where(
            over_sample_size > over_sample_size_lens,
            over_sample_size,
            over_sample_size_lens,
        )
        over_sample_size = al.Array2D(values=over_sample_size, mask=self.mask)

        dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        """
        Outputs info about the `Tracer` to the fit, so we know exactly how we simulated the image.
        """
        self.output_info(simulate_path=simulate_path, dataset=dataset, tracer=tracer)

        return dataset

    def output_info(self, simulate_path: str, dataset: al.Imaging, tracer: al.Imaging):
        """
        Output information about the data simulated for this iteration of sensitivity mapping.

        This information output is as follows:

        - A subplot of the simulated imaging dataset.
        - A subplot of the tracer used to simulate this imaging dataset.
        - A .json file containing the tracer galaxies.
        - Output the simulated dataset to .fits files which are used to load the data if a run is resumed.

        Parameters
        ----------
        simulate_path
            The path where the simulated dataset is output, contained within each sub-folder of the sensitivity
            mapping.
        dataset
            The simulated imaging dataset which is visualized.
        tracer
            The tracer used to simulate the imaging dataset, which is visualized and output to a .json file.
        """

        mat_plot = aplt.MatPlot2D(output=aplt.Output(path=simulate_path, format="png"))

        dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
        dataset_plotter.subplot_dataset()

        tracer_plotter = aplt.TracerPlotter(
            tracer=tracer, grid=dataset.grid, mat_plot_2d=mat_plot
        )
        tracer_plotter.subplot_lensed_images()

        al.output_to_json(
            obj=tracer,
            file_path=os.path.join(simulate_path, "tracer.json"),
        )

        dataset.output_to_fits(
            data_path=os.path.join(simulate_path, "data.fits"),
            psf_path=os.path.join(simulate_path, "psf.fits"),
            noise_map_path=os.path.join(simulate_path, "noise_map.fits"),
            overwrite=True,
        )


"""
__Base Fit__

We have defined a `Simulate` class that will be used to simulate every dataset simulated by the sensitivity mapper.
Each simulated dataset will have a unique set of parameters for the `subhalo` (e.g. due to different values of
`perturb_model`.

We will fit each simulated dataset using the `base_model`, which quantifies whether not including the dark matter
subhalo in the model changess the goodness-of-fit and therefore indicates if we are sensitive to the subhalo.

We now write a `BaseFit` class, defining how the `base_model` is fitted to each simulated dataset and the 
goodness-of-fit used to quantify whether the model fits the data well. As above, the `__init__` method can be
extended with new inputs to control how the model is fitted and the `__call__` method performs the fit.

In this example, we use a full non-linear search to fit the `base_model` to the simulated data and return
the `log_evidence` of the model fit as the goodness-of-fit. This fit could easily be something much simpler and
more computationally efficient, for example performing a single log likelihood evaluation of the `base_model` fit
to the simulated data.

Fucntionality which adapts the mesh and regularization of a pixelized source reconstruction to the unlensed source's 
morphology require an `adapt_images`. This is an input of the __init__ constructor which is passed to the `Analysis` 
for every simulated dataset.
"""


class BaseFit:
    def __init__(self, adapt_images, number_of_cores: int = 1):
        """
        Class used to fit every dataset used for sensitivity mapping with the base model (the model without the
        perturbed feature sensitivity mapping maps out).

        In this example, the base model therefore does not include the dark matter subhalo, but the simulated
        dataset includes one.

        The base fit is repeated for every parameter on the sensitivity grid and compared to the perturbed fit. This
        maps out the sensitivity of every parameter is (e.g. the sensitivity of the mass of the subhalo).

        The `__init__` constructor can be extended with new inputs which can be used to control how the dataset is
        fitted, below we include an input `analysis_cls` which is the `AnalysisImaging` class used to fit the model
        to the dataset.

        Parameters
        ----------
        adapt_images
            Contains the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
            reconstructed galaxy's morphology.
        number_of_cores
            The number of cores used to perform the non-linear search. If 1, each model-fit on the grid is performed
            in serial, if > 1 fits are distributed in parallel using the Python multiprocessing module.
        """
        self.adapt_images = adapt_images
        self.number_of_cores = number_of_cores

    def __call__(self, dataset, model, paths, instance):
        """
        The base fitting function which fits every dataset used for sensitivity mapping with the base model.

        This function receives as input each simulated dataset of the sensitivity map and fits it, in order to
        quantify how sensitive the model is to the perturbed feature.

        In this example, a full non-linear search is performed to determine how well the model fits the dataset.
        The `log_evidence` of the fit is returned which acts as the sensitivity map figure of merit.

        Parameters
        ----------
        dataset
            The dataset which is simulated with the perturbed model and which is fitted.
        model
            The model instance which is fitted to the dataset, which does not include the perturbed feature.
        paths
            The `Paths` instance which contains the path to the folder where the results of the fit are written to.
        instance
            The simulation instance, which includes the perturbed feature that is used to simulate the dataset.
            This is often not used, but may be useful for certain sensitivity mapping tasks, for example using
            true values of the simulated instance to set up aspects of the model-fit (e.g. the priors).
        """

        search = af.Nautilus(
            paths=paths,
            n_live=50,
            number_of_cores=self.number_of_cores,
        )

        analysis = al.AnalysisImaging(dataset=dataset)
        analysis._adapt_images = self.adapt_images

        return search.fit(model=model, analysis=analysis)


"""
__Perturb Fit__

We now define a `PerturbFit` class, which defines how the `perturb_model` is fitted to each simulated dataset. This
behaves analogously to the `BaseFit` class above, but now fits the `perturb_model` to the simulated data (as
opposed to the `base_model`).

Again, in this example we use a full non-linear search to fit the `perturb_model` to the simulated data and return
the `log_evidence` of the model fit as the goodness-of-fit. This fit could easily be something much simpler and
more computationally efficient, for example performing a single log likelihood evaluation of the `perturb_model` fit
to the simulated data.
"""


class PerturbFit:
    def __init__(self, adapt_images, fast_perturb_fit: bool, number_of_cores: int = 1):
        """
        Class used to fit every dataset used for sensitivity mapping with the perturbed model (the model with the
        perturbed feature sensitivity mapping maps out).

        In this example, the perturbed model therefore includes the dark matter subhalo, which is also in the
        simulated dataset.

        The perturbed fit is repeated for every parameter on the sensitivity grid and compared to the base fit. This
        maps out the sensitivity of every parameter is (e.g. the sensitivity of mass of the subhalo).

        The `__init__` constructor can be extended with new inputs which can be used to control how the dataset is
        fitted, below we include an input `analysis_cls` which is the `Analysis` class used to fit the model to the
        dataset.

        Parameters
        ----------
        adapt_images
            Contains the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
            reconstructed galaxy's morphology.
        fast_perturb_fit
            If True, a fast perturb fit is performed which performs a single iteration using the input model that
            simulated and fits the data.
        number_of_cores
            The number of cores used to perform the non-linear search. If 1, each model-fit on the grid is performed
            in serial, if > 1 fits are distributed in parallel using the Python multiprocessing module.
        """
        self.adapt_images = adapt_images
        self.fast_perturb_fit = fast_perturb_fit
        self.number_of_cores = number_of_cores

    def __call__(self, dataset, model, paths, instance):
        """
        The perturbed fitting function which fits every dataset used for sensitivity mapping with the perturbed model.

        This function receives as input each simulated dataset of the sensitivity map and fits it, in order to
        quantify how sensitive the model is to the perturbed feature.

        In this example, a full non-linear search is performed to determine how well the model fits the dataset.
        The `log_evidence` of the fit is returned which acts as the sensitivity map figure of merit.

        Parameters
        ----------
        dataset
            The dataset which is simulated with the perturbed model and which is fitted.
        model
            The model instance which is fitted to the dataset, which includes the perturbed feature.
        paths
            The `Paths` instance which contains the path to the folder where the results of the fit are written to.
        instance
            The simulation instance, which includes the perturbed feature that is used to simulate the dataset.
            This is often not used, but may be useful for certain sensitivity mapping tasks, for example using
            true values of the simulated instance to set up aspects of the model-fit (e.g. the priors).
        """

        analysis = al.AnalysisImaging(dataset=dataset)
        analysis._adapt_images = self.adapt_images

        if self.fast_perturb_fit:

            initializer = af.InitializerParamStartPoints(
                {
                    model.galaxies.source.bulge.centre.centre_0: instance.galaxies.source.bulge.centre[
                        0
                    ],
                    model.galaxies.source.bulge.centre.centre_1: instance.galaxies.source.bulge.centre[
                        1
                    ],
                    model.galaxies.source.bulge.ell_comps.ell_comps_0: instance.galaxies.source.bulge.ell_comps[
                        0
                    ],
                    model.galaxies.source.bulge.ell_comps.ell_comps_1: instance.galaxies.source.bulge.ell_comps[
                        1
                    ],
                    model.galaxies.source.bulge.effective_radius: instance.galaxies.source.bulge.effective_radius,
                    model.galaxies.source.bulge.sersic_index: instance.galaxies.source.bulge.sersic_index,
                    model.galaxies.lens.mass.centre.centre_0: instance.galaxies.lens.mass.centre[
                        0
                    ],
                    model.galaxies.lens.mass.centre.centre_1: instance.galaxies.lens.mass.centre[
                        1
                    ],
                    model.galaxies.lens.mass.ell_comps.ell_comps_0: instance.galaxies.lens.mass.ell_comps[
                        0
                    ],
                    model.galaxies.lens.mass.ell_comps.ell_comps_1: instance.galaxies.lens.mass.ell_comps[
                        1
                    ],
                    model.galaxies.lens.mass.einstein_radius: instance.galaxies.lens.mass.einstein_radius,
                    model.galaxies.lens.mass.slope: instance.galaxies.lens.mass.slope,
                    model.galaxies.lens.shear.gamma_1: instance.galaxies.lens.shear.gamma_1,
                    model.galaxies.lens.shear.gamma_2: instance.galaxies.lens.shear.gamma_2,
                    model.perturb.mass.centre.centre_0: instance.perturb.mass.centre[0],
                    model.perturb.mass.centre.centre_1: instance.perturb.mass.centre[1],
                    #                model.perturb.mass.mass_at_200: instance.perturb.mass.mass_at_200,
                    model.perturb.mass.log10m_vir: instance.perturb.mass.log10m_vir,
                    model.perturb.mass.c_gNFW: instance.perturb.mass.c_gNFW,
                    model.perturb.mass.overdens: instance.perturb.mass.overdens,
                    model.perturb.mass.inner_slope: instance.perturb.mass.inner_slope,
                }
            )

            search = af.Drawer(
                paths=paths,
                total_draws=1,
                initializer=initializer,
            )

        else:

            search = af.Nautilus(
                paths=paths,
                n_live=50,
                number_of_cores=self.number_of_cores,
            )

        return search.fit(model=model, analysis=analysis)


def base_model_narrow_priors_from(base_model, result, stretch: float = 1.0):
    """
    Returns a base model where priors are updated to `UniformPriors` with a `lower_limit` and `upper_limit` which
    are a narrow range over the `simulation_instance` parameter values.

    Using this updated model can significiantly speed up sensitivity mapping, as it dramatically reduces
    the volume of parameter space that needs to be sampled.

    The downside of this approach is that these narrow priors may remove viable solutions that alter the sensitivity
    or lead to an inaccurate estimate of the Bayesian evidence.

    The size of each parameter bounds have been chosen based on previous lens modeling intuition. For example, I have
    never seen a DM subhalo change the centre of a lens mass model by more than 0.01", therefore this is the value
    used for bounding that parameter.

    Parameters
    ----------
    base_model
        The base model which will be used for sensitivity mapping, which this function updates to have narrower priors.
    result
        The result used to set up the base model and which is used to set these updated priors.
    stretch
        A multiplicative factor which can be used to shrink or broaden the priors more.

    Returns
    -------
    A base model with priors updated to narrow uniform priors.
    """

    if hasattr(base_model.galaxies.lens, "mass"):
        base_model.galaxies.lens.mass.centre.centre_0 = result.model_centred_max_lh_bounded(
            b=0.01 * stretch
        ).galaxies.lens.mass.centre.centre_0
        base_model.galaxies.lens.mass.centre.centre_1 = result.model_centred_max_lh_bounded(
            b=0.01 * stretch
        ).galaxies.lens.mass.centre.centre_1
        base_model.galaxies.lens.mass.ell_comps.ell_comps_0 = result.model_centred_max_lh_bounded(
            b=0.05 * stretch
        ).galaxies.lens.mass.ell_comps.ell_comps_0
        base_model.galaxies.lens.mass.ell_comps.ell_comps_1 = result.model_centred_max_lh_bounded(
            b=0.05 * stretch
        ).galaxies.lens.mass.ell_comps.ell_comps_1
        base_model.galaxies.lens.mass.einstein_radius = result.model_centred_max_lh_bounded(
            b=0.1 * stretch
        ).galaxies.lens.mass.einstein_radius
        base_model.galaxies.lens.mass.slope = result.model_centred_max_lh_bounded(
            b=0.1 * stretch
        ).galaxies.lens.mass.slope

    if hasattr(base_model.galaxies.lens, "shear"):
        base_model.galaxies.lens.shear.gamma_1 = result.model_centred_max_lh_bounded(
            b=0.05 * stretch
        ).galaxies.lens.shear.gamma_1
        base_model.galaxies.lens.shear.gamma_2 = result.model_centred_max_lh_bounded(
            b=0.05 * stretch
        ).galaxies.lens.shear.gamma_2

    if hasattr(base_model.galaxies.source, "bulge"):
        base_model.galaxies.source.bulge.centre.centre_0 = result.model_centred_max_lh_bounded(
            b=0.1 * stretch
        ).galaxies.source.bulge.centre.centre_0
        base_model.galaxies.source.bulge.centre.centre_1 = result.model_centred_max_lh_bounded(
            b=0.1 * stretch
        ).galaxies.source.bulge.centre.centre_1
        base_model.galaxies.source.bulge.ell_comps.ell_comps_0 = result.model_centred_max_lh_bounded(
            b=0.1 * stretch
        ).galaxies.source.bulge.ell_comps.ell_comps_0
        base_model.galaxies.source.bulge.ell_comps.ell_comps_1 = result.model_centred_max_lh_bounded(
            b=0.1 * stretch
        ).galaxies.source.bulge.ell_comps.ell_comps_1
        base_model.galaxies.source.bulge.effective_radius = result.model_centred_max_lh_bounded(
            b=0.2 * stretch
        ).galaxies.source.bulge.effective_radius
        base_model.galaxies.source.bulge.sersic_index = result.model_centred_max_lh_bounded(
            b=0.2 * stretch
        ).galaxies.source.bulge.sersic_index

        if base_model.galaxies.source.bulge.effective_radius.lower_limit < 0.0:
            base_model.galaxies.source.bulge.effective_radius = af.UniformPrior(
                lower_limit=0.0,
                upper_limit=base_model.galaxies.source.bulge.effective_radius.upper_limit,
            )

        if base_model.galaxies.source.bulge.sersic_index.lower_limit < 0.0:
            base_model.galaxies.source.bulge.sersic_index = af.UniformPrior(
                lower_limit=0.0,
                upper_limit=base_model.galaxies.source.bulge.sersic_index.upper_limit,
            )

    return base_model


def run(
    settings_search: af.SettingsSearch,
    mask: al.Mask2D,
    psf: al.Kernel2D,
    mass_result: af.Result,
    subhalo_mass: af.Model = af.Model(al.mp.NFWMCRLudlowSph),
    adapt_images: Optional[al.AdaptImages] = None,
    grid_dimension_arcsec: float = 3.0,
    add_poisson_noise_to_data: bool = False,
    fast_perturb_fit: bool = True,
    number_of_steps: Union[Tuple[int], int] = 5,
    batch_range: Tuple[int, int] = None,
    sensitivity_mask: Optional[Union[al.Mask2D, List]] = None,
):
    """
    The SLaM SUBHALO PIPELINE for performing sensitivity mapping, which determines what mass dark matter subhalos
    can be detected where in the dataset.

    Parameters
    ----------
    mask
        The Mask2D that is applied to the imaging data for model-fitting.
    psf
        The Point Spread Function (PSF) used when simulating every image of the strong lens that is fitted by
        sensitivity mapping.
    mass_result
        The result of the SLaM MASS PIPELINE which ran before this pipeline.
    subhalo_mass
        The `MassProfile` used to fit the subhalo in this pipeline.
    grid_dimension_arcsec
        the arc-second dimensions of the grid in the y and x directions. An input value of 3.0" means the grid in
        all four directions extends to 3.0" giving it dimensions 6.0" x 6.0".
    add_poisson_noise_to_data
        Whether to simulate the sensitivity mapping data with noise, which by default is not included as one can still
        quantify the sensitivity without noise, which reduces poison fluctuations in the analysis.
    fast_perturb_fit
        If True, a fast perturb fit is performed which performs a single iteration using the input model that
        simulated and fits the data.
    number_of_steps
        The 2D dimensions of the grid (e.g. number_of_steps x number_of_steps) that the subhalo search is performed for.
    number_of_cores
        The number of cores used to perform the non-linear search grid search. If 1, each model-fit on the grid is
        performed in serial, if > 1 fits are distributed in parallel using the Python multiprocessing module.
    """

    """
    __Base Model__

    We now define the `base_model` that we use to perform sensitivity mapping. This is the lens model that is fitted to 
    every simulated strong lens without a subhalo, giving us the Bayesian evidence which we compare to the model which 
    includes one!). 

    For this model, we can use the `base_model` above, however we will use the result of fitting this model to the dataset
    before sensitivity mapping. This ensures the priors associated with each parameter are initialized so as to speed up
    each non-linear search performed during sensitivity mapping.
    """
    base_model = mass_result.model

    base_model = base_model_narrow_priors_from(
        base_model=base_model, result=mass_result
    )

    """
    __Perturb Model__

    We now define the `perturb_model`, which is the model component whose parameters we iterate over to perform 
    sensitivity mapping. In this case, this model is a `NFWMCRLudlowSph` model and we will iterate over its
    `centre` and `mass_at_200`. We set it up as a `Model` so it has an associated redshift and can be directly
    passed to the tracer in the simulate function below.

    Many instances of the `perturb_model` are created and used to simulate the many strong lens datasets that we fit. 
    However, it is only included in half of the model-fits; corresponding to the lens models which include a dark matter 
    subhalo and whose Bayesian evidence we compare to the simpler model-fits consisting of just the `base_model` to 
    determine if the subhalo was detectable.

    By fitting both models to every simulated lens, we therefore infer the Bayesian evidence of every model to every 
    dataset. Sensitivity mapping therefore maps out for what values of `centre` and `mass_at_200` in the dark matter 
    subhalo the model-fit including a subhalo provide higher values of Bayesian evidence than the simpler model-fit (and
    therefore when it is detectable!).
    """
    perturb_model = af.Model(al.Galaxy, redshift=0.5, mass=subhalo_mass)

    """
    __Mapping Perturb Grid__

    Sensitivity mapping is typically performed over a large range of parameters on a grid.

    We will perform sensitivity mapping over a 2D grid of (y,x) values, where each lens model-fit is performed on a
    different (y,x) coordinate on the grid. The size and shape of the grid is set by the input `grid_dimension_arcsec`.
    """
    perturb_model.mass.log10m_vir = 9.0
    perturb_model.mass.c_gNFW = 12.0
    perturb_model.mass.overdens = 200.0
    perturb_model.mass.inner_slope = 2.2

    perturb_model.mass.centre.centre_0 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    perturb_model.mass.centre.centre_1 = af.UniformPrior(
        lower_limit=-grid_dimension_arcsec, upper_limit=grid_dimension_arcsec
    )
    perturb_model.mass.redshift_object = mass_result.model.galaxies.lens.redshift
    perturb_model.mass.redshift_source = mass_result.model.galaxies.source.redshift

    """
    __Perturb Model Prior Func__

    The default priors on the `perturb_model` are `UniformPrior`'s bounded around each sensitivity grid cell.

    For example, the first simulated dark matter subhalo is at location (1.5, -1.5) and its priors are:  

    - y is `UniformPrior(lower_limit=0.0, upper_limit=3.0)`.
    - x is `UniformPrior(lower_limit=-3.0, upper_limit=0.0)`.

    The `mass_at_200` is fixed to a value of 1e10 in the `perturb_model` above, which is the fixed value used by
    the model fit.

    By passing a `perturb_model_prior_func` to the sensitivity mapper, we can manually overwrite the priors on 
    the `perturb_model` which are used instead for the fit.

    Below, we update the priors as follows:

    - The y and x priors are trimmed to much narrower bounded priors, confined to regions 0.05" each side of the 
      true DM subhalo.

    - The `mass_at_200` is made a free parameter with `LogUniformPrior(lower_limit=1e6, 1e12)`. This is a large range, 
      but ensures there are solutions where the DM subhalo can go to lower masses.    
    """

    def perturb_model_prior_func(perturb_instance, perturb_model):
        b = 0.05

        perturb_model.mass.centre.centre_0 = af.UniformPrior(
            lower_limit=perturb_instance.mass.centre[0] - b,
            upper_limit=perturb_instance.mass.centre[0] + b,
        )

        perturb_model.mass.centre.centre_1 = af.UniformPrior(
            lower_limit=perturb_instance.mass.centre[1] - b,
            upper_limit=perturb_instance.mass.centre[1] + b,
        )

        perturb_model.mass.log10m_vir = af.UniformPrior(lower_limit=6, upper_limit=12)

        return perturb_model

    """
    We are performing sensitivity mapping to determine where a subhalo is detectable. This will require us to 
    simulate many realizations of our dataset with a lens model, called the `simulation_instance`. This model uses the
    result of the fit above.

    The code below ensures that the lens light, mass and source parameters of the strong lens are used when simulating
    each dataset with a dark matter subhalo.
    """
    simulation_instance = mass_result.instance

    fit = mass_result.max_log_likelihood_fit

    simulation_instance.galaxies.lens = (
        fit.model_obj_linear_light_profiles_to_light_profiles.galaxies[0]
    )
    simulation_instance.galaxies.source = (
        fit.model_obj_linear_light_profiles_to_light_profiles.galaxies[-1]
    )

    """
    __Simulation + Fits__

    We set up the `simulate_cls` which defines how the mock dataset is simulated that is fitted. The `SimulationImaging`
    objected used to do this is defined at the top of the script.

    Above are also the `BaseFit` and `PerturbFit` classes, which define how for each step of the sensitivity mapper a
    model-fit is performed on the simulated dataset to determine the sensitivity of the model to the subhalo.

    These are described in full in the docstrings above each class and these should be referred to for a complete
    description of the inputs of each class.
    """

    """
    We can now combine all of the objects created above and perform sensitivity mapping. The inputs to the `Sensitivity`
    object below are:

    - `simulation_instance`: This is an instance of the model used to simulate every dataset that is fitted. In this example 
    it is a lens model that does not include a subhalo, which was inferred by fitting the dataset we perform sensitivity 
    mapping on.

    - `base_model`: This is the lens model that is fitted to every simulated dataset, which does not include a subhalo. In 
    this example is composed of an `Isothermal` lens and `Sersic` source.

    - `perturb_model`: This is the extra model component that alongside the `base_model` is fitted to every simulated 
    dataset. In this example it is a `NFWMCRLudlowSph` dark matter subhalo.

    - `simulate_cls`: This is the function that uses the `simulation_instance` and many instances of the `perturb_model` 
    to simulate many datasets that are fitted with the `base_model` and `base_model` + `perturb_model`.

    - `base_fit_cls`: This is the function that fits the `base_model` to every simulated dataset and returns the
    goodness-of-fit of the model to the data.

    - `perturb_fit_cls`: This is the function that fits the `base_model` + `perturb_model` to every simulated dataset and
    returns the goodness-of-fit of the model to the data.

    - `number_of_steps`: The number of steps over which the parameters in the `perturb_model` are iterated. In this 
    example, `mass_at_200` has a `LogUniformPrior` with lower limit 1e6 and upper limit 1e13, therefore 
    the `number_of_steps` of 2 will simulate and fit just 2 datasets where the `mass_at_200` is between 1e6 and 1e13.

    - `number_of_cores`: The number of cores over which the sensitivity mapping is performed, enabling parallel processing
    if set above 1.
    """

    paths = af.DirectoryPaths(
        name=f"subhalo__sensitivity",
        path_prefix=settings_search.path_prefix,
        unique_tag=settings_search.unique_tag,
    )

    subhalo_util.visualize_sensitivity_mask(
        mass_result=mass_result, sensitivity_mask=sensitivity_mask, paths=paths
    )

    sensitivity = af.Sensitivity(
        paths=paths,
        simulation_instance=simulation_instance,
        base_model=base_model,
        perturb_model=perturb_model,
        simulate_cls=SimulateImaging(
            mask=mask, psf=psf, add_poisson_noise_to_data=add_poisson_noise_to_data
        ),
        base_fit_cls=BaseFit(
            adapt_images=adapt_images, number_of_cores=settings_search.number_of_cores
        ),
        perturb_fit_cls=PerturbFit(
            adapt_images=adapt_images,
            fast_perturb_fit=fast_perturb_fit,
            number_of_cores=settings_search.number_of_cores,
        ),
        perturb_model_prior_func=perturb_model_prior_func,
        visualizer_cls=subhalo_util.Visualizer(mass_result=mass_result, mask=mask),
        number_of_steps=number_of_steps,
        batch_range=batch_range,
        mask=sensitivity_mask,
        number_of_cores=1,
    )

    result = sensitivity.run()

    return result
