import autofit as af
import autolens as al

from . import slam_util

from typing import Union, Optional


def run(
    settings_search: af.SettingsSearch,
    analysis_list: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    source_result_for_lens: af.Result,
    source_result_for_source: af.Result,
    lens_bulge: Optional[af.Model] = af.Model(al.lp.Sersic),
    lens_disk: Optional[af.Model] = None,
    lens_point: Optional[af.Model] = None,
    extra_galaxies: Optional[af.Collection] = None,
    dataset_model: Optional[af.Model] = None,
) -> af.Result:
    """
    The SlaM LIGHT LP PIPELINE, which fits a complex model for a lens galaxy's light with the mass and source models
    fixed.

    Parameters
    ----------
    settings_search
        A collection of settings that control the behaviour of PyAutoFit thoughout the pipeline (e.g. paths, database,
        parallelization, etc.).
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    source_result_for_lens
        The result of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline, used
        for initializing model components associated with the lens galaxy.
    source_result_for_source
        The result of the SLaM SOURCE LP PIPELINE or SOURCE PIX PIPELINE which ran before this pipeline, used
        for initializing model components associated with the source galaxy.
    lens_bulge
        The model used to represent the light distribution of the lens galaxy's bulge (set to
        None to omit a bulge).
    lens_disk
        The model used to represent the light distribution of the lens galaxy's disk (set to
        None to omit a disk).
    lens_point
        The model used to represent the light distribution of the lens galaxy's point-source(s)
        emission (e.g. a nuclear star burst region) or compact central structures (e.g. an unresolved bulge).
    extra_galaxies
        Additional extra galaxies containing light and mass profiles, which model nearby line of sight galaxies.
    dataset_model
        Add aspects of the dataset to the model, for example the arc-second (y,x) offset between two datasets for
        multi-band fitting or the background sky level.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    Search 1 of the LIGHT LP PIPELINE fits a lens model where:

     - The lens galaxy light is modeled using a light profiles [no prior initialization].
     - The lens galaxy mass is modeled using SOURCE PIPELINE's mass distribution [Parameters fixed from SOURCE PIPELINE].
     - The source galaxy's light is modeled using SOURCE PIPELINE's model [Parameters fixed from SOURCE PIPELINE].

    This search aims to produce an accurate model of the lens galaxy's light, which may not have been possible in the
    SOURCE PIPELINE as the mass and source models were not properly initialized.
    """

    analysis_factor_list = []

    for i, analysis in enumerate(analysis_list):

        source = al.util.chaining.source_custom_model_from(
            result=source_result_for_source[i], source_is_model=False
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=source_result_for_lens[i].instance.galaxies.lens.redshift,
                    bulge=lens_bulge,
                    disk=lens_disk,
                    point=lens_point,
                    mass=source_result_for_lens[i].instance.galaxies.lens.mass,
                    shear=source_result_for_lens[i].instance.galaxies.lens.shear,
                ),
                source=source,
            ),
            extra_galaxies=extra_galaxies,
            dataset_model=dataset_model,
        )

        analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

        analysis_factor_list.append(analysis_factor)

    factor_graph = af.FactorGraphModel(*analysis_factor_list)

    search = af.DynestyStatic(
        name="light[1]",
        **settings_search.search_dict,
        nlive=150,
    )

    result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

    return result
