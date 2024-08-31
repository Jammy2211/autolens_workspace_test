import numpy as np

import autofit as af
import autolens as al
import autolens.plot as aplt

from autofit.non_linear.grid.sensitivity.result import SensitivityResult


def visualize_subhalo_detect(
    result_no_subhalo: af.Result,
    result: af.GridSearchResult,
    analysis,
    paths: af.DirectoryPaths,
):
    """
    Visualize the results of a subhalo detection grid search using the SLaM pipeline.

    This outputs the following visuals:

    - The `log_evidence` increases in each cell of the subhalo detection grid search, which is plotted over a lens
    subtracted  image of the dataset.

    - The subhalo `mass` inferred for every cell of the grid search, plotted over the lens subtracted image.

    - A subplot showing different aspects of the fit, so that the its with and without a subhalo can be compared.

    Parameters
    ----------
    result_no_subhalo
        The result of the model-fitting without a subhalo.
    result
        The grid search result of the subhalo detection model-fitting.
    analysis
        The analysis class used to perform the model fit.
    paths
        The paths object which defines the output path for the results of the subhalo detection grid search.
    """
    result = al.subhalo.SubhaloGridSearchResult(
        result=result,
    )

    fit_no_subhalo = result_no_subhalo.max_log_likelihood_fit

    fit_imaging_with_subhalo = analysis.fit_from(
        instance=result.best_samples.max_log_likelihood(),
    )

    output = aplt.Output(
        path=paths.output_path,
        format="png",
    )

    evidence_max = 541.0
    evidence_half = evidence_max / 2.0

    colorbar = aplt.Colorbar(
        manual_tick_values=[0.0, evidence_half, evidence_max],
        manual_tick_labels=[
            0.0,
            np.round(evidence_half, 1),
            np.round(evidence_max, 1),
        ],
    )
    colorbar_tickparams = aplt.ColorbarTickParams(labelsize=22, labelrotation=90)

    mat_plot = aplt.MatPlot2D(
        axis=aplt.Axis(extent=result.extent),
        #  colorbar=colorbar,
        #  colorbar_tickparams=colorbar_tickparams,
        output=output,
    )

    subhalo_plotter = al.subhalo.SubhaloPlotter(
        result=result,
        fit_imaging_no_subhalo=fit_no_subhalo,
        fit_imaging_with_subhalo=fit_imaging_with_subhalo,
        mat_plot_2d=mat_plot,
    )

    subhalo_plotter.figure_figures_of_merit_grid(
        use_log_evidences=True,
        relative_to_value=result_no_subhalo.samples.log_evidence,
        remove_zeros=True,
    )

    subhalo_plotter.figure_mass_grid()
    subhalo_plotter.subplot_detection_imaging()
    subhalo_plotter.subplot_detection_fits()


def visualize_sensitivity(
    result: SensitivityResult,
    paths: af.DirectoryPaths,
    mass_result: af.Result,
    mask: al.Mask2D,
):
    """
    Visualize the results of strong lens sensitivity mapping via the SLaM pipeline.

    This outputs the following visuals:

    - The `log_evidences_differences` and `log_likelihood_differences` of the sensitivity mapping,
    overlaid as a 2D grid of values over the lens subtracted image of the dataset.

    - The `log_evidences_differences` and `log_likelihood_differences` of the sensitivity mapping, as a 2D array
    not overlaid an image.

    Parameters
    ----------
    result
        The result of the sensitivity mapping, which contains grids of the log evidence and log likelihood differences.
    paths
        The paths object which defines the output path for the results of the sensitivity mapping.
    mass_result
        The result of the mass pipeline, which is used to subtract the lens light from the dataset.
    mask
        The mask used to mask the dataset, which is plotted over the lens subtracted image.
    """

    result = al.SubhaloSensitivityResult(
        result=result,
    )

    output = aplt.Output(
        path=paths.output_path,
        format="png",
    )

    data_subtracted = (
        mass_result.max_log_likelihood_fit.subtracted_images_of_planes_list[-1]
    )

    data_subtracted = data_subtracted.apply_mask(mask=mask)

    mat_plot_2d = aplt.MatPlot2D(axis=aplt.Axis(extent=result.extent), output=output)

    plotter = aplt.SubhaloSensitivityPlotter(
        result=result, data_subtracted=data_subtracted, mat_plot_2d=mat_plot_2d
    )

    plotter.subplot_figures_of_merit_grid()
    plotter.figure_figures_of_merit_grid()
