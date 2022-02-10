"""
Convergence
-----------

This script compares the analytic convergence calculation of a mass profile (e.g. `convergence_2d_from()`) to a
convergence computed via its deflection angle map (e.g. `convergence_2d_via_hessian_from()`).

This can verify that the analytic calculation and deflecition angle map calculations correspond to the same mass
profile. If this is not the case, it indicates a bug in the `convergence_2d_from()` method of the mass profile.
"""
import autolens as al
import autolens.plot as aplt

"""
__Masking__

Define the mask where the convergence is evaluated in.
"""
mask = al.Mask2D.circular_annular(
    shape_native=(100, 100),
    pixel_scales=0.1,
    inner_radius=0.5,
    outer_radius=3.0,
)

"""
__Grids__

Compute the 2D grid from this mask which is the (y,x) coordinates the convergence is evaluated on.
"""
grid = al.Grid2D.from_mask(mask=mask)

"""
__Mass Profile__

Set up the mass profile whose analytic converge we are testing.
"""
# mass = al.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

mass = al.mp.EllPowerLaw(
    centre=(0.0, 0.0),
    elliptical_comps=(-0.5, 0.5),
    einstein_radius=2.0,
    slope=2.0
)

# mass = al.mp.EllIsothermal(
#     centre=(0.0, 0.0),
#     elliptical_comps=(-0.5, 0.5),
#     einstein_radius=2.0,
# )

# mass = al.mp.EllNFWMCRScatterLudlow(
#     centre=(0.0, 0.0),
#     elliptical_comps=(0.5, 0.5),
#     mass_at_200=5e13,
#     scatter_sigma=2.0,
#     redshift_object=0.5,
#     redshift_source=1.0
# )

"""
__Convergence__

Compute the convergence via both methods.
"""
convergence_analytic = mass.convergence_2d_from(grid=grid)
convergence_hessian = mass.convergence_2d_via_hessian_from(grid=grid)
# Have to convert the `ValuesIrregular` output of the hessian function to an Array2D.

# convergence_hessian = mass.convergence_2d_via_jacobian_from(grid=grid)
convergence_hessian = al.Array2D.manual_mask(array=convergence_hessian.slim, mask=mask)

"""
__Residuals__

The residuals of the two convergence profiles, which will inform us of whether the calculations agree.
"""
convergence_residual_map = convergence_analytic - convergence_hessian

"""
__Plot__

Plot each convergence profile and their residuals.
"""
mat_plot_2d = aplt.MatPlot2D(title=aplt.Title(fontsize=24))

plotter_analytic = aplt.Array2DPlotter(
    array=convergence_analytic, mat_plot_2d=mat_plot_2d
)
plotter_hessian = aplt.Array2DPlotter(array=convergence_hessian)
plotter_residual_map = aplt.Array2DPlotter(array=convergence_residual_map)

plotter_hessian.mat_plot_2d = plotter_analytic.mat_plot_2d
plotter_residual_map.mat_plot_2d = plotter_analytic.mat_plot_2d

plotter_analytic.open_subplot_figure(
    number_subplots=3, subplot_shape=(1, 3), subplot_figsize=(20, 7)
)

plotter_analytic.set_title("Convergence via Analytic")
plotter_analytic.figure_2d()

plotter_hessian.set_title("Convergence via Hessian")
plotter_hessian.figure_2d()

plotter_residual_map.set_title("Convergence Residuals")
plotter_residual_map.figure_2d()

plotter_analytic.mat_plot_2d.output.subplot_to_figure()
