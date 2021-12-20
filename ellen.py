import autolens as al
import autolens.plot as aplt

"""
We compute lensing quantities such as convergence and shear using a 2D grid of (y,x) coordinates.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

"""
Set up an `EllNFW` mass profile which we will use to compute its lensing derived quantities.
"""
nfw = al.mp.EllNFW(
    centre=(0.0, 0.0),
    elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=45.0),
    kappa_s=0.1,
    scale_radius=30.0,
)

"""
The function `convergence_2d_from` gives us the convergence, computed analytically.
"""
convergence = nfw.convergence_2d_from(grid=grid)
print(convergence)

"""
To plot this quantity we can use a `MassProfilePlotter`.
"""
plotter = aplt.MassProfilePlotter(mass_profile=nfw, grid=grid)
plotter.figures_2d(convergence=True)

"""
The shear of a mass profile is not computed analytically. 

It is derived using the deflection angle map, via the lensing jacobian and it is returned as a vector field:
"""
shear_yx = nfw.shear_yx_2d_via_jacobian_from(grid=grid)
shear_y = shear_yx[1]
shear_x = shear_yx[0]

shear_y = al.Array2D.manual_slim(
    array=shear_y.native, shape_native=grid.shape_native, pixel_scales=grid.pixel_scales
)
plotter = aplt.Array2DPlotter(array=shear_y)
plotter.figure_2d()

shear_x = al.Array2D.manual_slim(
    array=shear_x.native, shape_native=grid.shape_native, pixel_scales=grid.pixel_scales
)
plotter = aplt.Array2DPlotter(array=shear_x)
plotter.figure_2d()

"""
Alternative it can be computed via the hessian:
"""
shear_yx = nfw.shear_yx_2d_via_hessian_from(grid=grid)
shear_y = al.Array2D.manual_slim(
    array=shear_yx[0], shape_native=grid.shape_native, pixel_scales=grid.pixel_scales
)
shear_x = al.Array2D.manual_slim(
    array=shear_yx[1], shape_native=grid.shape_native, pixel_scales=grid.pixel_scales
)

shear_y = al.Array2D.manual_native(array=shear_y.native, pixel_scales=grid.pixel_scales)
plotter = aplt.Array2DPlotter(array=shear_y)
plotter.figure_2d()

shear_x = al.Array2D.manual_native(array=shear_x.native, pixel_scales=grid.pixel_scales)
plotter = aplt.Array2DPlotter(array=shear_x)
plotter.figure_2d()
