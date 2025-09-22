import autolens as al
import pandas as pd
import numpy as np

# Load the CSV (replace with your filename)
df = pd.read_csv("mass_profiles.csv")

lens_galaxies = []

for _, row in df.iterrows():
    # mass = al.mp.dPIE(
    #     centre=(row["center_y(arcsec)"], row["center_x(arcsec)"]),
    #     ell_comps=(0.0, 0.0),
    #     ra=row["r_core(arcsec)"],
    #     rs=row["r_cut(arcsec)"],
    #     kappa_scale=row["kappa_scale"],
    # )

    mass = al.mp.IsothermalSph(
        centre=(row["center_y(arcsec)"], row["center_x(arcsec)"]),
        einstein_radius=(row["kappa_scale"] / (834.6912404))
    )

    lens_galaxy = al.Galaxy(redshift=0.5, mass=mass)

    lens_galaxies.append(lens_galaxy)


source_galaxy = al.Galaxy(
    redshift=1.0,
    point_0=al.ps.Point(centre=(0.07, 0.07)),
)


tracer = al.Tracer(galaxies=lens_galaxies + [source_galaxy])

grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=1.0,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

print(grid)

print(np.max(tracer.deflections_yx_2d_from(grid=grid)))



solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
)

print(positions)

gggg