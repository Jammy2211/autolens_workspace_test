import json

import autolens as al
from autoconf.dictable import to_dict
from autolens.point.triangles.triangle_solver import TriangleSolver
import autolens.plot as aplt
import autofit as af

grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

mass = af.Model(al.mp.PowerLaw)

mass.centre = (0.0, 0.0)
mass.ell_comps.ell_comps_0 = af.GaussianPrior(
    mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
mass.ell_comps.ell_comps_1 = af.GaussianPrior(
    mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
)
mass.einstein_radius = af.UniformPrior(lower_limit=0.4, upper_limit=1.8)
# mass.slope = af.GaussianPrior(mean=2.0, sigma=0.1)
mass.slope = 2.0

shear = af.Model(al.mp.ExternalShear)

shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.05)
shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.05)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    mass=mass,
    # shear=shear,
)

isothermal_mass_profile = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=isothermal_mass_profile,
)

centre = af.TuplePrior(
    centre_0=af.UniformPrior(lower_limit=-0.1, upper_limit=0.1),
    centre_1=af.UniformPrior(lower_limit=-0.1, upper_limit=0.1),
)

point_source = af.Model(
    al.ps.PointSourceChi,
    centre=centre,
)

source_galaxy = af.Model(
    al.Galaxy,
    redshift=1.0,
    light=af.Model(
        al.lp.Exponential,
        centre=centre,
        intensity=0.1,
        effective_radius=0.1,
    ),
    point_0=point_source,
)

model = af.Collection(
    source_galaxy=source_galaxy,
    lens_galaxy=lens_galaxy,
)

for i in range(100):
    instance = model.random_instance()

    tracer = al.Tracer(galaxies=[instance.lens_galaxy, instance.source_galaxy])

    solver = TriangleSolver(
        grid=grid,
        lensing_obj=tracer,
        pixel_scale_precision=0.001,
        buffer=0.1,
    )

    triangle_positions = solver.solve(
        source_plane_coordinate=instance.source_galaxy.point_0.centre
    )

    visuals = aplt.Visuals2D(multiple_images=triangle_positions)

    output = aplt.Output(
        path="../../../scratch/examples", filename=f"{i}", format="png"
    )
    mat_plot = aplt.MatPlot2D(output=output)

    tracer_plotter = aplt.TracerPlotter(
        tracer=tracer,
        grid=grid,
        visuals_2d=visuals,
        mat_plot_2d=mat_plot,
    )
    tracer_plotter.figures_2d(image=True)

    with open(f"examples/{i}.json", "w") as f:
        json.dump(to_dict(instance), f, indent=4)
