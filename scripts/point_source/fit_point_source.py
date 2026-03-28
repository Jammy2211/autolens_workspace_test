import autolens as al
import autofit as af
from autolens.point.analysis import AnalysisAllToAllPointSource

grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.05,
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
    al.ps.Point,
    centre=centre,
)

model = af.Collection(
    source=point_source,
    lens=lens_galaxy,
)


analysis = AnalysisAllToAllPointSource(
    coordinates=[
        (-1.29, -0.98),
        (1.02, 1.11),
        (0.43, -1.58),
        (-1.61, 0.27),
    ],
    error=0.1,
    grid=grid,
)

if __name__ == "__main__":
    search = af.DynestyStatic("point_source")

    result = search.fit(model=model, analysis=analysis)

    print(result)
