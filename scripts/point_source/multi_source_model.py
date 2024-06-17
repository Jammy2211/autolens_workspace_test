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

analyses = []
multiple_image_collections = [
    [
        (1.4103868896870786, 0.6578124999999888),
        (-0.6094369426810129, -1.4421875000000037),
        (1.0346581598160152, -1.1992187500000047),
        (-1.2161057802279271, 1.138281249999987),
    ],
    [
        (-0.9341964691001788, 1.3906249999999865),
        (1.5673539941230086, -0.23437500000000808),
        (-0.6680740793955845, -1.376562500000004),
        (1.5835919704439665, 0.05937499999999091),
    ],
]

analysis = sum(
    AnalysisAllToAllPointSource(
        coordinates=coordinates,
        error=0.1,
        grid=grid,
    ).with_model(
        af.Collection(
            lens=lens_galaxy,
            source=af.Model(
                al.ps.Point,
            ),
        )
    )
    for coordinates in multiple_image_collections
)

if __name__ == "__main__":
    search = af.DynestyStatic("point_source")

    result = search.fit(model=None, analysis=analysis)

    print(result)
