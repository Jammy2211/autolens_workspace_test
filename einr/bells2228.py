import autolens as al

grid = al.Grid2D.uniform(shape_native=(1001, 1001), pixel_scales=0.02)

"""EEERR1"""

mass = al.mp.EllPowerLaw(
    centre=(0.0629, 0.0295),
    elliptical_comps=(-0.3977, -0.3306),
    einstein_radius=1.5034,
    slope=2.5343,
)
shear = al.mp.ExternalShear(elliptical_comps=(-0.256, -0.0068))

galaxy = al.Galaxy(redshift=0.5, mass=mass, shear=shear)
ein_r_0 = galaxy.einstein_radius_from(grid=grid)

print(ein_r_0)

"""EEERR2"""

mass = al.mp.EllPowerLaw(
    centre=(-0.0748, -0.1173),
    elliptical_comps=(-0.1202, -0.0221),
    einstein_radius=1.2056,
    slope=1.9373,
)
shear = al.mp.ExternalShear(elliptical_comps=(-0.1384, 0.1258))

galaxy = al.Galaxy(redshift=0.5, mass=mass, shear=shear)
ein_r_1 = galaxy.einstein_radius_from(grid=grid)

print(ein_r_1)
print(ein_r_0 / ein_r_1)
