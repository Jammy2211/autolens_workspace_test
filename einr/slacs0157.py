import autolens as al

grid = al.Grid2D.uniform(shape_native=(1001, 1001), pixel_scales=0.05)

mass = al.mp.EllPowerLaw(
    centre=(0.0259, -0.1313),
    elliptical_comps=(-0.1348, -0.2548),
    einstein_radius=1.0144,
    slope=2.2730,
)
shear = al.mp.ExternalShear(elliptical_comps=(-0.0705, -0.1813))

galaxy = al.Galaxy(redshift=0.5, mass=mass, shear=shear)
ein_r = galaxy.einstein_radius_from(grid=grid)

mass_no_shear = al.mp.EllPowerLaw(
    centre=(-0.0221, -0.003),
    elliptical_comps=(0.0590, -0.0582),
    einstein_radius=0.8613,
    slope=2.1583,
)

galaxy_no_shear = al.Galaxy(redshift=0.5, mass=mass_no_shear)
ein_r_no_shear = galaxy_no_shear.einstein_radius_from(grid=grid)

print(ein_r)
print(ein_r_no_shear)
print(ein_r / ein_r_no_shear)

# """SIE"""
# print()
#
# mass = al.mp.EllIsothermal(
#     centre=(-0.0023, -0.0013),
#     elliptical_comps=(0.0993, -0.1231),
#     einstein_radius=1.1985,
# )
# shear = al.mp.ExternalShear(elliptical_comps=(0.0294, 0.0017))
#
# galaxy = al.Galaxy(
#     redshift=0.5,
#     mass=mass,
#     shear=shear
# )
# ein_r = galaxy.einstein_radius_from(grid=grid)
#
# mass_no_shear = al.mp.EllIsothermal(
#     centre=(-0.0007, -0.0003),
#     elliptical_comps=(0.0291, -0.1184),
#     einstein_radius=1.1867,
# )
#
# galaxy_no_shear = al.Galaxy(
#     redshift=0.5,
#     mass=mass_no_shear
# )
# ein_r_no_shear = galaxy_no_shear.einstein_radius_from(grid=grid)
#
# print(ein_r)
# print(ein_r_no_shear)
# print(ein_r / ein_r_no_shear)