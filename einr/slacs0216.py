import autolens as al

grid = al.Grid2D.uniform(shape_native=(1001, 1001), pixel_scales=0.02)

mass = al.mp.EllPowerLaw(
    centre=(-0.0015, -0.0006),
    elliptical_comps=(0.0507, -0.1035),
    einstein_radius=1.1814,
    slope=2.0102,
)
shear = al.mp.ExternalShear(elliptical_comps=(-0.0031, 0.0077))

galaxy = al.Galaxy(redshift=0.5, mass=mass, shear=shear)
ein_r = galaxy.einstein_radius_from(grid=grid)

mass_no_shear = al.mp.EllPowerLaw(
    centre=(0.035, 0.01),
    elliptical_comps=(0.017, -0.073),
    einstein_radius=1.175,
    slope=1.735,
)

galaxy_no_shear = al.Galaxy(redshift=0.5, mass=mass_no_shear)
ein_r_no_shear = galaxy_no_shear.einstein_radius_from(grid=grid)

print(ein_r)
print(ein_r_no_shear)
print(ein_r / ein_r_no_shear)

"""SIE"""
print()

mass = al.mp.EllIsothermal(
    centre=(-0.0023, -0.0013),
    elliptical_comps=(0.0993, -0.1231),
    einstein_radius=1.1985,
)
shear = al.mp.ExternalShear(elliptical_comps=(0.0294, 0.0017))

galaxy = al.Galaxy(redshift=0.5, mass=mass, shear=shear)
ein_r = galaxy.einstein_radius_from(grid=grid)

mass_no_shear = al.mp.EllIsothermal(
    centre=(-0.0007, -0.0003),
    elliptical_comps=(0.0291, -0.1184),
    einstein_radius=1.1867,
)

galaxy_no_shear = al.Galaxy(redshift=0.5, mass=mass_no_shear)
ein_r_no_shear = galaxy_no_shear.einstein_radius_from(grid=grid)

print(ein_r)
print(ein_r_no_shear)
print(ein_r / ein_r_no_shear)
