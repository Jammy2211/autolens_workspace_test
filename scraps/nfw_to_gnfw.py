import autolens as al

nfw = al.mp.EllipticalNFWMCRLudlow(
    mass_at_200=5e13, redshift_object=0.169, redshift_source=0.451
)

print(nfw.kappa_s)
print(nfw.scale_radius)
print(nfw.concentration(redshift_profile=0.169, redshift_source=0.451))

nfw = al.mp.EllipticalNFWMCRLudlow(
    mass_at_200=1e14, redshift_object=0.169, redshift_source=0.451
)

print(nfw.kappa_s)
print(nfw.scale_radius)
print(nfw.concentration(redshift_profile=0.169, redshift_source=0.451))

nfw = al.mp.EllipticalNFWMCRLudlow(
    mass_at_200=5e14, redshift_object=0.169, redshift_source=0.451
)

print(nfw.kappa_s)
print(nfw.scale_radius)
print(nfw.concentration(redshift_profile=0.169, redshift_source=0.451))

nfw = al.mp.EllipticalGeneralizedNFW(kappa_s=1.0, scale_radius=60.0, inner_slope=2.0)

print(nfw.mass_at_200_solar_masses(redshift_object=0.169, redshift_source=0.451))
print(nfw.scale_radius)
print(nfw.concentration(redshift_profile=0.169, redshift_source=0.451))
