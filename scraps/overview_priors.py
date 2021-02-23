lens_galaxy_model.mass.centre.centre_0 = af.UniformPrior(-0.01, 0.01)
lens_galaxy_model.mass.centre.centre_1 = af.UniformPrior(-0.01, 0.01)
lens_galaxy_model.mass.elliptical_comps.elliptical_comps_0 = af.UniformPrior(0.19, 0.21)
lens_galaxy_model.mass.elliptical_comps.elliptical_comps_1 = af.UniformPrior(0.05, 0.07)
lens_galaxy_model.mass.einstein_radius = af.UniformPrior(1.62, 1.66)

source_galaxy_model.disk.centre.centre_0 = af.UniformPrior(-0.29, -0.27)
source_galaxy_model.disk.centre.centre_1 = af.UniformPrior(0.34, 0.36)
source_galaxy_model.disk.elliptical_comps.elliptical_comps_0 = af.UniformPrior(
    -0.05, -0.03
)
source_galaxy_model.disk.elliptical_comps.elliptical_comps_1 = af.UniformPrior(
    -0.38, -0.36
)
source_galaxy_model.disk.intensity = af.UniformPrior(0.14, 0.15)
source_galaxy_model.disk.effective_radius = af.UniformPrior(0.13, 0.15)
source_galaxy_model.disk.sersic_index = af.UniformPrior(0.49, 0.51)
