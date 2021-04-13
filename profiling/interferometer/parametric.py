import time

import autolens as al
import autolens.plot as aplt
import numpy as np

repeats = 1

workspace_path = "/Users/Jammy/Code/PyAuto/autolens_workspace"
dataset_type = "interferometer"
dataset_name = "sma"
dataset_path = path.join("dataset", dataset_type, dataset_name)

interferometer = al.Interferometer.from_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
)

# lens_galaxy = al.Galaxy(
#     redshift=0.5,
#     mass=al.mp.EllIsothermal(
#         centre=(0.4, 0.8), einstein_radius=0.1, elliptical_comps=(0.17647, 0.0)
#     ),
# )

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.0, 0.05)),
)

sersic = al.lp.EllSersic(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.01),
    intensity=0.1,
    effective_radius=0.2,
    sersic_index=3.0,
)

source_galaxy = al.Galaxy(redshift=1.0, light=sersic)

mask = al.Mask2D.circular(
    shape_native=(250, 250), pixel_scales=0.05, sub_size=2, radius=5.0
)

interferometer = al.MaskedInterferometer(
    interferometer=interferometer,
    real_space_mask=mask,
    visibilities_mask=np.full(
        fill_value=False, shape=interferometer.visibilities.shape
    ),
    settings=al.SettingsInterferometer(transformer_class=al.TransformerNUFFT),
)

print("Number of points = " + str(interferometer.grid.sub_shape_slim) + "\n")
print("Number of visibilities = " + str(interferometer.visibilities.shape_slim) + "\n")

start_overall = time.time()

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

start = time.time()

for i in range(repeats):
    fit = al.FitInterferometer(
        interferometer=interferometer,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_linear_operators=False),
    )

print(fit.log_likelihood)

diff = time.time() - start
print("Time to compute fit = {}".format(diff / repeats))

aplt.FitInterferometer.subplot_fit_real_space(fit=fit)
