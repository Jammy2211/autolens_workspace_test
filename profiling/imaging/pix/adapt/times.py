"""
__PROFILING: Inversion Voronoi__

This profiling script times how long it takes to fit `Imaging` data with a `Voronoi` pixelization for
datasets of varying resolution.

This represents the time taken by a single iteration of the **PyAutoLens** log likelihood function.
"""

import os
from os import path

import time
import json
from autoconf import conf
import autolens as al
import autolens.plot as aplt

"""
The path all profiling results are output.
"""
profiling_path = path.dirname(path.realpath(__file__))

file_path = os.path.join(profiling_path, "times", al.__version__)

"""
Whether w_tilde is used dictates the output folder.
"""
use_w_tilde = True
if use_w_tilde:
    file_path = os.path.join(file_path, "w_tilde")
else:
    file_path = os.path.join(file_path, "mapping")

"""
The number of repeats used to estimate the run time.
"""
repeats = conf.instance["general"]["profiling"]["repeats"]
print("Number of repeats = " + str(repeats))
print()

"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoLens values.
"""
sub_size = 4
mask_radius = 3.5
psf_shape_2d = (21, 21)
pixels = 1000

use_positive_only_solver = True

print(f"sub grid size = {sub_size}")
print(f"circular mask mask_radius = {mask_radius}")
print(f"psf shape = {psf_shape_2d}")
# print(f"pixels = {pixels}")

"""
The lens galaxy used to fit the data, which is identical to the lens galaxy used to simulate the data. 
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    # bulge=al.lp.Sersic(
    #     centre=(0.0, 0.0),
    #     ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    #     intensity=4.0,
    #     effective_radius=0.6,
    #     sersic_index=3.0,
    # ),
    # disk=al.lp.Exponential(
    #     centre=(0.0, 0.0),
    #     ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=30.0),
    #     intensity=2.0,
    #     effective_radius=1.6,
    # ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.001, gamma_2=0.001),
)


"""
The simulated data comes at five resolution corresponding to five telescopes:

vro: pixel_scale = 0.2", fastest run times.
euclid: pixel_scale = 0.1", fast run times
hst: pixel_scale = 0.05", normal run times, represents the type of data we do most our fitting on currently.
hst_up: pixel_scale = 0.03", slow run times.
ao: pixel_scale = 0.01", very slow :(
"""
# instrument = "vro"
# instrument = "euclid"
instrument = "hst"
# instrument = "hst_up"
# instrument = "ao"

pixel_scales_dict = {"vro": 0.2, "euclid": 0.1, "hst": 0.05, "hst_up": 0.03, "ao": 0.01}
pixel_scale = pixel_scales_dict[instrument]

"""
Load the dataset for this instrument / resolution.
"""
dataset_path = path.join("dataset", "imaging", "instruments", instrument)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scale,
    over_sample_size_pixelization=sub_size,
)

"""
Apply the 2D mask, which for the settings above is representative of the masks we typically use to model strong lenses.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

masked_dataset = dataset.apply_mask(mask=mask)

"""
Generate the adapt-images used to adapt the source pixelization and regularization.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)
lens_adapt_data = lens_galaxy.image_2d_from(grid=masked_dataset.grid)
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
traced_grid = tracer.traced_grid_2d_list_from(grid=masked_dataset.grid)[1]
source_adapt_data = source_galaxy.image_2d_from(grid=traced_grid)

"""
ADAPTIVE OVER SAMPLING, DO NOT USE
"""
# over_sampling = al.util.over_sample.over_sample_size_via_adapt_from(
#     data=source_adapt_data,
#     noise_map=masked_dataset.noise_map,
# )
#
# dataset = al.Imaging(
#     data=dataset.data,
#     noise_map=dataset.noise_map,
#     psf=dataset.psf,
#     over_sample_size_pixelization=over_sampling,
# )

masked_dataset = dataset.apply_mask(mask=mask)

masked_dataset = masked_dataset.apply_w_tilde()


"""
__JAX & Preloads__

In earlier examples (`imaging/features/pixelization/modeling`), we used JAX, which requires *preloading* array shapes
before compilation. In contrast, CPU modeling with `w_tilde` does **not** require JAX, allowing us to use larger meshes.

Below, notice how the `mesh_shape` is increased to **30 × 30**. Because CPU computation exploits sparse matrices and
benefits from larger system memory, we can now use higher-resolution pixelizations than were practical with JAX GPU
acceleration.
"""
image_mesh = None
mesh_shape = (30, 30)
total_mapper_pixels = mesh_shape[0] * mesh_shape[1]

total_linear_light_profiles = 0

preloads = al.Preloads(
    mapper_indices=al.mapper_indices_from(
        total_linear_light_profiles=total_linear_light_profiles,
        total_mapper_pixels=total_mapper_pixels,
    ),
    source_pixel_zeroed_indices=al.util.mesh.rectangular_edge_pixel_list_from(
        total_linear_light_profiles=total_linear_light_profiles,
        shape_native=mesh_shape,
    ),
)

"""
The source galaxy whose `VoronoiBrightness` `Pixelization` fits the data.
"""
# pixelization = al.Pixelization(
#     image_mesh=al.image_mesh.Hilbert(pixels=total_mapper_pixels, weight_floor=0.2, weight_power=3.0),
#     mesh=al.mesh.Delaunay(),
#     regularization=al.reg.AdaptiveBrightnessSplit(
#         inner_coefficient=0.01, outer_coefficient=100.0, signal_scale=0.05
#     ),
# )

pixelization = al.Pixelization(
    image_mesh=None,
    mesh=al.mesh.RectangularMagnification(shape=mesh_shape),
    regularization=al.reg.Constant(
        coefficient=1.0,
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
)

adapt_images = al.AdaptImages(
    galaxy_image_dict={
        lens_galaxy: lens_adapt_data,
        source_galaxy: source_adapt_data,
    }
)


"""
__Numba Caching__

Call FitImaging once to get all numba functions initialized.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(
    dataset=masked_dataset,
    tracer=tracer,
    adapt_images=adapt_images,
    settings_inversion=al.SettingsInversion(
        use_positive_only_solver=use_positive_only_solver,
    ),
)
print(f"Figure of Merit = {fit.figure_of_merit}")


"""
__Fit Time__

Time FitImaging by itself, to compare to profiling dict call.
"""
start = time.time()

print("")
print("")
print("")
print("")

repeats = 10

for i in range(repeats):
    fit = al.FitImaging(
        dataset=masked_dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings_inversion=al.SettingsInversion(
            use_positive_only_solver=use_positive_only_solver,
        ),
    )
    fit.log_evidence

fit_time = (time.time() - start) / repeats
print(f"Fit Time = {fit_time} \n")


print("")
print("")
print("")
print("")

"""
__Results__

These two numbers are the primary driver of run time. More pixels = longer run time.
"""

print(f"Inversion fit run times for image type {instrument} \n")
print(f"Number of pixels = {masked_dataset.grid.shape_slim} \n")
print(
    f"Number of sub-pixels = {masked_dataset.grids.pixelization.over_sampler.sub_total} \n"
)
fff


"""
__Output__

Output the profiling run times as a dictionary so they can be used in `profiling/graphs.py` to create graphs of the
profile run times.

This is stored in a folder using the **PyAutoLens** version number so that profiling run times can be tracked through
**PyAutoLens** development.
"""
if not os.path.exists(file_path):
    os.makedirs(file_path)

filename = f"{instrument}_run_time_dict.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

"""
Output the profiling run time of the entire fit.
"""
filename = f"{instrument}_fit_time.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

with open(path.join(file_path, filename), "w") as outfile:
    json.dump(fit_time, outfile)

print(fit)

"""
Output an image of the fit, so that we can inspect that it fits the data as expected.
"""
mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_fit", format="png"
    )
)
fit_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_plotter.subplot_fit()

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_of_plane_1", format="png"
    )
)
fit_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_plotter.subplot_of_planes(plane_index=1)

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path, filename=f"{instrument}_subplot_inversion_0", format="png"
    )
)
fit_plotter = aplt.InversionPlotter(inversion=fit.inversion, mat_plot_2d=mat_plot_2d)
fit_plotter.subplot_of_mapper(mapper_index=0)


