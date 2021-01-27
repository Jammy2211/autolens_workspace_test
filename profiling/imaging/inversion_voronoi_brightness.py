import time
from os import path
import numpy as np

import autolens as al
import autolens.plot as aplt

repeats = 1

print("Number of repeats = " + str(repeats))
print()

sub_size = 8
radius = 3.5
psf_shape_2d = (11, 11)
pixels = 2000

print("sub grid size = " + str(sub_size))
print("circular mask radius = " + str(radius) + "\n")
print("psf shape = " + str(psf_shape_2d) + "\n")
print("pixels = " + str(pixels) + "\n")

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, phi=45.0),
    ),
)

pixelization = al.pix.VoronoiBrightnessImage(pixels=pixels)

instruments = ["hst"]  # , "hst_up"]
pixel_scales = [0.05]  # , 0.03]

for instrument, pixel_scale in zip(instruments, pixel_scales):

    dataset_path = path.join("dataset", "instruments", instrument)

    imaging = al.Imaging.from_fits(
        image_path=path.join(dataset_path, "image.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=pixel_scale,
    )

    mask = al.Mask2D.circular(
        shape_native=imaging.shape_native,
        pixel_scales=imaging.pixel_scales,
        sub_size=sub_size,
        radius=radius,
    )

    masked_imaging = al.MaskedImaging(
        imaging=imaging, mask=mask, settings=al.SettingsMaskedImaging(sub_size=sub_size)
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=pixelization,
        regularization=al.reg.Constant(coefficient=1.0),
        hyper_model_image=masked_imaging.image,
        hyper_galaxy_image=masked_imaging.image,
    )

    print("Inversion fit run times for image type " + instrument + "\n")
    print("Number of points = " + str(masked_imaging.grid.sub_shape_slim) + "\n")

    start_overall = time.time()

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    start = time.time()
    for i in range(repeats):
        cluster_weight_map = pixelization.weight_map_from_hyper_image(
            hyper_image=masked_imaging.image
        )
    diff = time.time() - start
    print("Time to Setup Cluster Weight Map = {}".format(diff / repeats))

    for i in range(repeats):
        sparse_grid = pixelization.sparse_grid_from_grid(
            grid=masked_imaging.grid, hyper_image=masked_imaging.image
        )
    diff = time.time() - start
    print("Time to perform KMeans clustering = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        traced_grid = tracer.traced_grids_of_planes_from_grid(grid=masked_imaging.grid)[
            -1
        ]
    diff = time.time() - start
    print("Time to Setup Traced Grid2D = {}".format(diff / repeats))

    traced_sparse_grid = tracer.traced_sparse_grids_of_planes_from_grid(
        grid=masked_imaging.grid
    )[-1]

    start = time.time()
    for i in range(repeats):
        mapper = pixelization.mapper_from_grid_and_sparse_grid(
            grid=traced_grid,
            sparse_grid=traced_sparse_grid,
            settings=al.SettingsPixelization(use_border=True),
        )
    diff = time.time() - start
    print("Time to create mapper (Border Relocation) = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        mapping_matrix = mapper.mapping_matrix
    diff = time.time() - start
    print("Time to compute mapping matrix = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        blurred_mapping_matrix = masked_imaging.convolver.convolve_mapping_matrix(
            mapping_matrix=mapping_matrix
        )
    diff = time.time() - start
    print("Time to compute blurred mapping matrix = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        data_vector = al.util.inversion.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=blurred_mapping_matrix,
            image=masked_imaging.image,
            noise_map=masked_imaging.noise_map,
        )
    diff = time.time() - start
    print("Time to compute data vector = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        curvature_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=blurred_mapping_matrix, noise_map=masked_imaging.noise_map
        )
    diff = time.time() - start
    print("Time to compute curvature matrix = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
            coefficient=1.0,
            pixel_neighbors=mapper.source_pixelization_grid.pixel_neighbors,
            pixel_neighbors_size=mapper.source_pixelization_grid.pixel_neighbors_size,
        )
    diff = time.time() - start
    print("Time to compute reguarization matrix = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)
    diff = time.time() - start
    print("Time to compute curvature reguarization Matrix = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        preconditioner_matrix = al.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=1.0,
            regularization_matrix=regularization_matrix,
        )
    diff = time.time() - start
    print("Time to compute preconditioner matrix = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)
    diff = time.time() - start
    print("Time to peform reconstruction = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        al.util.inversion.mapped_reconstructed_data_from(
            mapping_matrix=blurred_mapping_matrix, reconstruction=reconstruction
        )
    diff = time.time() - start
    print("Time to compute mapped reconstruction = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)
    diff = time.time() - start
    print("Time to perform complete fit = {}".format(diff / repeats))

    print()

    print(fit.log_evidence)

    fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_imaging_plotter.subplot_fit_imaging()
    fit_imaging_plotter.subplot_of_planes(plane_index=1)
