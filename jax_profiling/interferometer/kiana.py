import autolens as al
import autolens.plot as aplt
import numpy as np
import time


### Key run time parameters ###

mask_radius = 3.0
total_visibilities = 1000000

### Setup Data ###

real_space_mask = al.Mask2D.circular(
    shape_native=(350, 350), pixel_scales=0.05, radius=mask_radius
)

data = al.Visibilities(np.random.normal(loc=0.0, scale=1.0, size=total_visibilities) + 1j * np.random.normal(
    loc=0.0, scale=1.0, size=total_visibilities
))

noise_map = np.real(np.ones(total_visibilities) + 1j * np.ones(total_visibilities))

uv_wavelengths = np.random.uniform(
    low=-300.0, high=300.0, size=(total_visibilities, 2)
)

dataset = al.Interferometer(
    data=data,
    noise_map=noise_map,
    uv_wavelengths=uv_wavelengths,
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

# import time
#
# start = time.time()
#
# dataset = dataset.apply_sparse_operator()
#
# print(f"Time to compute W-Tilde: {time.time() - start} seconds")
#
# fff

interferometer = dataset


coeff = 1.0

whole_start_time = time.time()
print('--------------------------------------------------')
print('Starting to make cubes for Coeff ', coeff)

source_cube = np.zeros((1, 401, 401))
image_cube = np.zeros((1, 350, 350))
error_cube = np.zeros((1, 401, 401))

for i in range(0, 1):
    bin_n = i
    start_time = time.time()
    print('--------------------------------------------------')
    print('Starting to make images for Coeff ', coeff)
    # ------------------------------------------------------------------------------
    # Set Path and read in visbilites
    # ------------------------------------------------------------------------------
    Path = '/disk/aop4_1/kade/Autolens_G097/HighRes_CO65/'

    # visuals = aplt.Visuals2D(mask=real_space_mask_2d)
    # dataset_plotter = aplt.InterferometerPlotter(interferometer, visuals_2d=visuals)
    # dataset_plotter.subplot_dataset()
    # ------------------------------------------------------------------------------
    # Parameters for lensing model
    # ------------------------------------------------------------------------------
    z_source = 3.63

    # the following arrays have this form
    # z[0]	center_x[1]	center_y[2]	ellcomps0[3]	ellcomps1[4]	Einstein_Rad[5]

    lens_1 = al.Galaxy(redshift=0.5,
                       mass=al.mp.Isothermal(
                           centre=(0.0, 0.0),
                           ell_comps=(0.0, 0.0),
                           einstein_radius=1.0))

    lens_2 = al.Galaxy(redshift=0.5,
                       mass=al.mp.Isothermal(
                           centre=(0.0, 0.0),
                           ell_comps=(0.0, 0.0),
                           einstein_radius=1.0))

    ### Set up Delaunay using new features ###

    image_mesh=al.image_mesh.Overlay(shape=(60, 60))

    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask,
    )

    total_mapper_pixels = image_plane_mesh_grid.shape[0]

    total_linear_light_profiles = 0

    mapper_indices = al.mapper_indices_from(
        total_linear_light_profiles=total_linear_light_profiles,
        total_mapper_pixels=total_mapper_pixels,
    )

    preloads = al.Preloads(
        mapper_indices=mapper_indices,
    )

    pixelization = al.Pixelization(
        # image_mesh=al.image_mesh.Overlay(shape=(60, 60)),
        mesh=al.mesh.Delaunay(),
        regularization=al.reg.ConstantSplit(coefficient=coeff),
    )

    source_1 = al.Galaxy(redshift=z_source, pixelization=pixelization)

    adapt_images = al.AdaptImages(
        galaxy_image_plane_mesh_grid_dict={source_1: image_plane_mesh_grid}
    )

    # ------------------------------------------------------------------------------
    # Set up the tracer
    # ------------------------------------------------------------------------------
    tracer_ex = al.Tracer(galaxies=[lens_1, lens_2, source_1])

    fit = al.FitInterferometer(
        dataset=interferometer,
        tracer=tracer_ex,
        preloads=preloads,
        adapt_images=adapt_images
    )

    # tracer_to_inversion = al.TracerToInversion(
    #     tracer=tracer_ex,
    #     dataset=interferometer, )
    # data=interferometer.data,
    # noise_map=interferometer.noise_map,
    # w_tilde=True,)
    # settings_inversion = settings_inversion)

    inversion = fit.inversion
    mapper = inversion.linear_obj_list[0]

    mapper = inversion.cls_list_from(cls=al.AbstractMapper)[
        0
    ]  # Only one source-plane so only one mapper, would be a list if multiple source planes

    mapper_valued = al.MapperValued(
        mapper=mapper,
        values=inversion.reconstruction_dict[mapper]
    )

    interpolated_reconstruction = mapper_valued.interpolated_array_from(
        shape_native=(401, 401)
    )

    source_plane_array = interpolated_reconstruction.native
    image_plane_array = inversion.mapped_reconstructed_image.native


    mapper_valued = al.MapperValued(
        mapper=mapper,
        values=inversion.reconstruction_noise_map_dict[mapper]
    )

    interpolated_reconstruction = mapper_valued.interpolated_array_from(
        shape_native=(401, 401)
    )

    error_array = mapper_valued.interpolated_array_from(
        shape_native=(401, 401)
    ).native
    # ------------------------------------------------------------------------------
    # Bayesian log evidence for fit evaluation
    # ------------------------------------------------------------------------------
    # print("Bayesian Evidence Without Regularization for Coeff" + str(coeff) + ":")
    # print(fit.log_evidence)

    # ------------------------------------------------------------------------------
    # Px scale and offset things for source and image plane cubes
    # ------------------------------------------------------------------------------
    source_grid = mapper.mapper_grids.source_plane_mesh_grid
    image_grid = mapper.mapper_grids.image_plane_mesh_grid

    source_plane_xmin = min(source_grid[:, 0])
    source_plane_xmax = max(source_grid[:, 0])
    source_plane_ymin = min(source_grid[:, 1])
    source_plane_ymax = max(source_grid[:, 1])
    px_scale_sourceplane = (abs(source_plane_xmin) + source_plane_xmax) / source_grid.shape[0]
    px_scale_sourceplane_interpolated = (abs(source_plane_xmin) + source_plane_xmax) / source_plane_array.shape[0]
    source_ref_pix_x, source_ref_pix_y = source_plane_array.shape[0] / 2, source_plane_array.shape[1] / 2

    image_plane_xmin = min(image_grid[:, 0])
    image_plane_xmax = max(image_grid[:, 0])
    image_plane_ymin = min(image_grid[:, 1])
    image_plane_ymax = max(image_grid[:, 1])
    px_scale_imageplane = (abs(image_plane_xmin) + image_plane_xmax) / image_grid.shape[
        0]  # I don't really know why this is wrong but the below is correct
    px_scale_imageplane = (abs(image_plane_xmin) + image_plane_xmax) / image_plane_array.shape[0]
    image_ref_pix_x, image_ref_pix_y = image_plane_array.shape[0] / 2, image_plane_array.shape[1] / 2
    # ------------------------------------------------------------------------------
    # Source Plane Plot
    # ------------------------------------------------------------------------------
    title = aplt.Title(label="", color="black", fontsize=20)
    ylabel = aplt.YLabel(ylabel="Offset [arcsec]", color="black", fontsize=25)
    xlabel = aplt.XLabel(xlabel="Offset [arcsec]", color="black", fontsize=25)
    yticks = aplt.YTicks(fontsize=25, rotation="vertical")
    xticks = aplt.XTicks(fontsize=25, rotation="horizontal")

    figure = aplt.Figure(
        figsize=(7, 7),
        dpi=300.0,
        facecolor="white",
        edgecolor="black",
        frameon=True,
        clear=False,
        tight_layout=True,
        constrained_layout=False,
    )

    mat_plot_2d_cb = aplt.MatPlot2D(
        colorbar=aplt.Colorbar(manual_tick_values=[-1e-6, 6e-7, 2e-6], \
                               manual_tick_labels=['', '', ''], manual_unit='\n' + ''),
        colorbar_tickparams=aplt.ColorbarTickParams(labelsize=15, labelrotation=90),
        # axis = aplt.Axis(extent=[-0.25, 0.3, -0.35, 0.2]),
        title=title, ylabel=ylabel, xlabel=xlabel, yticks=yticks, xticks=xticks, figure=figure
    )

    tangential_caustics_plot = aplt.TangentialCausticsPlot(linestyle="-", linewidth=1, c="k")
    # radial_caustics_plot = aplt.RadialCausticsPlot(linestyle="--", linewidth=1, c="white")

    mat_plot_cau = aplt.MatPlot2D(
        tangential_caustics_plot=tangential_caustics_plot,
        #	radial_caustics_plot=radial_caustics_plot,
    )

    tangential_caustic_list = tracer_ex.tangential_caustic_list_from(grid=interferometer.grid)
    #   radial_caustics_list = tracer_ex.radial_caustic_list_from(grid=interferometer.grid)

    visuals = aplt.Visuals2D(
        tangential_caustics=tangential_caustic_list,
        #	radial_caustics=radial_caustics_list,
    )

    delaunay_drawer = aplt.DelaunayDrawer(edgecolor="black", linewidth=0.5, linestyle="-")

    cmap = aplt.Cmap(cmap="RdYlBu_r", norm="linear", vmin=np.std(inversion.reconstruction) * -2,
                     vmax=np.std(inversion.reconstruction) * 4)
    axis = aplt.Axis(extent=[-0.3, 0.3, -0.2, 0.4])
    mat_plot = aplt.MatPlot2D(delaunay_drawer=delaunay_drawer, cmap=cmap, axis=axis)

    m = mat_plot_2d_cb + mat_plot + mat_plot_cau

    inversion_plotter = aplt.InversionPlotter(inversion=inversion, mat_plot_2d=m,
                                              visuals_2d=visuals)
    inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)

    # ------------------------------------------------------------------------------
    # Image Plane Plot
    # ------------------------------------------------------------------------------
    title = aplt.Title(label="", color="black", fontsize=20)
    ylabel = aplt.YLabel(ylabel="Offset [arcsec]", color="black", fontsize=25)
    xlabel = aplt.XLabel(xlabel="Offset [arcsec]", color="black", fontsize=25)
    yticks = aplt.YTicks(fontsize=25, rotation="vertical")
    xticks = aplt.XTicks(fontsize=25, rotation="horizontal")


    figure = aplt.Figure(
        figsize=(7, 7),
        dpi=300.0,
        facecolor="white",
        edgecolor="black",
        frameon=True,
        clear=False,
        tight_layout=True,
        constrained_layout=False, )

    mat_plot_2d_cb = aplt.MatPlot2D(
        colorbar=aplt.Colorbar(manual_tick_values=[-1e-6, 6e-7, 2e-6], \
                               manual_tick_labels=['', '', ''], manual_unit='\n' + ''),
        colorbar_tickparams=aplt.ColorbarTickParams(labelsize=15, labelrotation=90),
        title=title, ylabel=ylabel, xlabel=xlabel, yticks=yticks, xticks=xticks, figure=figure)

    tangential_critical_curves_plot = aplt.TangentialCriticalCurvesPlot(linestyle="-", linewidth=1, c=['k', 'k'])
    # radial_critical_curves_plot = aplt.RadialCriticalCurvesPlot(linestyle="-", linewidth=1, c=['k', 'k'])

    mat_plot_crit = aplt.MatPlot2D(tangential_critical_curves_plot=tangential_critical_curves_plot,
                                   # radial_critical_curves_plot=radial_critical_curves_plot,
                                   )

    tangential_critical_curve_list = tracer_ex.tangential_critical_curve_list_from(grid=interferometer.grid)
    radial_critical_curves_list = tracer_ex.radial_critical_curve_list_from(grid=interferometer.grid)

    visuals = aplt.Visuals2D(tangential_critical_curves=tangential_critical_curve_list,
                             radial_critical_curves=radial_critical_curves_list)

    delaunay_drawer = aplt.DelaunayDrawer(edgecolor="black", linewidth=0.5, linestyle="-")

    cmap = aplt.Cmap(cmap="RdYlBu_r", norm="linear", vmin=np.std(inversion.reconstruction) * -2,
                     vmax=np.std(inversion.reconstruction) * 4)
    axis = aplt.Axis(extent=[-1.5, 1.5, -1.5, 1.5])
    mat_plot = aplt.MatPlot2D(delaunay_drawer=delaunay_drawer, cmap=cmap, axis=axis)

    m = mat_plot_2d_cb + mat_plot + mat_plot_crit

    inversion_plotter = aplt.InversionPlotter(inversion=inversion, mat_plot_2d=m,
                                              visuals_2d=visuals)
    inversion_plotter.figures_2d(reconstructed_image=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('finished coeff ', coeff, 'bin ', i, ": elapsed time for bin", i, ':', elapsed_time, "seconds")
    source_cube[bin_n, :, :] = source_plane_array
    image_cube[bin_n, :, :] = image_plane_array
    error_cube[bin_n, :, :] = np.sqrt(error_array)

# make_cubes(source_cube, 0.0, 50.0, source_ref_pix_x, source_ref_pix_y, px_scale_sourceplane_interpolated,
#            Path + 'Autolens_tracer_plotting/' + save_path, 'source_plane_cube')
# make_cubes(error_cube, 0.0, 50.0, source_ref_pix_x, source_ref_pix_y, px_scale_sourceplane_interpolated,
#            Path + 'Autolens_tracer_plotting/' + save_path, 'source_plane_error_cube')
# make_cubes(image_cube, 0.0, 50.0, image_ref_pix_x, image_ref_pix_y, px_scale_imageplane,
#            Path + 'Autolens_tracer_plotting/' + save_path, 'image_plane_cube')

whole_end_time = time.time()
whole_elapsed_time = whole_end_time - whole_start_time
print('Finished to make cubes for Coeff ', coeff, 'in', whole_elapsed_time, 'seconds')
print('--------------------------------------------------')



