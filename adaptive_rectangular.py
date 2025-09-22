# <<<<<<< HEAD
# import jax.numpy as jnp
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Tuple
#
# from autoconf import cached_property
#
# from autoarray.structures.grids.irregular_2d import Grid2DIrregular
# from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper
# from autoarray.inversion.pixelization.mappers.abstract import PixSubWeights
# from autoarray.structures.mesh.rectangular_2d import Mesh2DRectangular
#
# from autoarray.inversion.pixelization.mappers import mapper_util
#
#
# shape_native = (10, 10)
# source_plane_mesh_grid = np.load("source_plane_mesh_grid.npy")
# source_plane_data_grid = np.load("source_plane_data_grid.npy")
# source_plane_data_grid_native = np.load("source_plane_data_grid_native.npy")
# source_plane_data_grid_over_sampled = np.load("source_plane_data_grid_over_sampled.npy")
#
# mesh = Mesh2DRectangular.overlay_grid(
#     shape_native=shape_native,
#     grid=source_plane_data_grid
# )
#
# np.save("edges", mesh.edges)
#
# fff
#
#
# y_edges = jnp.unique(mesh.edges[...,0])  # shape (Ny+1,)
# x_edges = jnp.unique(mesh.edges[...,1])  # shape (Nx+1,)
#
# print(y_edges)
# print(x_edges)
#
# Ny, Nx = shape_native
# # Collect x and y edges: each pixel contributes (x0,x1) and (y0,y1)
# y_edges = jnp.unique(mesh.edges_transformed[...,0])  # shape (Ny+1,)
# x_edges = jnp.unique(mesh.edges_transformed[...,1])  # shape (Nx+1,)
#
# print(y_edges)
# print(x_edges)
# ffff
# #
# # X, Y = jnp.meshgrid(x_edges, y_edges)
#
# grid_edges_2d = mesh.edges_transformed.reshape(-1, 2).reshape(shape_native[0]+1, shape_native[1]+1, 2)
#
# print(grid_edges_2d.shape)
#
# C = jnp.arange(1, Ny * Nx + 1).reshape(Ny, Nx)
#
# plt.pcolormesh(
#     grid_edges_2d[..., 0],
#     grid_edges_2d[..., 1],
#     C,
#     shading='flat'
# )
#
#
# # print(mesh.edges_transformed)
#
# # mappings, weights = (
# #     mapper_util.rectangular_mappings_weights_via_interpolation_from(
# #         shape_native=shape_native,
# #         source_plane_mesh_grid=source_plane_mesh_grid,
# #         source_plane_data_grid=source_plane_data_grid,
# #     )
# # )
# #
# # print(source_plane_data_grid_over_sampled)
#
# # source_plane_data_grid_over_sampled[36,0] = 0.03
# # source_plane_data_grid_over_sampled[36,1] = 0.03
# #
# # mappings, weights = (
# #     mapper_util.adaptive_rectangular_mappings_weights_via_interpolation_from(
# #         source_grid_size=32,
# #         source_plane_data_grid=jnp.array(source_plane_data_grid),
# #         source_plane_data_grid_over_sampled=jnp.array(source_plane_data_grid_over_sampled),
# #     )
# # )
# #
# # for i in range(len(mappings)):
# #     print(i, source_plane_data_grid_over_sampled[i,:], mappings[i], weights[i])
# #     if i == 36:
# #         fff
# =======
# from pathlib import Path
# import autolens as al
# import autolens.plot as aplt
#
# grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)
#
# dataset_name = "lens_sersic"
# dataset_path = Path("dataset") / "imaging" / dataset_name
#
# dataset = al.Imaging.from_fits(
#     data_path=dataset_path / "data.fits",
#     psf_path=dataset_path / "psf.fits",
#     noise_map_path=dataset_path / "noise_map.fits",
#     pixel_scales=0.1,
# )
#
# mask = al.Mask2D.circular(
#     shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
# )
#
# dataset = dataset.apply_mask(mask=mask)
#
# lens_galaxy = al.Galaxy(
#     redshift=0.5,
#     bulge=al.lp.Sersic(
#         centre=(0.0, 0.0),
#         ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
#         intensity=1.0,
#         effective_radius=0.8,
#         sersic_index=4.0,
#     ),
#     mass=al.mp.Isothermal(
#         centre=(0.0, 0.0),
#         einstein_radius=1.6,
#         ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
#     ),
#     shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
# )
#
# pixelization = al.Pixelization(
#     image_mesh=al.image_mesh.Overlay(shape=(25, 25)),
#     mesh=al.mesh.Rectangular(shape=(10, 10)),
#     regularization=al.reg.Constant(coefficient=1.0),
# )
#
# source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)
#
# tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
#
# fit = al.FitImaging(dataset=dataset, tracer=tracer)
#
# inversion = fit.inversion
#
# print(fit.figure_of_merit)
# >>>>>>> b4aa4ff9b1fc4006fb46a1d3e4df51b5cb94e0c8
#
