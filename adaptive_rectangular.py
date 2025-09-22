import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from autoconf import cached_property

from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper
from autoarray.inversion.pixelization.mappers.abstract import PixSubWeights
from autoarray.structures.mesh.rectangular_2d import Mesh2DRectangular

from autoarray.inversion.pixelization.mappers import mapper_util


shape_native = (10, 10)
source_plane_mesh_grid = np.load("source_plane_mesh_grid.npy")
source_plane_data_grid = np.load("source_plane_data_grid.npy")
source_plane_data_grid_native = np.load("source_plane_data_grid_native.npy")
source_plane_data_grid_over_sampled = np.load("source_plane_data_grid_over_sampled.npy")

mesh = Mesh2DRectangular.overlay_grid(
    shape_native=shape_native,
    grid=source_plane_data_grid
)

np.save("edges", mesh.edges)

fff


y_edges = jnp.unique(mesh.edges[...,0])  # shape (Ny+1,)
x_edges = jnp.unique(mesh.edges[...,1])  # shape (Nx+1,)

print(y_edges)
print(x_edges)

Ny, Nx = shape_native
# Collect x and y edges: each pixel contributes (x0,x1) and (y0,y1)
y_edges = jnp.unique(mesh.edges_transformed[...,0])  # shape (Ny+1,)
x_edges = jnp.unique(mesh.edges_transformed[...,1])  # shape (Nx+1,)

print(y_edges)
print(x_edges)
ffff
#
# X, Y = jnp.meshgrid(x_edges, y_edges)

grid_edges_2d = mesh.edges_transformed.reshape(-1, 2).reshape(shape_native[0]+1, shape_native[1]+1, 2)

print(grid_edges_2d.shape)

C = jnp.arange(1, Ny * Nx + 1).reshape(Ny, Nx)

plt.pcolormesh(
    grid_edges_2d[..., 0],
    grid_edges_2d[..., 1],
    C,
    shading='flat'
)


# print(mesh.edges_transformed)

# mappings, weights = (
#     mapper_util.rectangular_mappings_weights_via_interpolation_from(
#         shape_native=shape_native,
#         source_plane_mesh_grid=source_plane_mesh_grid,
#         source_plane_data_grid=source_plane_data_grid,
#     )
# )
#
# print(source_plane_data_grid_over_sampled)

# source_plane_data_grid_over_sampled[36,0] = 0.03
# source_plane_data_grid_over_sampled[36,1] = 0.03
#
# mappings, weights = (
#     mapper_util.adaptive_rectangular_mappings_weights_via_interpolation_from(
#         source_grid_size=32,
#         source_plane_data_grid=jnp.array(source_plane_data_grid),
#         source_plane_data_grid_over_sampled=jnp.array(source_plane_data_grid_over_sampled),
#     )
# )
#
# for i in range(len(mappings)):
#     print(i, source_plane_data_grid_over_sampled[i,:], mappings[i], weights[i])
#     if i == 36:
#         fff

