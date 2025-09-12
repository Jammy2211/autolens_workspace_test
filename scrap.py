from dask.distributed import Client
import time
import numpy as np
import jax.numpy as jnp
import jax
from jax import grad
from os import path

import autofit as af
import autolens as al
from autoconf import conf

def fit():

    dataset_name = "simple"
    dataset_path = path.join("dataset", "imaging", dataset_name)

    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, "data.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=0.1,
    )

    mask_2d = al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )

    dataset = dataset.apply_mask(mask=mask_2d)

    dataset = dataset.apply_over_sampling(over_sample_size_lp=1)

    positions = al.Grid2DIrregular(
        al.from_json(file_path=path.join(dataset_path, "positions.json"))
    )

    dataset.convolver

    # Lens:

    bulge = af.Model(al.lp_linear.Sersic)

    mass = af.Model(al.mp.Isothermal)

    shear = af.Model(al.mp.ExternalShear)

    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

    # Source:

    bulge = af.Model(al.lp_linear.Sersic)

    source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

    # Overall Lens Model:

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
    )

    from autofit.non_linear.fitness import Fitness

    fitness = Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )

    # 1) Start Dask client with 4 worker processes, 1 thread each
    client = Client(processes=True, n_workers=4, threads_per_worker=1)
    print("Dashboard:", client.dashboard_link)


    # 3) Scatter the entire dict as ONE Future (broadcast to all workers)
    #    Now workers will each have a local copy of lookup_dict
    fitness_future = client.scatter(fitness, broadcast=True)

    # 4) Define the worker function; it takes the key AND the lookup dict
    def process_key(key: str, lookup_dict: dict[str, float]) -> float:
        # lookup_dict is a plain dict here (not a Future)
        return lookup_dict[key] * 2.0

    # 5) Submit tasks in parallel, passing lookup_future as the second argument
    futures = [
        client.submit(process_key, key, lookup_future)
        for key in ["a", "b", "c"]
    ]

    # 6) Gather results back to the client and print
    results = client.gather(futures)
    print("Results:", results)

    # 7) Clean up
    client.close()


if __name__ == "__main__":
    fit()

#
#
#
# gggg
#
# import jax
# import jax.numpy as jnp
#
# # Original 5-element data vector
# data_vector = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
#
# # Original 5x5 matrix
# curvature_reg_matrix = jnp.array([
#     [1.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 2.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 3.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 4.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 5.0]
# ])
#
# # Indices not to solve for
# ids_to_remove = jnp.array([1, 4])
#
# # Create a boolean mask: True = keep, False = ignore
# mask = jnp.ones(data_vector.shape[0], dtype=bool).at[ids_to_remove].set(False)
#
# # Zero out entries we don't want to solve for
# data_vector_masked = data_vector * mask
#
# # Zero rows and columns in the matrix we want to ignore
# mask_matrix = mask[:, None] * mask[None, :]
# curvature_reg_matrix_masked = curvature_reg_matrix * mask_matrix
#
# # Fake solution: just pass masked data through (in practice you’d solve a system here)
# reconstruction_full = data_vector_masked  # this simulates solving with masked inputs
#
# # Show output
# print("Mask:", mask)
# print("Masked data vector:", data_vector_masked)
# print("Masked matrix:\n", curvature_reg_matrix_masked)
# print("Reconstruction (full size):", reconstruction_full)
#
#
# fff
#
# import jax
# import jax.numpy as jnp
#
# import autoarray as aa
#
# source_plane_mesh_grid = jnp.array(
#       [[ 0.83333334, -0.83333334],
#        [ 0.83333334,  0.        ],
#        [ 0.83333334,  0.83333334],
#        [-0.        , -0.83333334],
#        [-0.        ,  0.        ],
#        [-0.        ,  0.83333334],
#        [-0.83333334, -0.83333334],
#        [-0.83333334,  0.        ],
#        [-0.83333334,  0.83333334]]
# )
#
# source_plane_data_grid = aa.Grid2DIrregular([[ 1.25, -1.25],
#        [ 1.25, -0.75],
#        [ 0.75, -1.25],
#        [ 0.75, -0.75],
#        [ 1.25, -0.25],
#        [ 1.25,  0.25],
#        [ 0.75, -0.25],
#        [ 0.75,  0.25],
#        [ 1.25,  0.75],
#        [ 1.25,  1.25],
#        [ 0.75,  0.75],
#        [ 0.75,  1.25],
#        [ 0.25, -1.25],
#        [ 0.25, -0.75],
#        [-0.25, -1.25],
#        [-0.25, -0.75],
#        [ 0.25, -0.25],
#        [ 0.25,  0.25],
#        [-0.25, -0.25],
#        [-0.25,  0.25],
#        [ 0.25,  0.75],
#        [ 0.25,  1.25],
#        [-0.25,  0.75],
#        [-0.25,  1.25],
#        [-0.75, -1.25],
#        [-0.75, -0.75],
#        [-1.25, -1.25],
#        [-1.25, -0.75],
#        [-0.75, -0.25],
#        [-0.75,  0.25],
#        [-1.25, -0.25],
#        [-1.25,  0.25],
#        [-0.75,  0.75],
#        [-0.75,  1.25],
#        [-1.25,  0.75],
#        [-1.25,  1.25]])
#
#
# shape_native = (3,3)
# source_plane_mesh_grid = source_plane_mesh_grid.reshape(*shape_native, 2)
#
# # Assume mesh is shaped (Ny, Nx, 2)
# Ny, Nx = source_plane_mesh_grid.shape[:2]
#
# # Get mesh spacings and lower corner
# y_coords = source_plane_mesh_grid[:, 0, 0]  # shape (Ny,)
# x_coords = source_plane_mesh_grid[0, :, 1]  # shape (Nx,)
#
# dy = y_coords[1] - y_coords[0]
# dx = x_coords[1] - x_coords[0]
#
# y_min = y_coords[0]
# x_min = x_coords[0]
#
# # shape (N_irregular, 2)
# irregular = source_plane_data_grid
#
# # Compute normalized mesh coordinates (floating indices)
# fy = (irregular[:, 0] - y_min) / dy
# fx = (irregular[:, 1] - x_min) / dx
#
# # Integer indices of top-left corners
# ix = jnp.floor(fx).astype(jnp.int32)
# iy = jnp.floor(fy).astype(jnp.int32)
#
# # Clip to stay within bounds
# ix = jnp.clip(ix, 0, Nx - 2)
# iy = jnp.clip(iy, 0, Ny - 2)
#
# # Local coordinates inside the cell (0 <= tx, ty <= 1)
# tx = fx - ix
# ty = fy - iy
#
# # Bilinear weights
# w00 = (1 - tx) * (1 - ty)
# w10 = tx * (1 - ty)
# w01 = (1 - tx) * ty
# w11 = tx * ty
#
# weights = jnp.stack([w00, w10, w01, w11], axis=1)  # shape (N_irregular, 4)
#
# # Compute indices of 4 surrounding pixels in the flattened mesh
# i00 = iy * Nx + ix
# i10 = iy * Nx + (ix + 1)
# i01 = (iy + 1) * Nx + ix
# i11 = (iy + 1) * Nx + (ix + 1)
#
# indices = jnp.stack([i00, i10, i01, i11], axis=1)  # shape (N_irregular, 4)
#
# print(weights)
# print(indices)
#
#
#
#
# import numpy as np
# from scipy.interpolate import RegularGridInterpolator
#
# # Step 1: Create the regular mesh grid (3x3), reshape to (3,3,2)
# source_plane_mesh_grid = np.array(
#     [[0.83333334, -0.83333334],
#      [0.83333334,  0.        ],
#      [0.83333334,  0.83333334],
#      [-0.        , -0.83333334],
#      [-0.        ,  0.        ],
#      [-0.        ,  0.83333334],
#      [-0.83333334, -0.83333334],
#      [-0.83333334,  0.        ],
#      [-0.83333334,  0.83333334]]
# ).reshape((3, 3, 2))
#
# # Step 2: Extract 1D coordinate arrays
# y_coords = source_plane_mesh_grid[:, 0, 0]
# x_coords = source_plane_mesh_grid[0, :, 1]
#
# # Step 3: Use a fake value field for testing, e.g. v = y + x
# values = source_plane_mesh_grid[..., 0] + source_plane_mesh_grid[..., 1]
#
# # Step 4: Define the interpolator
# interpolator = RegularGridInterpolator((y_coords, x_coords), values, method='linear')
#
# # Step 5: Irregular grid points (e.g., your over_sampled grid)
# source_plane_data_grid = np.array([
#     [ 1.25, -1.25],
#     [ 1.25, -0.75],
#     [ 0.75, -1.25],
#     [ 0.75, -0.75],
#     [ 1.25, -0.25],
#     # ... add more if needed
# ])
#
# # Step 6: Interpolate!
# interpolated = interpolator(source_plane_data_grid)
#
# print(interpolated)
#
#
#
