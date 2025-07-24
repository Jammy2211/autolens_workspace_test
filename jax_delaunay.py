import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial

"""
__JAX Delaunay Triangulation__

Load a 2D grid of 1000 (y,x) coordinates, which form the vertexes (corners) of a Delaunay triangulation.

Also includes are grids of 500 and 2000 points for profiling purposes.
"""
grid = np.load("source_plane_mesh_grid_x1000.npy")

"""
Plot the grid to see it looks like a source-plane.
"""
plt.figure(figsize=(8, 8))
plt.scatter(grid[:, 1], grid[:, 0], s=1, c="black")
plt.title("Source Plane Grid")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

"""
Input into scipy spatial Delaunay triangulation to create a triangulation object.

The `np.asarray` and transpose are what the source code uses to orient the grid correctly, but they are somewhat
arbitrary.
"""
delaunay = spatial.Delaunay(np.asarray([grid[:, 0], grid[:, 1]]).T)

"""
The key output of the Delaunay triangulation—difficult to compute in JAX—is the simplices, which are the indices of the 
vertices that form each triangle.

SciPy’s spatial library reliably computes these simplices, effectively mapping each point to its three neighboring 
vertices that form a Delaunay triangle.

Since there’s currently no practical way to compute this directly in JAX, the goal is to wrap this as a 
CPU-side "black box" function that can be called from a JAX workflow.


"""
print(delaunay.simplices)

"""
__CPU Rationale__

The Delaunay mesh computed above is a required input to the PyAutoLens likelihood function. Since it depends on the 
mass model—which changes during optimization—it must be recomputed for every evaluation and cannot be precomputed or 
treated as static in JAX.

While SciPy’s Delaunay function must run on the CPU, it can be safely wrapped as a “black box” callable 
within a JAX-based pipeline for the following reasons:

- Fast CPU Runtime: The input mesh consists of only ~1000–2000 points. Profiling on a typical laptop CPU 
shows runtimes < 0.01 seconds. In contrast, full likelihood evaluations for high-resolution datasets are >100× slower, 
making the Delaunay step a negligible cost.

- Minimal Data Transfer: Only a single small array—simplices, with shape approximately (1000, 3)—needs to be transferred 
from CPU to GPU memory. This data size is trivial and will not impact overall performance.

Static Array Shapes: The arrays containing the Delaunay simplices have fixed shapes (e.g. (1000, 3)), since the mesh
 size is determined by the number of source-plane pixels and remains constant across all likelihood evaluations. This
 should make CPU / GPU transfer feasible.

- Gradient Support: Even if two or more evaluations are needed to approximate gradients (e.g. for finite differences), 
the small size of the data being transferred ensures that this remains efficient.

__Other Methods__

The following methods are those used by PyAutoLens once the Delaunay triangulation is computed, but my understanding is
they will be easy to convert to JAX or are not needed at all.

__Points Property__

The delaunay `points `property is used, but it just contains the original grid coordinates.
"""
print(delaunay.points)

"""
__Find Simplex Method__

The calculation below uses the `source_plane_data_grid` which is a 2D grid of (y,x) coordinates that correspond to the
centre of every image pixel in the source-plane. This is NOT the Delaunay triangulation grid, but rather the pixels
which form the mappings of data to the source-plane.

This grid is oversampled, with the example loaded below using 4 x 4 oversampling (i.e. 16 pixels per source-plane pixel),
which for the low resolution data I built this example on is 2828 * 16 = 45,248 pixels.

For higher resolution data (or higher oversampling), this grid can have quite a lot of data points, and the example
`source_plane_data_grid_x.npy` contains ? points, allowing for profiling of examples with more data points.

This grid is not used to compute the `simplicies` and therefore should always stay in GPU memory, so regardless
of how big it is it should not impact the performance of the JAX code.
"""
source_plane_data_grid = np.load("source_plane_data_grid_sub_4.npy")

"""
The delaunay `find_simplex` method is used to find the simplex (triangle) that contains a given point, including 
the interpolated barycentric coordinates within that triangle.

This method receives as input the ray-traced data grid, which is a 2D grid of (y,x) coordinates that correspond to the
centre of every image pixel in the source-plane. 

This grid can have quite a lot of data points, as it is often oversampled,
with the example loaded below using 4 x 4 oversampling (i.e. 16 pixels per source-plane pixel).

My understanding is that this method is straight forward to write in JAX, with the hard part only being the computation
of the `simplices` property above. Furthermore, the input to this matrix is the traced image pixels, which whilst 
large will already be in GPU memory.
"""
delaunay.find_simplex(source_plane_data_grid)

"""
__Vertex Neighbor Vertices Method__

Another method used in the source code is `vertex_neighbor_vertices`, which returns the indices of the vertices
that are neighbors of each vertex in the triangulation.

Again, my understanding is this method is also straight forward to write in JAX.

This is only used for regularization and other calculations which are technically optional (e.g. we could use
the Matern kernel for regularization instead of the Delaunay triangulation).

If it were difficult, we can therefore bypass it.
"""
indptr, indices = delaunay.vertex_neighbor_vertices

"""
The areas of all Delaunay triangles are also used, but computed within the source code and look easy to do in JAX.
"""
