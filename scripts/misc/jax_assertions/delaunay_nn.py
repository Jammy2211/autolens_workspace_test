"""
Correctness, reconstruction, autodiff, and performance gate for DelaunayNN.

The first check exercises the public ``aa.mesh.DelaunayNN`` through a Mapper
and Inversion and compares its reconstructed source with the otherwise
identical barycentric ``aa.mesh.Delaunay`` inversion. The remaining checks use
synthetic production-sized arrays so they can be timed on an accelerator:

* public mesh/interpolator, split-regularization, and source reconstruction;
* JIT execution and finite, normalized Sibson weights;
* exact linear precision and its analytic query-coordinate gradient;
* continuity of values and gradients through a Delaunay diagonal flip;
* a jitted ``vmap`` through independent qhull callbacks;
* single-pass parity: the one concatenated data-grid + split-point Sibson pass
  inside ``jax_delaunay_nn`` reproduces two separate ``jax_sibson`` passes
  exactly, for both halves;
* split-regularization compaction parity: the compacted JAX assembly of the
  ``ConstantSplit`` matrix (a narrow main scatter plus a wide-row supplement)
  reproduces the NumPy matrix, including on a table with a forced row wider
  than the compact width, and NaNs out when the wide-row budget overflows;
* query-chunk invariance: the ``jax.lax.map`` block size is a pure memory
  guard, so 64, 256 and 1024 give bit-identical tables, and the import-time
  ``PYAUTO_SIBSON_QUERY_CHUNK`` override reaches ``DelaunayNN.query_chunk``;
* warm runtime against the current barycentric Delaunay interpolation.

Override ``SIBSON_POINTS``, ``SIBSON_QUERIES`` and ``SIBSON_REPEATS`` for a
short local probe or a larger accelerator run.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Every check below gates a JAX code path — JIT execution, the analytic
query-coordinate gradient and a jitted ``vmap`` through the qhull callbacks —
so JAX must stay enabled. The timing gate runs on synthetic production-sized
arrays, so the SMALL_DATASETS cap must stay off or it measures nothing. The
env-override check spawns one short subprocess with
``PYAUTO_SIBSON_QUERY_CHUNK`` set; that variable must be unset in the parent,
because the library reads it once at import.

ENV: jax full_datasets
"""

import os
import subprocess
import sys
import time

import autoarray as aa
import jax
import jax.numpy as jnp
import numpy as np
from autoarray import fixtures
from autoarray.inversion.mesh.interpolator.delaunay import (
    _jax_delaunay_tables,
    jax_delaunay,
    pix_indexes_delaunay_walk_from,
    pixel_weights_delaunay_from,
)
from autoarray.inversion.mesh.interpolator.sibson import (
    SIBSON_QUERY_CHUNK,
    InterpolatorDelaunayNN,
    jax_delaunay_nn,
    jax_sibson,
)
from autoarray.inversion.regularization import regularization_util
from autoarray.inversion.regularization.regularization_util import (
    SPLIT_REG_COMPACT_WIDTH,
)

jax.config.update("jax_enable_x64", True)

POINT_COUNT = int(os.environ.get("SIBSON_POINTS", "1200"))
QUERY_COUNT = int(os.environ.get("SIBSON_QUERIES", "15000"))
# The repeat loop only tightens a printed timing; no assertion reads it, so one pass is enough
# on the gate. Raise `SIBSON_REPEATS` when actually benchmarking.
REPEATS = int(os.environ.get("SIBSON_REPEATS", "1"))
MAX_CAVITY_TRIANGLES = int(os.environ.get("SIBSON_CAVITY", "32"))
MAX_NEIGHBORS = int(os.environ.get("SIBSON_NEIGHBORS", "32"))
QUERY_CHUNK = int(os.environ.get("SIBSON_CHUNK", "256"))


def mapper_from(mesh, mesh_grid, data_grid):
    interpolator = mesh.interpolator_from(
        source_plane_data_grid=data_grid,
        source_plane_mesh_grid=mesh_grid,
        adapt_data=aa.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1),
    )
    return aa.Mapper(
        interpolator=interpolator,
        regularization=aa.reg.Constant(coefficient=1.0),
        image_plane_mesh_grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1),
    )


# Public integration path: the same dataset, source-plane vertices and
# regularization are inverted with Delaunay and DelaunayNN. The source vectors
# need not be identical because the interpolation bases differ, but a smooth
# reconstruction should be numerically very close.
mesh_grid_9 = aa.Grid2D.no_mask(
    values=[
        [0.6, -0.3],
        [0.5, -0.8],
        [0.2, 0.1],
        [0.0, 0.5],
        [-0.3, -0.8],
        [-0.6, -0.5],
        [-0.4, -1.1],
        [-1.2, 0.8],
        [-1.5, 0.9],
    ],
    shape_native=(3, 3),
    pixel_scales=1.0,
)
data_grid = fixtures.make_grid_2d_sub_2_7x7()
dataset = fixtures.make_masked_imaging_7x7()

delaunay_mapper = mapper_from(aa.mesh.Delaunay(pixels=9), mesh_grid_9, data_grid)
delaunay_nn_mesh = aa.mesh.DelaunayNN(pixels=9)
delaunay_nn_mapper = mapper_from(delaunay_nn_mesh, mesh_grid_9, data_grid)

assert delaunay_nn_mesh.interpolator_cls is InterpolatorDelaunayNN
assert isinstance(delaunay_nn_mapper.interpolator, InterpolatorDelaunayNN)
assert not np.asarray(delaunay_nn_mapper.interpolator.delaunay.overflow).any()
assert not np.asarray(delaunay_nn_mapper.interpolator.delaunay.degenerate).any()
assert not np.asarray(delaunay_nn_mapper.interpolator.delaunay.split_overflow).any()
assert not np.asarray(delaunay_nn_mapper.interpolator.delaunay.split_degenerate).any()

delaunay_inversion = aa.Inversion(
    dataset=dataset,
    linear_obj_list=[delaunay_mapper],
)
delaunay_nn_inversion = aa.Inversion(
    dataset=dataset,
    linear_obj_list=[delaunay_nn_mapper],
)

delaunay_source = np.asarray(delaunay_inversion.reconstruction)
delaunay_nn_source = np.asarray(delaunay_nn_inversion.reconstruction)
delaunay_source_relative_l2 = np.linalg.norm(
    delaunay_nn_source - delaunay_source
) / np.linalg.norm(delaunay_source)
delaunay_source_correlation = np.corrcoef(delaunay_nn_source, delaunay_source)[0, 1]

assert delaunay_source_relative_l2 < 5.0e-4
assert delaunay_source_correlation > 0.995


def public_interpolator_objective(mesh_points):
    interpolator = delaunay_nn_mesh.interpolator_from(
        source_plane_data_grid=data_grid,
        source_plane_mesh_grid=aa.Grid2DIrregular(values=mesh_points, xp=jnp),
        xp=jnp,
    )
    source_values = jnp.linspace(0.2, 1.8, mesh_points.shape[0])
    safe_mappings = jnp.maximum(interpolator.mappings, 0)
    mapped = source_values[safe_mappings] * interpolator.weights
    return jnp.sum(mapped**2)


public_value, public_mesh_gradient = jax.jit(
    jax.value_and_grad(public_interpolator_objective)
)(jnp.asarray(mesh_grid_9.array))
assert np.isfinite(float(public_value))
assert np.isfinite(np.asarray(public_mesh_gradient)).all()
assert float(jnp.linalg.norm(public_mesh_gradient)) > 0.0

# Exercise the public implementation's full JAX table contract, including
# the 4*N split points used by split regularization.
full_tables_jit = jax.jit(
    lambda mesh_points, queries: jax_delaunay_nn(
        mesh_points,
        queries,
        max_cavity_triangles=MAX_CAVITY_TRIANGLES,
        max_neighbors=MAX_NEIGHBORS,
        query_chunk=QUERY_CHUNK,
    )
)
full_tables = full_tables_jit(
    jnp.asarray(mesh_grid_9.array),
    jnp.asarray(data_grid.over_sampled.array),
)
jax.block_until_ready(full_tables)
assert full_tables[6].shape == (4 * mesh_grid_9.shape[0], MAX_NEIGHBORS)
assert not np.asarray(full_tables[10]).any()
assert not np.asarray(full_tables[11]).any()
assert not np.asarray(full_tables[13]).any()
assert not np.asarray(full_tables[14]).any()


def full_table_objective(mesh_points, queries):
    tables = jax_delaunay_nn(
        mesh_points,
        queries,
        max_cavity_triangles=MAX_CAVITY_TRIANGLES,
        max_neighbors=MAX_NEIGHBORS,
        query_chunk=QUERY_CHUNK,
    )
    source_values = jnp.linspace(0.2, 1.8, mesh_points.shape[0])
    mappings, weights = tables[2], tables[4]
    split_mappings, split_weights = tables[6], tables[8]
    mapped = source_values[jnp.maximum(mappings, 0)] * weights
    split_mapped = source_values[jnp.maximum(split_mappings, 0)] * split_weights
    return jnp.sum(mapped**2) + 0.1 * jnp.sum(split_mapped**2)


_, (mesh_gradient, query_gradient) = jax.jit(
    jax.value_and_grad(full_table_objective, argnums=(0, 1))
)(
    jnp.asarray(mesh_grid_9.array),
    jnp.asarray(data_grid.over_sampled.array),
)
assert np.isfinite(np.asarray(mesh_gradient)).all()
assert np.isfinite(np.asarray(query_gradient)).all()
assert float(jnp.linalg.norm(mesh_gradient)) > 0.0
assert float(jnp.linalg.norm(query_gradient)) > 0.0


def adaptive_mesh(count, rng):
    blob_count = count // 2
    blob = rng.normal(size=(blob_count, 2)) * 0.15
    angle = rng.uniform(0.0, 2.0 * np.pi, size=count - blob_count)
    radius = 1.0 + rng.normal(size=count - blob_count) * 0.12
    ring = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    return np.concatenate([blob, ring])


def barycentric_tables(points, query_points):
    simplices, neighbors, vertex_simplex = _jax_delaunay_tables(points)
    mappings = pix_indexes_delaunay_walk_from(
        query_points=query_points,
        points=points,
        simplices_padded=simplices,
        simplex_neighbors=neighbors,
        vertex_simplex=vertex_simplex,
        xp=jnp,
    )
    weights = pixel_weights_delaunay_from(
        data_grid=query_points,
        mesh_grid=points,
        pix_indexes_for_sub_slim_index=mappings,
        xp=jnp,
    )
    return mappings, weights


def sibson_tables(points, query_points):
    return jax_sibson(
        points,
        query_points,
        max_cavity_triangles=MAX_CAVITY_TRIANGLES,
        max_neighbors=MAX_NEIGHBORS,
        query_chunk=QUERY_CHUNK,
    )[2:]


def delaunay_full_tables(points, query_points):
    return jax_delaunay(points, query_points)


def delaunay_nn_full_tables(points, query_points):
    return jax_delaunay_nn(
        points,
        query_points,
        max_cavity_triangles=MAX_CAVITY_TRIANGLES,
        max_neighbors=MAX_NEIGHBORS,
        query_chunk=QUERY_CHUNK,
    )


def warm_times(function, points, query_points):
    compiled = jax.jit(function)
    start = time.perf_counter()
    output = compiled(points, query_points)
    jax.block_until_ready(output)
    compile_and_first_s = time.perf_counter() - start

    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        output = compiled(points, query_points)
        jax.block_until_ready(output)
        samples.append(time.perf_counter() - start)
    return output, compile_and_first_s, float(np.median(samples)), samples


def flip_points(offset):
    return jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0 + offset]])


flip_query = jnp.array([[0.35, 0.55]])
flip_values = jnp.array([0.0, 1.0, 2.0, 4.0])


def barycentric_flip_value(offset):
    mappings, weights = barycentric_tables(flip_points(offset), flip_query)
    return jnp.sum(flip_values[jnp.maximum(mappings, 0)] * weights)


def sibson_flip_value(offset):
    mappings, _, weights, *_ = sibson_tables(flip_points(offset), flip_query)
    return jnp.sum(flip_values[jnp.maximum(mappings, 0)] * weights)


rng = np.random.default_rng(4)
points = jnp.asarray(adaptive_mesh(POINT_COUNT, rng))
query_points = jnp.asarray(adaptive_mesh(QUERY_COUNT, rng))

barycentric_output, barycentric_compile_s, barycentric_warm_s, barycentric_runs = (
    warm_times(barycentric_tables, points, query_points)
)
sibson_output, sibson_compile_s, sibson_warm_s, sibson_runs = warm_times(
    sibson_tables, points, query_points
)
(
    delaunay_full_output,
    delaunay_full_compile_s,
    delaunay_full_warm_s,
    delaunay_full_runs,
) = warm_times(delaunay_full_tables, points, query_points)
(
    delaunay_nn_full_output,
    delaunay_nn_full_compile_s,
    delaunay_nn_full_warm_s,
    delaunay_nn_full_runs,
) = warm_times(delaunay_nn_full_tables, points, query_points)

mappings, sizes, weights, cavity_sizes, overflow, degenerate = sibson_output
assert not np.asarray(overflow).any(), "Sibson cavity cap overflowed"
assert not np.asarray(degenerate).any(), "Sibson Watson calculation was degenerate"
np.testing.assert_allclose(np.asarray(weights.sum(axis=1)), 1.0, atol=1.0e-11)
assert not np.asarray(delaunay_nn_full_output[10]).any()
assert not np.asarray(delaunay_nn_full_output[11]).any()
assert not np.asarray(delaunay_nn_full_output[13]).any()
assert not np.asarray(delaunay_nn_full_output[14]).any()


def arrays_identical(left, right):
    """Return whether two arrays match element-for-element, NaN included.

    Overflow and Watson degeneracy deliberately produce NaN weights, so a
    parity check that used ``==`` would pass on any pair of NaN tables and
    fail on a legitimately NaN one. Floating point is compared with
    ``equal_nan``; integer and boolean tables are compared exactly.
    """
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.inexact):
        return np.array_equal(left, right, equal_nan=True)
    return np.array_equal(left, right)


# Single-pass parity (issue #532). `jax_delaunay_nn` locates and interpolates
# the data grid and its own 4N split-cross points in ONE concatenated Sibson
# pass and slices the six outputs at `n_query`; that halves the kernel-launch
# count of a latency-bound program but must not move a single per-query row.
# Both halves are therefore compared with `jax_sibson` run separately on
# exactly the same coordinates -- the split points are taken from
# `jax_delaunay_nn`'s own return (index 5) so both sides see identical inputs
# -- with the same caps and the same chunk.
#
# The two functions return different tuple layouts, mapped explicitly here:
#   jax_sibson      -> (points, simplices, mappings, sizes, weights,
#                       cavity_sizes, overflow, degenerate)
#   jax_delaunay_nn -> (points, simplices, mappings, sizes, weights,
#                       split_points, split_mappings, split_sizes,
#                       split_weights, cavity_sizes, overflow, degenerate,
#                       split_cavity_sizes, split_overflow, split_degenerate)
# `sibson_tables` already returns `jax_sibson(...)[2:]`, i.e. the six outputs
# in the order below, and `sibson_output` is that call on `query_points`.
SIBSON_OUTPUT_NAMES = (
    "mappings",
    "sizes",
    "weights",
    "cavity_sizes",
    "overflow",
    "degenerate",
)
DATA_HALF_INDEXES = (2, 3, 4, 9, 10, 11)
SPLIT_HALF_INDEXES = (6, 7, 8, 12, 13, 14)

split_points_single_pass = delaunay_nn_full_output[5]
single_pass_parity_start = time.perf_counter()
split_output_separate_pass = jax.jit(sibson_tables)(points, split_points_single_pass)
jax.block_until_ready(split_output_separate_pass)

for name, index, separate in zip(SIBSON_OUTPUT_NAMES, DATA_HALF_INDEXES, sibson_output):
    assert arrays_identical(delaunay_nn_full_output[index], separate), (
        f"single-pass parity failed on the data half for {name}: the "
        "concatenated jax_delaunay_nn pass and a separate jax_sibson pass "
        "disagree"
    )
for name, index, separate in zip(
    SIBSON_OUTPUT_NAMES, SPLIT_HALF_INDEXES, split_output_separate_pass
):
    assert arrays_identical(delaunay_nn_full_output[index], separate), (
        f"single-pass parity failed on the split half for {name}: the "
        "concatenated jax_delaunay_nn pass and a separate jax_sibson pass "
        "disagree"
    )
single_pass_parity_s = time.perf_counter() - single_pass_parity_start


# Split-regularization compaction parity (PyAutoArray issue #536). The JAX
# assembly of the `ConstantSplit` matrix no longer scatters the full padded
# `(4P, K, K)` outer product: it scatters the first
# `SPLIT_REG_COMPACT_WIDTH` columns of every row and supplements the
# `SPLIT_REG_WIDE_ROW_BUDGET` widest rows with the blocks the compact pass did
# not cover. The result must stay exact, so it is checked against the NumPy
# builder both through the public `ConstantSplit` object and on the
# production-size split tables with a row forced wider than the compact width
# (the supplement path, which the production geometry itself never reaches).
compaction_parity_start = time.perf_counter()

constant_split = aa.reg.ConstantSplit(coefficient=1.0)

public_split_matrix_np = constant_split.regularization_matrix_from(
    linear_obj=delaunay_nn_mapper, xp=np
)
public_split_matrix_jax = np.asarray(
    constant_split.regularization_matrix_from(linear_obj=delaunay_nn_mapper, xp=jnp)
)
np.testing.assert_allclose(
    public_split_matrix_jax, public_split_matrix_np, rtol=1.0e-10, atol=1.0e-14
)


def split_regularization_matrix_from(mappings, sizes, weights, xp, **kwargs):
    """The `ConstantSplit.regularization_matrix_from` chain, on explicit tables."""
    (
        splitted_mappings,
        splitted_sizes,
        splitted_weights,
    ) = regularization_util.reg_split_from(
        splitted_mappings=mappings,
        splitted_sizes=sizes,
        splitted_weights=weights,
        xp=xp,
    )
    pixels = len(splitted_mappings) // 4
    return regularization_util.pixel_splitted_regularization_matrix_from(
        regularization_weights=xp.full(fill_value=1.0, shape=(pixels,)),
        splitted_mappings=splitted_mappings,
        splitted_sizes=splitted_sizes,
        splitted_weights=splitted_weights,
        xp=xp,
        **kwargs,
    )


split_mappings_np = np.asarray(delaunay_nn_full_output[6]).astype(np.int32).copy()
split_sizes_np = np.asarray(delaunay_nn_full_output[7]).astype(np.int32).copy()
split_weights_np = np.asarray(delaunay_nn_full_output[8]).astype(np.float64).copy()

SPLIT_TABLE_WIDTH = split_mappings_np.shape[1]
FORCED_WIDE_SIZE = SPLIT_REG_COMPACT_WIDTH + 8

assert SPLIT_TABLE_WIDTH > FORCED_WIDE_SIZE
assert int(split_sizes_np.max()) <= SPLIT_REG_COMPACT_WIDTH, (
    "the production split geometry is already wider than the compact width; the "
    "forced-wide-row check below no longer isolates the supplement path"
)

forced_row = int(np.argmax(split_sizes_np))
forced_pixels = np.arange(FORCED_WIDE_SIZE) % POINT_COUNT
forced_weights = np.linspace(0.05, 1.0, FORCED_WIDE_SIZE)
split_mappings_np[forced_row] = -1
split_weights_np[forced_row] = 0.0
split_mappings_np[forced_row, :FORCED_WIDE_SIZE] = forced_pixels
split_weights_np[forced_row, :FORCED_WIDE_SIZE] = forced_weights / forced_weights.sum()
split_sizes_np[forced_row] = FORCED_WIDE_SIZE

wide_row_count = int((split_sizes_np > SPLIT_REG_COMPACT_WIDTH).sum())
assert wide_row_count >= 1

split_matrix_np = split_regularization_matrix_from(
    split_mappings_np.copy(), split_sizes_np.copy(), split_weights_np.copy(), xp=np
)

split_matrix_jax_jit = jax.jit(
    lambda mappings, sizes, weights: split_regularization_matrix_from(
        mappings, sizes, weights, xp=jnp
    )
)
split_matrix_jax = np.asarray(
    split_matrix_jax_jit(
        jnp.asarray(split_mappings_np),
        jnp.asarray(split_sizes_np),
        jnp.asarray(split_weights_np),
    )
)
np.testing.assert_allclose(
    split_matrix_jax, split_matrix_np, rtol=1.0e-10, atol=1.0e-14
)
assert np.isfinite(split_matrix_jax).all()

# The same tables through the uncompacted scatter (compact width = the table
# width), i.e. the pre-#536 path, as a direct old-versus-new comparison that
# does not go through the NumPy builder.
split_matrix_uncompacted = np.asarray(
    jax.jit(
        lambda mappings, sizes, weights: split_regularization_matrix_from(
            mappings, sizes, weights, xp=jnp, compact_width=SPLIT_TABLE_WIDTH
        )
    )(
        jnp.asarray(split_mappings_np),
        jnp.asarray(split_sizes_np),
        jnp.asarray(split_weights_np),
    )
)
np.testing.assert_allclose(
    split_matrix_jax, split_matrix_uncompacted, rtol=1.0e-12, atol=1.0e-14
)

# Budget overflow poisons the matrix with NaN (the Sibson cap convention)
# rather than silently truncating the wide rows.
split_matrix_overflow = np.asarray(
    jax.jit(
        lambda mappings, sizes, weights: split_regularization_matrix_from(
            mappings, sizes, weights, xp=jnp, wide_row_budget=wide_row_count - 1
        )
    )(
        jnp.asarray(split_mappings_np),
        jnp.asarray(split_sizes_np),
        jnp.asarray(split_weights_np),
    )
)
assert np.isnan(split_matrix_overflow).all()

compaction_parity_s = time.perf_counter() - compaction_parity_start

# Chunk invariance. `query_chunk` sets the `jax.lax.map` block size and is a
# memory guard on the per-cavity intermediates only, so every value must give
# bit-identical tables. `sibson_output` is already the QUERY_CHUNK leg, so
# only the other chunks are recomputed.
CHUNK_INVARIANCE_CHUNKS = (64, 256, 1024)
chunk_invariance_start = time.perf_counter()
chunk_invariance_outputs = {QUERY_CHUNK: sibson_output}
for chunk in CHUNK_INVARIANCE_CHUNKS:
    if chunk in chunk_invariance_outputs:
        continue
    chunk_output = jax.jit(
        lambda mesh_points, queries, chunk=chunk: jax_sibson(
            mesh_points,
            queries,
            max_cavity_triangles=MAX_CAVITY_TRIANGLES,
            max_neighbors=MAX_NEIGHBORS,
            query_chunk=chunk,
        )[2:]
    )(points, query_points)
    jax.block_until_ready(chunk_output)
    chunk_invariance_outputs[chunk] = chunk_output

for chunk in CHUNK_INVARIANCE_CHUNKS:
    for name, reference, candidate in zip(
        SIBSON_OUTPUT_NAMES, sibson_output, chunk_invariance_outputs[chunk]
    ):
        assert arrays_identical(reference, candidate), (
            f"query_chunk={chunk} changed {name}: the chunk is a memory guard "
            "and must not alter a single output"
        )
chunk_invariance_s = time.perf_counter() - chunk_invariance_start

# The chunk is also settable without a source edit, for the accelerator sweep.
# `PYAUTO_SIBSON_QUERY_CHUNK` is read once at import, so the override can only
# be proven in a fresh interpreter -- and only if this process did not itself
# inherit the variable.
assert os.environ.get("PYAUTO_SIBSON_QUERY_CHUNK") is None, (
    "unset PYAUTO_SIBSON_QUERY_CHUNK before running this gate: it is read at "
    "import time, so a value inherited here would silence the override check"
)
assert aa.mesh.DelaunayNN.query_chunk == SIBSON_QUERY_CHUNK

chunk_override_start = time.perf_counter()
override_environment = dict(os.environ, PYAUTO_SIBSON_QUERY_CHUNK="64")
override_process = subprocess.run(
    [
        sys.executable,
        "-c",
        "import autoarray as aa;"
        "from autoarray.inversion.mesh.interpolator.sibson import"
        " SIBSON_QUERY_CHUNK;"
        "print(aa.mesh.DelaunayNN.query_chunk, SIBSON_QUERY_CHUNK)",
    ],
    capture_output=True,
    text=True,
    env=override_environment,
    check=True,
)
override_chunks = override_process.stdout.strip().splitlines()[-1].split()
assert override_chunks == ["64", "64"], (
    "PYAUTO_SIBSON_QUERY_CHUNK=64 did not reach DelaunayNN.query_chunk; the "
    f"subprocess printed {override_process.stdout!r}"
)
chunk_override_s = time.perf_counter() - chunk_override_start

# Natural-neighbour coordinates reproduce every affine field exactly.  This
# simultaneously checks the weights and their JAX derivative with respect to
# the moving query coordinates.
source_values = 2.0 * points[:, 0] - 3.0 * points[:, 1] + 0.7
gradient_vertex_indexes = rng.integers(0, POINT_COUNT, size=(64, 3))
gradient_coefficients = rng.dirichlet(np.ones(3), size=64)
gradient_query_points = jnp.asarray(
    np.sum(
        np.asarray(points)[gradient_vertex_indexes] * gradient_coefficients[:, :, None],
        axis=1,
    )
)


def linear_sum(query):
    linear_mappings, _, linear_weights, *_ = sibson_tables(points, query)
    safe_mappings = jnp.maximum(linear_mappings, 0)
    return jnp.sum(source_values[safe_mappings] * linear_weights)


gradient = jax.grad(linear_sum)(gradient_query_points)
expected_gradient = jnp.broadcast_to(jnp.array([2.0, -3.0]), gradient.shape)
np.testing.assert_allclose(
    np.asarray(gradient), np.asarray(expected_gradient), atol=1.0e-9
)

epsilon = 1.0e-7
barycentric_value_and_grad = jax.jit(jax.value_and_grad(barycentric_flip_value))
sibson_value_and_grad = jax.jit(jax.value_and_grad(sibson_flip_value))
barycentric_left = barycentric_value_and_grad(jnp.asarray(-epsilon))
barycentric_right = barycentric_value_and_grad(jnp.asarray(epsilon))
sibson_left = sibson_value_and_grad(jnp.asarray(-epsilon))
sibson_right = sibson_value_and_grad(jnp.asarray(epsilon))

vmap_offsets = jnp.array([-2.0e-3, -1.0e-3, 1.0e-3, 2.0e-3])
vmap_values = jax.jit(jax.vmap(sibson_flip_value))(vmap_offsets)
assert np.isfinite(np.asarray(vmap_values)).all()

assert abs(float(barycentric_left[0] - barycentric_right[0])) > 0.1
np.testing.assert_allclose(
    np.asarray(sibson_left), np.asarray(sibson_right), rtol=1.0e-5, atol=1.0e-6
)

print(f"device: {jax.devices()[0]}")
print(
    "source reconstruction parity: "
    f"relative_l2={delaunay_source_relative_l2:.6e} "
    f"correlation={delaunay_source_correlation:.9f}"
)
print(f"shape: {POINT_COUNT} mesh points x {QUERY_COUNT} queries")
print(
    "barycentric: "
    f"compile+first={barycentric_compile_s:.6f}s "
    f"warm_median={barycentric_warm_s:.6f}s runs={barycentric_runs}"
)
print(
    "sibson: "
    f"compile+first={sibson_compile_s:.6f}s "
    f"warm_median={sibson_warm_s:.6f}s runs={sibson_runs}"
)
print(f"sibson/barycentric warm ratio: {sibson_warm_s / barycentric_warm_s:.3f}x")
print(
    "delaunay full mapper: "
    f"compile+first={delaunay_full_compile_s:.6f}s "
    f"warm_median={delaunay_full_warm_s:.6f}s runs={delaunay_full_runs}"
)
print(
    "delaunay_nn full mapper: "
    f"compile+first={delaunay_nn_full_compile_s:.6f}s "
    f"warm_median={delaunay_nn_full_warm_s:.6f}s runs={delaunay_nn_full_runs}"
)
print(
    "delaunay_nn/delaunay full mapper ratio: "
    f"{delaunay_nn_full_warm_s / delaunay_full_warm_s:.3f}x"
)
print(
    "single-pass parity: "
    f"data half {len(SIBSON_OUTPUT_NAMES)}/{len(SIBSON_OUTPUT_NAMES)} identical "
    f"({QUERY_COUNT} queries), "
    f"split half {len(SIBSON_OUTPUT_NAMES)}/{len(SIBSON_OUTPUT_NAMES)} identical "
    f"({split_points_single_pass.shape[0]} split points) "
    f"in {single_pass_parity_s:.3f}s"
)
print(
    "split-regularization compaction parity: "
    f"public ConstantSplit max|jax-numpy|={np.abs(public_split_matrix_jax - public_split_matrix_np).max():.3e}, "
    f"production tables ({split_mappings_np.shape[0]} rows x {SPLIT_TABLE_WIDTH}) "
    f"max|jax-numpy|={np.abs(split_matrix_jax - split_matrix_np).max():.3e} "
    f"with {wide_row_count} row(s) wider than the compact width "
    f"{SPLIT_REG_COMPACT_WIDTH}; uncompacted scatter identical to "
    f"{np.abs(split_matrix_jax - split_matrix_uncompacted).max():.3e}; "
    f"budget {wide_row_count - 1} -> NaN "
    f"in {compaction_parity_s:.3f}s"
)
print(
    "chunk invariance: "
    f"chunks {CHUNK_INVARIANCE_CHUNKS} identical on "
    f"{len(SIBSON_OUTPUT_NAMES)}/{len(SIBSON_OUTPUT_NAMES)} outputs "
    f"in {chunk_invariance_s:.3f}s; "
    f"PYAUTO_SIBSON_QUERY_CHUNK=64 -> DelaunayNN.query_chunk=64 "
    f"(default {SIBSON_QUERY_CHUNK}) in {chunk_override_s:.3f}s"
)
print(
    "sibson diagnostics: "
    f"max_cavity={int(np.asarray(cavity_sizes).max())} "
    f"max_neighbors={int(np.asarray(sizes).max())}"
)
print(
    "flip values/gradients: "
    f"barycentric left={tuple(map(float, barycentric_left))} "
    f"right={tuple(map(float, barycentric_right))}; "
    f"sibson left={tuple(map(float, sibson_left))} "
    f"right={tuple(map(float, sibson_right))}"
)
