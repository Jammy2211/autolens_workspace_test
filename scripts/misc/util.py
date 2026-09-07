"""
Shared utilities for the ``jax_grad`` scripts: compare ``jax.grad`` of a scalar
likelihood function against central finite differences, parameter by parameter.

The scripts in this folder previously only asserted that gradients are finite
and non-zero. Finiteness does not catch wrong gradients (e.g. terms silently
dropped by ``lax.stop_gradient``, zeroed cotangents through frozen structures,
or incorrect custom VJPs). These helpers make correctness checkable:

- ``fd_gradient`` — central finite differences of ``f`` with a per-parameter
  step scaled to the parameter's magnitude. Requires float64 (the default in
  this stack) — float32 finite differences are dominated by round-off.
- ``compare_gradients`` — evaluates autodiff and finite differences, prints a
  per-parameter table and returns the arrays plus error metrics.
- ``assert_gradients_match`` — the assertion used by the scripts. A parameter
  passes if ``|ad - fd| <= atol + rtol * max(|ad|, |fd|)``. Parameters whose
  comparison is knowingly unreliable (an intentionally approximate gradient,
  or finite differences that cross a documented discontinuity) can be
  excluded via ``skip_indices`` — the point is that every exclusion is
  explicit and visible in the calling script, never hidden by a loose global
  tolerance.

Evaluation honesty: ``f`` is evaluated eagerly (no ``jax.jit``) by default.
Under a single JIT trace, ``jax.pure_callback`` results can be constant-folded
into the compiled program, which fakes both values and (zero) gradients. If a
jitted ``f`` is passed for speed, call ``assert_eager_jit_consistent`` first
with the un-jitted function so the two agree at the base point.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Drive jax.value_and_grad + finite-difference gradient checks; need JAX
enabled and full-resolution float64 data.

ENV: jax full_datasets
"""

import numpy as np
import jax
import jax.numpy as jnp


def parameter_names_from(model):
    """
    Human-readable parameter names for a PyAutoFit model, used to label the
    comparison table. Falls back to indices if the API is unavailable.
    """
    try:
        return list(model.model_component_and_parameter_names)
    except AttributeError:
        return None


def fd_gradient(f, x, rel_step=1e-5, abs_floor=0.1):
    """
    Central finite-difference gradient of the scalar function ``f`` at ``x``.

    The step for parameter ``i`` is ``rel_step * max(|x_i|, abs_floor)`` so
    that parameters of very different magnitude (ell_comps ~0.05, Einstein
    radius ~1.6) each get a sensibly scaled perturbation.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.dtype != np.float64:
        raise ValueError("fd_gradient requires float64 parameters.")
    grad = np.zeros_like(x)
    for i in range(x.size):
        h = rel_step * max(abs(x[i]), abs_floor)
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        f_plus = float(f(jnp.array(x_plus)))
        f_minus = float(f(jnp.array(x_minus)))
        grad[i] = (f_plus - f_minus) / (2.0 * h)
    return grad


def compare_gradients(
    f,
    x,
    param_names=None,
    rel_step=1e-5,
    abs_floor=0.1,
    f_fd=None,
    rel_steps=None,
):
    """
    Compute autodiff and finite-difference gradients of ``f`` at ``x`` and
    print a per-parameter comparison table.

    ``f_fd`` optionally supplies a faster (e.g. jitted) function for the
    ``2 * n_params`` finite-difference evaluations while autodiff runs on the
    eager ``f``; guard it with ``assert_eager_jit_consistent`` first.

    ``rel_steps`` switches to **FD-step-sweep mode**: finite differences are
    computed at every step in the tuple and, per parameter, the FD closest to
    autodiff is used for the comparison (the full sweep matrix is printed).
    This exists because pixelized-source likelihoods contain measure-thin
    solver branch flips — probed 2026-07-10 on the kernel-CDF meshes: single
    float inputs (width < 1e-15 in the parameter) where the positive-only
    solver converges to a marginally different solution (ΔLL ~1.6e-3 on the
    interferometer sparse config, up to ~14 on imaging 28×28), pseudo-randomly
    poisoning individual FD evaluations while the surface is smooth (LL exactly
    linear over ±2e-8 elsewhere). Any single step therefore fails sporadically;
    the sweep is still falsifiable — clean FD steps converge to the true
    gradient (observed 1e-6..1e-9 relative), so a *wrong* autodiff fails at
    every clean step, not just an unlucky one.

    Returns a dict with ``ad``, ``fd``, ``abs_err`` and ``rel_err`` arrays,
    where ``rel_err = abs_err / max(|ad|, |fd|)`` (0 where both are 0).
    """
    x = jnp.asarray(x)
    ad = np.array(jax.grad(f)(x))
    if rel_steps is None:
        fd = fd_gradient(
            f_fd if f_fd is not None else f,
            np.array(x),
            rel_step=rel_step,
            abs_floor=abs_floor,
        )
    else:
        fd_all = np.stack(
            [
                fd_gradient(
                    f_fd if f_fd is not None else f,
                    np.array(x),
                    rel_step=r,
                    abs_floor=abs_floor,
                )
                for r in rel_steps
            ]
        )
        best = np.argmin(np.abs(fd_all - ad[None, :]), axis=0)
        fd = fd_all[best, np.arange(len(ad))]
        print(f"\nFD step sweep (rel_steps={rel_steps}; * = used for comparison):")
        for i in range(len(ad)):
            cells = [
                f"{'*' if s == best[i] else ' '}{fd_all[s, i]:>14.6e}"
                for s in range(len(rel_steps))
            ]
            print(f"  p[{i:>2}] ad={ad[i]:>14.6e}  fd: {'  '.join(cells)}")

    abs_err = np.abs(ad - fd)
    denom = np.maximum(np.abs(ad), np.abs(fd))
    rel_err = np.divide(abs_err, denom, out=np.zeros_like(abs_err), where=denom > 0)

    if param_names is None:
        param_names = [f"p[{i}]" for i in range(len(ad))]
    name_width = max(len(name) for name in param_names)

    print(
        f"\n{'parameter':<{name_width}}  {'autodiff':>14}  {'finite diff':>14}"
        f"  {'abs err':>10}  {'rel err':>10}"
    )
    for name, a, d, ae, re in zip(param_names, ad, fd, abs_err, rel_err):
        print(f"{name:<{name_width}}  {a:>14.6e}  {d:>14.6e}  {ae:>10.3e}  {re:>10.3e}")

    return {"ad": ad, "fd": fd, "abs_err": abs_err, "rel_err": rel_err}


def assert_gradients_match(comparison, rtol=1e-3, atol=1e-4, skip_indices=()):
    """
    Assert autodiff and finite differences agree parameter-wise:
    ``|ad - fd| <= atol + rtol * max(|ad|, |fd|)``.

    ``skip_indices`` names parameters whose autodiff-vs-FD comparison is
    knowingly unreliable (for example, an approximate autodiff rule or FD
    samples that cross a documented discontinuity). Each exclusion must be
    justified by a comment at the call site. The skipped parameters are still
    printed by ``compare_gradients`` so the deviation stays measured and
    visible.
    """
    ad, fd, abs_err = comparison["ad"], comparison["fd"], comparison["abs_err"]
    tol = atol + rtol * np.maximum(np.abs(ad), np.abs(fd))
    failures = [
        i for i in range(len(ad)) if i not in set(skip_indices) and abs_err[i] > tol[i]
    ]
    assert not failures, (
        f"Autodiff vs finite-difference mismatch at parameter indices {failures}: "
        f"ad={ad[failures]}, fd={fd[failures]}, abs_err={abs_err[failures]}, "
        f"tolerance={tol[failures]}"
    )


def assert_eager_jit_consistent(f_eager, f_jit, x, rtol=1e-10):
    """
    Guard against ``pure_callback`` constant-folding: a jitted likelihood must
    agree with the eager likelihood at the same point before its gradients can
    be trusted for comparison work.
    """
    v_eager = float(f_eager(jnp.asarray(x)))
    v_jit = float(f_jit(jnp.asarray(x)))
    assert np.isclose(v_eager, v_jit, rtol=rtol), (
        f"Eager ({v_eager}) and jitted ({v_jit}) evaluations disagree — "
        "possible pure_callback constant-folding; do not trust jitted gradients."
    )


def assert_mesh_callback_not_constant_folded(
    f_eager, f_jit, x, x_perturbed, *, min_abs_diff, rtol=1e-6
):
    """
    Test the ``pure_callback`` constant-folding hazard **directly**, for
    likelihoods whose mesh connectivity comes from a host callback (qhull
    Delaunay tables, KNN neighbour lists).

    The hazard: under a single JIT trace the host callback's int32 tables can
    be baked into the compiled program, so every subsequent evaluation of the
    jitted function reuses the *trace point's* triangulation while the traced
    vertex positions keep moving. The jitted value is then wrong away from the
    trace point, and its gradients are worthless — but it is still finite,
    still parameter-dependent and still smooth, so no self-consistency check
    on the jitted function alone can see it.

    Measured on ``imaging/jax_grad/delaunay.py`` (2026-09-07, jax 0.10.2) by
    freezing the qhull callback's return value at the trace point:

    - ``|f_jit(x) - f_jit(x_perturbed)|`` stayed large (1.4e2 .. 1.4e3) with
      the tables frozen, i.e. asserting only that the two jitted values
      *differ* does not detect the hazard — a frozen table does not flatten
      the likelihood, it corrupts it.
    - ``f_eager`` vs ``f_jit`` **at the perturbed point** separated by
      8.3e-3 .. 2.7e-1 relative when frozen, against 2.2e-10 .. 3.5e-9 when
      honest — six to eight orders of margin, and no tolerance to calibrate
      against float64 reassociation noise.

    So the check is: evaluate both functions at ``x_perturbed``, a point whose
    triangulation genuinely differs from ``x``'s, and require they agree.
    ``f_eager`` recomputes the tables on every call by construction, so it is
    the reference.

    ``min_abs_diff`` guards the test against being vacuous: the eager
    likelihood must move by more than this floor between ``x`` and
    ``x_perturbed``, otherwise the perturbation is too small to re-triangulate
    the mesh and agreement at ``x_perturbed`` would prove nothing. Choose
    ``x_perturbed`` by perturbing the parameters that *define* the mesh (for a
    ray-traced source mesh, the mass model), by enough to re-wire the
    triangulation — a few per cent on the Einstein radius rewires >90% of the
    simplex table on the production Delaunay meshes.
    """
    # Call the jitted function at ``x`` FIRST. Whichever point it is first
    # executed at is the point a constant-folded callback would freeze its
    # tables at, so this pins that point to ``x`` — the base point the rest of
    # the script works at. Without it the perturbed evaluation below would be
    # the trace point, the tables would freeze there, and the comparison would
    # agree by construction (observed 2026-09-07: the check passed under an
    # injected fold until this call was added).
    float(f_jit(jnp.asarray(x)))

    v_eager = float(f_eager(jnp.asarray(x)))
    v_eager_perturbed = float(f_eager(jnp.asarray(x_perturbed)))

    shift = abs(v_eager_perturbed - v_eager)
    assert shift > min_abs_diff, (
        f"Vacuous constant-folding test: the perturbation moved the eager "
        f"likelihood by only {shift} (floor {min_abs_diff}), so it likely "
        "leaves the triangulation intact. Perturb the mesh-defining "
        "parameters harder."
    )

    v_jit_perturbed = float(f_jit(jnp.asarray(x_perturbed)))
    assert np.isclose(v_eager_perturbed, v_jit_perturbed, rtol=rtol), (
        f"Eager ({v_eager_perturbed}) and jitted ({v_jit_perturbed}) "
        "evaluations disagree at a point whose mesh differs from the JIT "
        "trace point — the host mesh callback has been constant-folded into "
        "the compiled program (its tables are frozen at the trace point). Do "
        "not trust jitted values or gradients away from that point."
    )
