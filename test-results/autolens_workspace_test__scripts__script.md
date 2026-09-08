# Test Report: autolens_workspace_test / scripts (script)

**29 scripts** | 3 failed | 26 passed

| Status | Count |
|--------|-------|
| failed | 3 |
| passed | 26 |

## Failures

### `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/imaging/jax_likelihood/lp.py` — FAILED (6.9s)

Command '['python3', '/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/imaging/jax_likelihood/lp.py']' returned non-zero exit status 1.

```
Traceback (most recent call last):
  File "/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/imaging/jax_likelihood/lp.py", line 264, in <module>
    np.testing.assert_allclose(
  File "/home/jammy/venv/PyAuto/lib/python3.12/site-packages/numpy/testing/_private/utils.py", line 1715, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "/home/jammy/venv/PyAuto/lib/python3.12/site-packages/numpy/testing/_private/utils.py", line 921, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Not equal to tolerance rtol=0.0001, atol=0
lp: JAX vmap likelihood mismatch
Mismatched elements: 10 / 10 (100%)
Max absolute difference among violations: 8.22965961
Max relative difference among violations: 0.01331673
 ACTUAL: array([626.223585, 626.223585, 626.223585, 626.223585, 626.223585,
       626.223585, 626.223585, 626.223585, 626.223585, 626.223585])
 DESIRED: array(617.993925)
```

### `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/imaging/jax_likelihood/smbh.py` — FAILED (6.2s)

Command '['python3', '/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/imaging/jax_likelihood/smbh.py']' returned non-zero exit status 1.

```
Traceback (most recent call last):
  File "/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/imaging/jax_likelihood/smbh.py", line 227, in <module>
    np.testing.assert_allclose(
  File "/home/jammy/venv/PyAuto/lib/python3.12/site-packages/numpy/testing/_private/utils.py", line 1715, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "/home/jammy/venv/PyAuto/lib/python3.12/site-packages/numpy/testing/_private/utils.py", line 921, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Not equal to tolerance rtol=0.0001, atol=0
smbh: JAX vmap likelihood mismatch
Mismatched elements: 10 / 10 (100%)
Max absolute difference among violations: 5.02046586
Max relative difference among violations: 0.00809375
 ACTUAL: array([625.30935, 625.30935, 625.30935, 625.30935, 625.30935, 625.30935,
       625.30935, 625.30935, 625.30935, 625.30935])
 DESIRED: array(620.288884)
```

### `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/multi_dataset/jax_likelihood/mge.py` — FAILED (16.7s)

Command '['python3', '/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/multi_dataset/jax_likelihood/mge.py']' returned non-zero exit status 1.

```
Traceback (most recent call last):
  File "/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/multi_dataset/jax_likelihood/mge.py", line 191, in <module>
    np.testing.assert_allclose(
  File "/home/jammy/venv/PyAuto/lib/python3.12/site-packages/numpy/testing/_private/utils.py", line 1715, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "/home/jammy/venv/PyAuto/lib/python3.12/site-packages/numpy/testing/_private/utils.py", line 921, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Not equal to tolerance rtol=0.0001, atol=0
multi_dataset/mge: JAX vmap likelihood mismatch
Mismatched elements: 3 / 3 (100%)
Max absolute difference among violations: 45.06605143
Max relative difference among violations: 0.00150204
 ACTUAL: array([-29958.208093, -29958.208093, -29958.208093])
 DESIRED: array(-30003.274145)
```

## Passed

- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/imaging/jax_likelihood/delaunay.py` (24.3s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/imaging/jax_likelihood/rectangular.py` (24.4s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/imaging/jax_likelihood/mge.py` (20.4s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/interferometer/jax_likelihood/rectangular.py` (23.7s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/interferometer/jax_likelihood/mge.py` (14.4s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/point_source/jax_likelihood/point.py` (28.1s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/misc/jax_assertions/delaunay_nn.py` (34.0s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/misc/jax_assertions/delaunay_nn_caps.py` (23.5s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/misc/jax_assertions/delaunay_walk.py` (16.6s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/misc/jax_assertions/fit_imaging_sparse_operator.py` (4.9s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/misc/jax_assertions/fit_interferometer_sparse_operator.py` (4.0s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/interferometer/datacube/shared_preloads.py` (19.4s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/multi_dataset/jax_likelihood/shared_preloads.py` (27.0s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/imaging/jax_likelihood/potential_correction.py` (7.3s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/interferometer/jax_likelihood/potential_correction.py` (11.6s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/imaging/subhalo_recovery.py` (29.4s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/interferometer/subhalo_recovery_interferometer.py` (10.5s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/misc/aggregator/fit_imaging.py` (10.0s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/misc/aggregator/fit_interferometer.py` (7.6s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/misc/aggregator/tracer.py` (6.5s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/multi_galaxy/composition_mge.py` (4.9s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/multi_galaxy/model_fit.py` (5.0s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/multi_galaxy/jax_likelihood/lp.py` (11.0s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/misc/latent/latent_variables_smoke.py` (4.4s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/misc/latent/latent_nan_robustness_jax.py` (4.3s)
- `/home/jammy/Code/PyAutoLabs-wt/workspace-lp-sub-size-1-retire/autolens_workspace_test/scripts/imaging/over_sample_adapt_snr.py` (4.2s)
