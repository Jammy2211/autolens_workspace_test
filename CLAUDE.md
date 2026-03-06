# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

**autolens_workspace_test** is the integration test suite for PyAutoLens. It contains Python scripts that are run on the build server to verify that the core PyAutoLens functionality works end-to-end. It is not a user-facing workspace — see `../autolens_workspace` for example scripts and tutorials.

Dependencies: `autolens`, `autogalaxy`, `autofit`, `numba`. Python version: 3.11.

## Workspace Structure

```
scripts/                     Integration test scripts run on the build server
  imaging/                   CCD imaging model-fit tests
  interferometer/            Interferometer model-fit tests
  point_source/              Point source model-fit tests
  jax_likelihood_functions/  JAX likelihood function tests (imaging, interferometer, point_source)
failed/                      Failure logs written here when a script errors (one .txt per failure)
dataset/                     Input .fits files and example data
config/                      YAML configuration files
output/                      Model-fit results written here at runtime
```

## Running Tests

Scripts are run from the repository root **without** `PYAUTOFIT_TEST_MODE=1` — the non-linear searches run for real (using sampler limits like `n_like_max` to keep runtimes short):

```bash
python scripts/imaging/model_fit.py
```

To run all tests and log failures to `failed/`:

```bash
bash run_all_scripts.sh
```

Each failed script produces a `.txt` file in `failed/` named after the script path (with `/` replaced by `__`), containing the exit code and full output.

Unlike `../autolens_workspace`, there is no resume/skip logic — every run executes all scripts in `scripts/` from scratch.

## Integration Test Runner

`run_all_scripts.sh` at the repo root:
- Finds all `*.py` files under `scripts/` and runs them in order (no test mode flag)
- On failure: writes a log to `failed/<script_path_with_slashes_replaced>.txt`
- Does not skip previously-run scripts (stateless, always runs all)

## jax_likelihood_functions

`scripts/jax_likelihood_functions/` contains scripts that test JAX can successfully compute log-likelihood gradients for various model types:
- `imaging/` — light parametric, MGE, Delaunay, rectangular pixelization tests
- `interferometer/` — interferometer likelihood tests
- `point_source/` — point source likelihood tests

These were originally in `jax_profiling/` at the repo root and moved here so they are included in the standard `run_all_scripts.sh` test run.

## Line Endings — Always Unix (LF)

All files **must use Unix line endings (LF, `\n`)**. Never write `\r\n` line endings.
