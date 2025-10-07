import numpy as np
import jax.numpy as jnp
import time
import os


class SersicProfile:
    def __init__(self, intensity=1.0, effective_radius=1.0, sersic_index=4.0):
        self._intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index
        self.sersic_constant = 2.0 * self.sersic_index - 0.327  # crude approx

    def image_2d_via_radii_from(self, grid_radii: np.ndarray) -> np.ndarray:
        """
        Returns the 2D image of the Sersic light profile from a grid of coordinates
        which are the radial distances of each coordinate from its centre.
        """
        return jnp.multiply(
            self._intensity,
            jnp.exp(
                jnp.multiply(
                    -self.sersic_constant,
                    jnp.add(
                        jnp.power(
                            jnp.divide(grid_radii, self.effective_radius),
                            1.0 / self.sersic_index,
                        ),
                        -1,
                    ),
                )
            ),
        )


# -------------------
# Profiling experiment
# -------------------


def profile_run(size=5000, repeats=5):
    # Create a big fake radial grid
    grid = np.abs(np.random.randn(size, size))
    profile = SersicProfile()

    # Warmup
    profile.image_2d_via_radii_from(grid)

    # Time runs
    times = []
    for _ in range(repeats):
        t0 = time.time()
        _ = profile.image_2d_via_radii_from(grid)
        times.append(time.time() - t0)

    return np.mean(times), np.std(times)


if __name__ == "__main__":
    print("Num threads (check via env):")
    print("  OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
    print("  MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"))
    print("  OPENBLAS_NUM_THREADS =", os.environ.get("OPENBLAS_NUM_THREADS"))

    mean, std = profile_run(size=4000, repeats=3)
    print(f"Runtime (mean ± std): {mean:.3f} ± {std:.3f} s")
