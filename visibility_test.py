import autolens as al
import numpy as np

data = np.array([6.0 + 3.0j, 7.0 + 4.0j])
model_data = np.array([9.0 + 100.0j, 2.0 + 2.0j])
noise_map = np.array([1.0 + 4.0j, 8.0 + 7.0j])

residual1=(data-model_data)

print(residual1)

normalized_res1=residual1.real/noise_map.real+1j*residual1.imag/noise_map.imag

print(normalized_res1)

import autoarray as aa

residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

print(residual_map)

normalized_res1 = aa.util.fit.normalized_residual_map_complex_with_mask_from(
            residual_map=residual_map, noise_map=noise_map, mask=np.array([False, False])
        )

print(normalized_res1)

# real_space_mask =
#
# interferometer = aa.Interferometer(visibilities=data, noise_map=noise_map, uv_wavelengths=None, real_space_mask=None)
#
# fit = aa.FitInterferometer(interferometer=interferometer, model_visibilities=model_data)

