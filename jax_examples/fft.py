import numpy as np

import autolens as al

"""
__Data__
"""
data = al.Visibilities.full(shape_slim=(7,), fill_value=1.0)
noise_map = al.VisibilitiesNoiseMap.full(shape_slim=(7,), fill_value=2.0)

uv_wavelengths = np.array(
        [
            [-55636.4609375, 171376.90625],
            [-6903.21923828, 51155.578125],
            [-63488.4140625, 4141.28369141],
            [55502.828125, 47016.7265625],
            [54160.75390625, -99354.1796875],
            [-9327.66308594, -95212.90625],
            [0.0, 0.0],
        ]
    )

mask = np.array(
    [
        [True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True],
        [True, True, False, False, False, True, True],
        [True, True, False, False, False, True, True],
        [True, True, False, False, False, True, True],
        [True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True],
    ]
)

mask = al.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

interferometer = al.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=mask,
        transformer_class=al.TransformerNUFFT,
    )


"""
__PyNUFFT__
"""
image_pynufft = interferometer.transformer.image_from(visibilities=data)

image = al.Array2D(
    values=[
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    mask=mask,
)

visibilities_pynufft = interferometer.transformer.visibilities_from(image=image)

print(image_pynufft)
print()
print(visibilities_pynufft)
print()
print(interferometer.w_tilde.dirty_image)
print("----")

"""
__
"""