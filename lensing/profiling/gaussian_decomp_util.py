import numpy as np
from scipy.special import wofz, comb
import autolens as al


def kesi(p):
    """
    see Eq.(6) of 1906.08263
    """
    n_list = np.arange(0, 2 * p + 1, 1)
    return (2.0 * p * np.log(10) / 3.0 + 2.0 * np.pi * n_list * 1j) ** (0.5)


def eta(p):
    """
    see Eq.(6) of 1906.00263
    """
    eta_list = np.zeros(int(2 * p + 1))
    kesi_list = np.zeros(int(2 * p + 1))
    kesi_list[0] = 0.5
    kesi_list[1 : p + 1] = 1.0
    kesi_list[int(2 * p)] = 1.0 / 2.0 ** p

    for i in np.arange(1, p, 1):
        kesi_list[2 * p - i] = kesi_list[2 * p - i + 1] + 2 ** (-p) * comb(p, i)

    for i in np.arange(0, 2 * p + 1, 1):
        eta_list[i] = (
            (-1) ** i * 2.0 * np.sqrt(2.0 * np.pi) * 10 ** (p / 3.0) * kesi_list[i]
        )

    return eta_list


def decompose_convergence_into_gaussians(
    func, radii_min, radii_max, func_terms=28, func_gaussians=20
):

    kesis = kesi(func_terms)  # kesi in Eq.(6) of 1906.08263
    etas = eta(func_terms)  # eta in Eqr.(6) of 1906.08263

    def f(sigma):
        """Eq.(5) of 1906.08263"""
        return np.sum(etas * np.real(target_function(sigma * kesis)))

    # sigma is sampled from logspace between these radii.

    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), func_gaussians)
    d_log_sigma = log_sigmas[1] - log_sigmas[0]
    sigmas = np.exp(log_sigmas)

    amps = np.zeros(func_gaussians)

    for i in range(func_gaussians):
        f_sigma = np.sum(etas * np.real(func(sigmas[i] * kesis)))
        if (i == -1) or (i == (func_gaussians - 1)):
            amps[i] = 0.5 * f_sigma * d_log_sigma / np.sqrt(2.0 * np.pi)
        else:
            amps[i] = f_sigma * d_log_sigma / np.sqrt(2.0 * np.pi)

    return amps, sigmas


def w_f_approx(z):
    """
    Compute the Faddeeva function :math:`w_{\mathrm F}(z)` using the
    approximation given in Zaghloul (2017).
    :param z: complex number
    :type z: ``complex`` or ``numpy.array(dtype=complex)``
    :return: :math:`w_\mathrm{F}(z)`
    :rtype: ``complex``

    # This function is copied from
    # "https://github.com/sibirrer/lenstronomy/tree/master/lenstronomy/LensModel/Profiles"
    # written by Anowar J. Shajib (see 1906.08263)
    """

    reg_minus_imag = z.imag < 0.0
    z[reg_minus_imag] = np.conj(z[reg_minus_imag])

    sqrt_pi = 1 / np.sqrt(np.pi)
    i_sqrt_pi = 1j * sqrt_pi

    wz = np.empty_like(z)

    z_imag2 = z.imag ** 2
    abs_z2 = z.real ** 2 + z_imag2

    reg1 = abs_z2 >= 38000.0
    if np.any(reg1):
        wz[reg1] = i_sqrt_pi / z[reg1]

    reg2 = (256.0 <= abs_z2) & (abs_z2 < 38000.0)
    if np.any(reg2):
        t = z[reg2]
        wz[reg2] = i_sqrt_pi * t / (t * t - 0.5)

    reg3 = (62.0 <= abs_z2) & (abs_z2 < 256.0)
    if np.any(reg3):
        t = z[reg3]
        wz[reg3] = (i_sqrt_pi / t) * (1 + 0.5 / (t * t - 1.5))

    reg4 = (30.0 <= abs_z2) & (abs_z2 < 62.0) & (z_imag2 >= 1e-13)
    if np.any(reg4):
        t = z[reg4]
        tt = t * t
        wz[reg4] = (i_sqrt_pi * t) * (tt - 2.5) / (tt * (tt - 3.0) + 0.75)

    reg5 = (62.0 > abs_z2) & np.logical_not(reg4) & (abs_z2 > 2.5) & (z_imag2 < 0.072)
    if np.any(reg5):
        t = z[reg5]
        u = -t * t
        f1 = sqrt_pi
        f2 = 1
        s1 = [1.320522, 35.7668, 219.031, 1540.787, 3321.99, 36183.31]
        s2 = [1.841439, 61.57037, 364.2191, 2186.181, 9022.228, 24322.84, 32066.6]

        for s in s1:
            f1 = s - f1 * u
        for s in s2:
            f2 = s - f2 * u

        wz[reg5] = np.exp(u) + 1j * t * f1 / f2

    reg6 = (30.0 > abs_z2) & np.logical_not(reg5)
    if np.any(reg6):
        t3 = -1j * z[reg6]

        f1 = sqrt_pi
        f2 = 1
        s1 = [5.9126262, 30.180142, 93.15558, 181.92853, 214.38239, 122.60793]
        s2 = [
            10.479857,
            53.992907,
            170.35400,
            348.70392,
            457.33448,
            352.73063,
            122.60793,
        ]

        for s in s1:
            f1 = f1 * t3 + s
        for s in s2:
            f2 = f2 * t3 + s

        wz[reg6] = f1 / f2

    # wz[reg_minus_imag] = np.conj(wz[reg_minus_imag])

    return wz


def zeta_from_grid(grid, amps, sigmas, axis_ratio):

    """
    The key part to compute the deflection angle of each Gaussian.
    Because of my optimization, there are blocks looking weird and indirect. What I'm doing here
    is trying to avoid big matrix operation to save time.
    I think there are still spaces we can optimize.

    It seems when using w_f_approx, it gives some errors if y < 0. So when computing for places
    where y < 0, we first compute the value at - y, and then change its sign.
    """

    output_grid_final = np.zeros(grid.shape[0], dtype="complex128")

    q2 = axis_ratio ** 2.0

    scale_factor = axis_ratio / (sigmas[0] * np.sqrt(2.0 * (1.0 - q2)))

    xs = (grid[:, 1] * scale_factor).copy()
    ys = (grid[:, 0] * scale_factor).copy()

    ys_minus = ys < 0.0
    ys[ys_minus] *= -1
    z = xs + 1j * ys
    zq = axis_ratio * xs + 1j * ys / axis_ratio

    expv = -(xs ** 2.0) * (1.0 - q2) - ys ** 2.0 * (1.0 / q2 - 1.0)

    for i in range(len(sigmas)):

        if i > 0:
            z /= sigmas[i] / sigmas[i - 1]
            zq /= sigmas[i] / sigmas[i - 1]
            expv /= (sigmas[i] / sigmas[i - 1]) ** 2.0

        output_grid = -1j * (w_f_approx(z) - np.exp(expv) * w_f_approx(zq))

        output_grid[ys_minus] = np.conj(output_grid[ys_minus])

        output_grid_final += (amps[i] * sigmas[i]) * output_grid

    return output_grid_final


def deflections_2d_from_grid_via_gaussians(
    func,
    axis_ratio,
    angle_profile,
    grid,
    radii_min,
    radii_max,
    func_terms=28,
    func_gaussians=20,
    sigmas_factor=1.0,
):

    axis_ratio = axis_ratio

    if axis_ratio > 0.9999:
        axis_ratio = 0.9999

    amps, sigmas = decompose_convergence_into_gaussians(
        func=func,
        radii_min=radii_min,
        radii_max=radii_max,
        func_terms=func_terms,
        func_gaussians=func_gaussians,
    )
    sigmas *= sigmas_factor

    angle = zeta_from_grid(grid=grid, amps=amps, sigmas=sigmas, axis_ratio=axis_ratio)

    angle *= np.sqrt((2.0 * np.pi) / (1.0 - axis_ratio ** 2.0))

    grid = np.vstack((-angle.imag, angle.real)).T

    return al.util.geometry.transform_grid_2d_from_reference_frame(
        grid_2d=grid, centre=(0.0, 0.0), angle=angle_profile
    )
