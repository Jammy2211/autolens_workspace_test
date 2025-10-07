import numpy as np
import math


class MassProfile:
    """Parent class for mass profiles used in gravitational lensing"""

    def __init__(self, ellipticity=0, position_angle=0, cx=0, cy=0):
        """Initialize the base mass profile

        Parameters:
        -----------
        ellipticity : float
            Unified ellipticity parameter (0-1, where 0 is circular)
        position_angle : float
            Position angle in radians
        cx, cy : float
            Center coordinates of the profile
        """
        self.PI = math.pi
        self.cx = cx
        self.cy = cy
        self.ellipticity = ellipticity
        self.position_angle = position_angle

    def get_nfw_eps(self):
        """Convert unified ellipticity to NFW eps parameter

        For NFW profile, we can use the ellipticity directly

        Returns:
        --------
        float : eps parameter for NFW
        """
        return self.ellipticity

    def get_piemd_epot(self):
        """Convert unified ellipticity to PIEMD epot parameter

        PIEMD internally uses epot=(1-sqrt(1-e²))/e

        Returns:
        --------
        float : epot parameter for PIEMD
        """
        e = self.ellipticity
        if abs(e) < 1e-6:  # Handle circular case
            return 0
        return (1 - np.sqrt(1 - e**2)) / e

    def kappa(self, x, y):
        """Calculate the convergence at the given coordinates

        Parameters:
        -----------
        x, y : float or array
            Coordinates where to evaluate convergence

        Returns:
        --------
        float or array : convergence kappa
        """
        raise NotImplementedError("Subclasses must implement kappa(x, y)")

    def dpl(self, x, y):
        """Calculate the deflection angle at the given coordinates

        Parameters:
        -----------
        x, y : float or array
            Coordinates where to evaluate deflection

        Returns:
        --------
        tuple : (alpha_x, alpha_y) deflection angles
        """
        raise NotImplementedError("Subclasses must implement dpl(x, y)")

    def gamma(self, x, y):
        """Calculate the shear at the given coordinates

        Parameters:
        -----------
        x, y : float or array
            Coordinates where to evaluate shear

        Returns:
        --------
        dict : {'gamma1': γ₁, 'gamma2': γ₂} shear components
        """
        raise NotImplementedError("Subclasses must implement gamma(x, y)")


# 定义矩阵类
class Matrix:
    def __init__(self):
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0

    def set_from_array(self, a, b, c, d):
        """Set matrix values from arrays"""
        self.a = a
        self.b = b
        self.c = c
        self.d = d


class PIEMD(MassProfile):

    def __init__(
        self,
        ellipticity: float = 0,
        rcut: float = 1.0,
        rc: float = 0.1,
        sigma: float = 200.0,
        position_angle: float = 0,
        cx: float = 0,
        cy: float = 0,
    ):
        """Initialize PIEMD profile with either new or old parameter format

        Parameters:
        -----------
        ellipticity : float, optional
            Unified ellipticity parameter (0-1)
        rcut : float
            Cutoff radius
        rc : float
            Core radius
        sigma : float
            Velocity dispersion
        position_angle : float, optional
            Position angle in radians
        cx, cy : float, optional
            Center coordinates
        emass : float, optional
            Legacy ellipticity parameter (if provided, overrides ellipticity)
        theta : float, optional
            Legacy position angle (if provided, overrides position_angle)
        """

        super().__init__(ellipticity, position_angle, cx, cy)

        self.epot = self.get_piemd_epot()
        self.rcut = rcut
        self.rc = rc
        self.pia_c2 = 7.209970e-06
        self.sigma = sigma
        self.b0 = 6.0 * self.pia_c2 * sigma**2

        # Maintain theta for backward compatibility
        self.theta = position_angle

    def update(self):
        self.epot = self.get_piemd_epot()
        self.b0 = 6.0 * self.pia_c2 * self.sigma**2

    def mdci05(self, x, y, eps, rc, b0, res):
        # Convert inputs to numpy arrays if they aren't already
        x = np.asarray(x)
        y = np.asarray(y)

        sqe = np.sqrt(eps)
        cx1 = (1.0 - eps) / (1.0 + eps)
        cx1inv = 1.0 / cx1
        cxro = (1.0 + eps) * (
            1.0 + eps
        )  # rem^2 = x^2 / (1+e^2) + y^2 / (1-e^2) Eq 2.3.6
        cyro = (1.0 - eps) * (1.0 - eps)
        ci = 0.5 * (1.0 - eps * eps) / sqe
        wrem = np.sqrt(
            rc * rc + x * x / cxro + y * y / cyro
        )  # wrem^2 = w^2 + rem^2 with w core radius

        # Calculate denominators and numerators
        den1 = 2.0 * sqe * wrem - y * cx1inv
        den1 = cx1 * cx1 * x * x + den1 * den1
        num2 = 2.0 * rc * sqe - y
        den2 = x * x + num2 * num2

        # Calculate derivatives
        didxre = ci * (
            cx1
            * (2.0 * sqe * x * x / cxro / wrem - 2.0 * sqe * wrem + y * cx1inv)
            / den1
            + num2 / den2
        )
        didyre = ci * ((2.0 * sqe * x * y * cx1 / cyro / wrem - x) / den1 + x / den2)
        didyim = ci * (
            (
                2.0 * sqe * wrem * cx1inv
                - y * cx1inv * cx1inv
                - 4 * eps * y / cyro
                + 2.0 * sqe * y * y / cyro / wrem * cx1inv
            )
            / den1
            - num2 / den2
        )

        # Update res matrix
        res.a = b0 * didxre
        res.b = b0 * didyre
        res.d = res.b  # They're equal
        res.c = b0 * didyim

    def main(self, x, y):
        """Main calculation method that now takes x, y as parameters"""
        g05c = Matrix()
        g05cut = Matrix()
        g2 = Matrix()

        # Get rotated coordinates
        x_rot, y_rot = self._get_rotated_coords(x, y)

        # Handle very small ellipticity
        epot = self.get_piemd_epot()
        if epot < 2e-4:
            epot = 2e-4

        if epot > 0:
            t05 = self.rcut / (self.rcut - self.rc)
            self.mdci05(x_rot, y_rot, epot, self.rc, self.b0, g05c)
            self.mdci05(x_rot, y_rot, epot, self.rcut, self.b0, g05cut)

            g2.a = t05 * (g05c.a - g05cut.a)
            g2.b = t05 * (g05c.b - g05cut.b)
            g2.c = t05 * (g05c.c - g05cut.c)
            g2.d = t05 * (g05c.d - g05cut.d)
        else:
            # Using vectorized operations for array support
            RR = x_rot**2 + y_rot**2

            # Create arrays to store results
            g2.a = np.zeros_like(RR)
            g2.c = np.zeros_like(RR)
            g2.b = np.zeros_like(RR)
            g2.d = np.zeros_like(RR)

            # Handle the case where RR > 0
            mask = RR > 0.0
            if np.any(mask):
                X = self.rc
                Y = self.rcut
                t05 = self.b0 * Y / (Y - X)

                # Apply calculations only where mask is True
                z = np.sqrt(RR[mask] + X * X) - X - np.sqrt(RR[mask] + Y * Y) + Y
                X_val = RR[mask] / X
                Y_val = RR[mask] / Y
                p = (1.0 - 1.0 / np.sqrt(1.0 + X_val / self.rc)) / X_val - (
                    1.0 - 1.0 / np.sqrt(1.0 + Y_val / self.rcut)
                ) / Y_val
                X_val = x_rot[mask] ** 2 / RR[mask]
                Y_val = y_rot[mask] ** 2 / RR[mask]

                g2.a[mask] = t05 * (p * X_val + z * Y_val / RR[mask])
                g2.c[mask] = t05 * (p * Y_val + z * X_val / RR[mask])

                X_val = x_rot[mask] * y_rot[mask] / RR[mask]
                g2.b[mask] = t05 * (p * X_val - z * X_val / RR[mask])
                g2.d[mask] = g2.b[mask]  # They're equal

            # Handle the case where RR == 0
            mask_zero = ~mask
            if np.any(mask_zero):
                g2.a[mask_zero] = self.b0 / self.rc / 2.0
                g2.c[mask_zero] = self.b0 / self.rc / 2.0
                # g2.b and g2.d are already zeros

        return g2

    def kappa(self, x, y):
        grad2 = self.main(x, y)
        kappa = 0.5 * (grad2.a + grad2.c)
        return kappa

    def gamma(self, x, y):
        grad2 = self.main(x, y)
        gamma1 = 0.5 * (grad2.a - grad2.c)
        gamma2 = grad2.b
        return {"gamma1": gamma1, "gamma2": gamma2}

    def ci05f(self, x, y):
        """Complex calculation method that now takes x, y as parameters"""
        # Get rotated coordinates
        x_rot, y_rot = self._get_rotated_coords(x, y)

        # Handle scalar inputs as arrays
        is_scalar = np.isscalar(x_rot) and np.isscalar(y_rot)

        if is_scalar:
            x_arr = np.array([x_rot])
            y_arr = np.array([y_rot])
        else:
            x_arr = x_rot.flatten()
            y_arr = y_rot.flatten()

        # Ensure epot is not too small
        epot = self.get_piemd_epot()
        if epot < 2e-4:
            epot = 2e-4

        sqe = np.sqrt(epot)
        cx1 = (1.0 - epot) / (1.0 + epot)
        cxro = (1.0 + epot) ** 2
        cyro = (1.0 - epot) ** 2
        rem2 = x_arr * x_arr / cxro + y_arr * y_arr / cyro

        zci = complex(0, -0.5 * (1.0 - epot**2) / sqe)

        # rc:
        znum_rc = cx1 * x_arr + 1j * (
            2.0 * sqe * np.sqrt(self.rc * self.rc + rem2) - y_arr / cx1
        )
        zden_rc = x_arr + 1j * (2.0 * self.rc * sqe - y_arr)

        # rcut:
        znum_rcut = znum_rc.real + 1j * (
            2.0 * sqe * np.sqrt(self.rcut * self.rcut + rem2) - y_arr / cx1
        )
        zden_rcut = zden_rc.real + 1j * (2.0 * self.rcut * sqe - y_arr)

        # Compute the ratio zis_rc / zis_rcut
        aa = znum_rc.real * zden_rc.real - znum_rc.imag * zden_rcut.imag
        bb = znum_rc.real * zden_rcut.imag + znum_rc.imag * zden_rc.real
        cc = znum_rc.real * zden_rc.real - zden_rc.imag * znum_rcut.imag
        dd = znum_rc.real * zden_rc.imag + zden_rc.real * znum_rcut.imag

        # Compute the norm
        norm = cc * cc + dd * dd
        aaa = (aa * cc + bb * dd) / norm
        bbb = (bb * cc - aa * dd) / norm

        # Compute the logarithm
        norm2 = aaa * aaa + bbb * bbb
        zr = np.log(np.sqrt(norm2)) + 1j * (np.arctan2(bbb, aaa))

        # Compute the final result
        zres = (zci.real * zr.real - zci.imag * zr.imag) + 1j * (
            zci.imag * zr.real + zci.real * zr.imag
        )

        # Return scalar if input was scalar
        if is_scalar:
            return zres[0]
        return zres

    def deflections_yx_2d_from(self, grid, **kwargs):
        """Calculate deflection angle for PIEMD profile

        For scalar inputs, we use the matrix approach
        For array inputs, we use the complex number approach
        """

        y = grid[:, 0]
        x = grid[:, 1]

        # Get rotated coordinates
        x_rel = x - self.cx
        y_rel = y - self.cy

        def rotation(x, y, theta):
            """
            Rotate coordinates by angle theta.

            Parameters
            ----------
            x, y : float or array_like
                Input coordinates
            theta : float
                Rotation angle in radians

            Returns
            -------
            Q_x, Q_y : float or array_like
                Rotated coordinates
            """
            Q_x = x * np.cos(theta) + y * np.sin(theta)
            Q_y = y * np.cos(theta) - x * np.sin(theta)
            return Q_x, Q_y

        # Apply rotation
        x_rot, y_rot = rotation(x_rel, y_rel, self.position_angle)

        t05 = self.b0 * self.rcut / (self.rcut - self.rc)
        zis = self.ci05f(x, y)

        g_x = (t05 * zis.real).reshape(x_rot.shape)
        g_y = (t05 * zis.imag).reshape(y_rot.shape)

        dpl_x, dpl_y = rotation(g_x, g_y, -self.position_angle)
        return dpl_x, dpl_y
