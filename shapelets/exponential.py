import autolens as al
import autolens.plot as aplt

import numpy as np
from scipy.special import hermite
from scipy.special import factorial, genlaguerre

beta = 2.0
n = 2
m = 2

grid = al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.1)

radial = (grid[:, 0] ** 2 + grid[:, 1] ** 2) / beta
theta = np.arctan(grid[:, 1] / grid[:, 0])

# prefactor=   1. /  sqrt(2*!pi)/beta      * (n+0.5)^(-1-abs(m)) *        sqrt(factorial(n-abs(m))     /float(2*n+1)/factorial(n+abs(m))   )*(-1)^(n+m)
prefactor = (
    1.0
    / np.sqrt(2 * np.pi)
    / beta
    * (n + 0.5) ** (-1 - np.abs(m))
    * (-1) ** (n + m)
    * np.sqrt(factorial(n - np.abs(m)) / 2 * n + 1 / factorial(n + np.abs(m)))
)

# laguerre(r/(n[i]+0.5),n[i]-abs(m[i]),2*abs(m[i]),/DOUBLE)
laguerre = genlaguerre(n=n - np.abs(m), alpha=2 * np.abs(m))
shapelet = laguerre(radial / (n + 0.5))

#  BasisF[*,*,i]=prefactor[i] * exp(-r/(2*n[i]+1.)) * r^abs(m[i]) * shapelet * complex(cos(m[i]*theta),-sin(m[i]*theta))
basis = np.abs(
    prefactor
    * np.exp(-radial / (2 * n + 1))
    * radial ** (np.abs(m))
    * shapelet
    * np.cos(m * theta)
    + -1.0j * np.sin(m * theta)
)
