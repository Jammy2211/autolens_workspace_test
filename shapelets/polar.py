import autolens as al
import autolens.plot as aplt

import numpy as np
from scipy.special import hermite
from scipy.special import factorial, genlaguerre

beta = 2.0
n = 2
m = 2

grid = al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.1)

radial = (grid[:, 0] ** 2 + grid[:, 1] ** 2) / beta**2.0
theta = np.arctan(grid[:, 1] / grid[:, 0])

# const  = ((-1 ) ^(( n -abs(m) ) /2)) * sqrt(factorial(( n -abs(m) ) /2 ) /factorial(( n +abs(m) ) /2)) / beta / sqrt(!pi)
const = (
    ((-1) ** ((n - np.abs(m)) / 2))
    * np.sqrt(factorial((n - np.abs(m)) / 2) / factorial((n + np.abs(m)) / 2))
    / beta
    / np.sqrt(np.pi)
)

# gauss  = exp(-rsq /2.)
gauss = np.exp(-radial / 2.0)

laguerre = genlaguerre(n=(n - np.abs(m)) / 2.0, alpha=np.abs(m))

shapelet = laguerre(radial)
print(shapelet)

basis = np.abs(
    const
    * radial ** (np.abs(m / 2.0))
    * shapelet
    * gauss
    * np.exp(0.0 + 1j * -m * theta)
)

print(basis)


# for i=0 ,n_coeffs-1 do BasisF[* ,* ,i] =  $
# (const[i] * rs q ^(abs(m[i ] /2.)) )     * $
# laguerre(rsq ,(n[i ] -abs(m[i]) ) /2 ,abs(m[i])) * $
# gauss          * $
# exp(complex(0 ,-m[i ] *(theta)))
