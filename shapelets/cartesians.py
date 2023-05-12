import autolens as al
import autolens.plot as aplt

import numpy as np
from scipy.special import hermite
from scipy.special import factorial


beta = 2.0
n_x = 2
n_y = 2

grid = al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.1)

hermite_y = hermite(n=n_y)
hermite_x = hermite(n=n_x)

y = grid.slim[:, 0]
x = grid.slim[:, 1]

shapelet_y = hermite_y(y / beta)
shapelet_x = hermite_x(x / beta)

basis = (
    shapelet_y
    * shapelet_x
    * np.exp(-0.5 * (y**2 + x**2) / (beta**2))
    / beta
    / (np.sqrt(2 ** (n_x + n_y) * (np.pi**0.5) * factorial(n_y) * factorial(n_x)))
)

print(basis)

xss

# BasisF = shapelets_hermite(n[0] ,(x 1 /float(beta))) * shapelets_hermite(n[1] ,(x 2 /float(beta))) * $
# exp(-. 5 *(x 1 ^ 2 +x 2 ^2 ) /float(bet a ^2))       / $
# beta                                     / $
# sqrt( 2 . ^(n[0 ] +n[1]) * !pi * factorial(n[0]) * factorial(n[1]) )
