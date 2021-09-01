import numpy as np

A = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 3.0]])

D = np.array([1.0, 2.0, 3.0])

S = np.linalg.solve(A, D)

print(S)

A_chol = np.linalg.cholesky(A)

print(A_chol)

print(A_chol)

from scipy import linalg

# A_chol = linalg.cho_factor(A)
# print(A_chol)
S_chol = linalg.cho_solve((A_chol, True), D)

print(S_chol)
