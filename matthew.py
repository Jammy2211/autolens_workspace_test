import numpy as np
import scipy

def fnnls_modified(
    ZTZ,
    ZTx,
    P_initial=np.zeros(0, dtype=int),
    lstsq=lambda A, x: scipy.linalg.solve(A, x, assume_a="pos"),
):
    """
    Implementation of the Fast Non-megative Least Squares Algorithm described
    in the paper "A fast non-negativity-constrained least squares algorithm"
    by Rasmus Bro and Sijmen De Jong.

    This algorithm seeks to find min_d ||x - Zd|| subject to d >= 0

    Some of the comments, such as "B2", refer directly to the steps of
    the fnnls algorithm as presented in the paper by Bro et al.

    Parameters
    ----------
    Z: NumPy array
        Z is an m x n matrix.

    x: Numpy array
        x is a m x 1 vector.

    P_initial: Numpy array, dtype=int
        By default, an empty array. An estimate for
        the indices of the support of the solution.

        lstsq: function
        By default, numpy.linalg.lstsq with rcond=None.
        Least squares function to use when calculating the
        least squares solution min_x ||Ax - b||.
        Must be of the form x = f(A,b).

    Returns
    -------
    d: Numpy array
        d is a nx1 vector
    """

    # Z, x, P_initial = map(np.asarray_chkfinite, (Z, x, P_initial))

    n = np.shape(ZTZ)[0]

    # Calculating ZTZ and ZTx in advance to improve the efficiency of calculations
    # ZTZ = Z.T.dot(Z)
    # ZTx = Z.T.dot(x)

    # Declaring constants for tolerance and max repetitions
    epsilon = 2.2204e-16
    tolerance = epsilon * n

    # number of contsecutive times the set P can remain unchanged loop until we terminate
    max_repetitions = 2

    # A1 + A2
    P = np.zeros(n, dtype=np.bool)
    P[P_initial] = True

    # A3
    d = np.zeros(n)

    # A4
    w = ZTx - (ZTZ) @ d

    # Initialize s
    s = np.zeros(n)

    # Count of amount of consecutive times set P has remained unchanged
    no_update = 0

    count = 0
    sub_count = 0

    # Extra loop in case a support is set to update s and d
    if P_initial.shape[0] != 0:

        s[P] = lstsq((ZTZ)[P][:, P], (ZTx)[P])
        d = s.clip(min=0)

    # B1
    while (not np.all(P)) and np.max(w[~P]) > tolerance:

        count += 1

        current_P = (
            P.copy()
        )  # make copy of passive set to check for change at end of loop

        # B2 + B3
        P[np.argmax(w * ~P)] = True

        # B4
        s[P] = lstsq((ZTZ)[P][:, P], (ZTx)[P])

        # C1
        while np.any(P) and np.min(s[P]) <= tolerance:

            sub_count += 1

            s, d, P = fix_constraint(ZTZ, ZTx, s, d, P, tolerance, lstsq)

        # B5
        d = s.copy()
        # B6
        w = ZTx - (ZTZ) @ d

        # check if there has been a change to the passive set
        if np.all(current_P == P):
            no_update += 1
        else:
            no_update = 0

        if no_update >= max_repetitions:
            break

    # res = np.linalg.norm(x - Z@d) #Calculate residual loss ||x - Zd||

    print(f"Total Iterations = {count} / {sub_count}")

    return d

def fix_constraint(
    ZTZ,
    ZTx,
    s,
    d,
    P,
    tolerance,
    lstsq=lambda A, x: scipy.linalg.solve(A, x, assume_a="pos"),
):
    """
    The inner loop of the Fast Non-megative Least Squares Algorithm described
    in the paper "A fast non-negativity-constrained least squares algorithm"
    by Rasmus Bro and Sijmen De Jong.

    One iteration of the loop to adjust the new estimate s to satisfy the
    nonnegativity contraint of the solution.

    Some of the comments, such as "B2", refer directly to the steps of
    the fnnls algorithm as presented in the paper by Bro et al.

    Parameters
    ----------
    ZTZ: NumPy array
        ZTZ is an n x n matrix equal to Z.T * Z

    ZTx: Numpy array
        ZTx is an n x 1 vector equal to Z.T * x

    s: Numpy array
        The new estimate of the solution with possible
        negative values that do not meet the constraint

    d: Numpy array
        The previous estimate of the solution that satisfies
        the nonnegativity contraint

    P: Numpy array, dtype=np.bool
        The current passive set, which comtains the indices
        that are not fixed at the value zero.

    tolerance: float
        A tolerance, below which values are considered to be
        0, allowing for more reasonable convergence.

    lstsq: function
        By default, numpy.linalg.lstsq with rcond=None.
        Least squares function to use when calculating the
        least squares solution min_x ||Ax - b||.
        Must be of the form x = f(A,b).

    Returns
    -------
    s: Numpy array
        The updated new estimate of the solution.
    d: Numpy array
        The updated previous estimate, now as close
        as possible to s while maintaining nonnegativity.
    P: Numpy array, dtype=np.bool
        The updated passive set
    """
    # C2
    q = P * (s <= tolerance)
    alpha = np.min(d[q] / (d[q] - s[q]))

    # C3
    d = d + alpha * (
        s - d
    )  # set d as close to s as possible while maintaining non-negativity

    # C4
    P[d <= tolerance] = False

    # C5
    s[P] = lstsq((ZTZ)[P][:, P], (ZTx)[P])

    # C6
    s[~P] = 0.0

    return s, d, P


from scipy import linalg
import time

ZTZ = np.load("ZTZ.npy")
ZTx = np.load("ZTx.npy")

start = time.time()

reconstruction = fnnls_modified(
    ZTZ,
    ZTx,
    lstsq=lambda A, x: linalg.solve(A, x, assume_a="pos", overwrite_a=True, overwrite_b=True, check_finite=False),
)

print(time.time() - start)