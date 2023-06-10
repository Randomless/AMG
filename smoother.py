import numpy as np
from numpy import array, zeros, diag, diagflat, dot


def gauss_seidel(A, b, iter_eps, max_iter, x0):
    """
    Perform Gauss-Seidel iteration to solve the linear system Ax = b.

    Args:
        A (array): Coefficient matrix
        b (array): Right-hand side vector
        iter_eps (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
        x0 (array): Initial solution vector

    Returns:
        x (array): Solution vector
    """
    n = A.shape[0]
    iter_cnt = 0
    err = 1
    x = x0.copy()

    while err >= iter_eps and iter_cnt < max_iter:
        x0 = x.copy()

        for i in range(n):
            A_ii = A[i, i]
            x[i] = (b[i] - np.dot(A[i, :].toarray(), x) + A_ii * x[i]) / A_ii

        iter_cnt += 1
        err = np.max(np.abs(x0 - x))

    return x


def jacobi(A, b, iter_eps, max_iter, x0):
    """
    Perform Jacobi iteration to do relaxation on the linear system Ax = b.

    Args:
        A (array): Coefficient matrix
        b (array): Right-hand side vector
        iter_eps (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
        x0 (array): Initial solution vector

    Returns:
        x (array): Solution vector
    """
    n = A.shape[0]
    iter_cnt = 0
    err = 1

    # Create an initial guess if needed
    x = zeros(x0.shape[0])

    A_array = A.toarray()
    b_squeeze = b.squeeze(-1)
    # Create a vector of the diagonal elements of A
    # and subtract them from A
    D = diag(A_array)
    R = A_array - diagflat(D)

    # Iterate for N times
    # for i in range(max_iter):
    while err >= iter_eps and iter_cnt < max_iter:
        x = (b_squeeze - dot(R, x)) / D
        iter_cnt += 1
        err = np.max(np.abs(x0 - x))
    return x
