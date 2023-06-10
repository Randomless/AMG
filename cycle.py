import numpy as np
# https://www.wikiwand.com/en/Multigrid_method

def get_cycling(cycle_type="V"):
    print("Using cycle_type: ", cycle_type)
    if cycle_type == "V":
        return V_cycling
    elif cycle_type == "W":
        return W_cycling
    elif cycle_type == "F":
        return F_cycling
    else:
        raise TypeError(f"Unrecognized cycle type ({cycle_type})")


def V_cycling(
    level_idx,
    A_list,
    P_list,
    b,
    x0,
    n_max_coarse,
    PR_coef,
    smoother,
    pre_steps,
    pos_steps,
):
    A = A_list[level_idx]

    n = b.shape[0]
    if n <= n_max_coarse:
        x = np.linalg.solve(A.toarray(), b)
        return x

    x = smoother(A, b, 1e-14, pre_steps, x0)

    # Load restriction operator and construct interpolation operator
    P = P_list[level_idx]
    R = P.T * PR_coef
    coarse_n = R.shape[0]

    # Compute residual and transfer to coarse grid
    r = b - (A @ x).reshape(-1, 1)
    r_C = (R @ r).reshape(-1, 1)

    x0 = np.zeros(coarse_n)
    e_C = V_cycling(
        level_idx + 1,
        A_list,
        P_list,
        r_C,
        x0,
        n_max_coarse,
        PR_coef,
        smoother,
        pre_steps,
        pos_steps,
    )

    # Transfer error to fine grid and correct
    x = x + (P @ e_C).squeeze()

    # Post-smoothing
    x = smoother(A, b, 1e-14, pos_steps, x)
    return x


def W_cycling(
    level_idx,
    A_list,
    P_list,
    b,
    x0,
    n_max_coarse,
    PR_coef,
    smoother,
    pre_steps,
    pos_steps,
):
    """Multigrid cycling.

    Parameters
    ----------
    level_idx : int
        Solve problem on level `level_idx`
    x : numpy array
        Initial guess `x` and return correction
    b : numpy array
        Right-hand side for Ax=b
    cycle : {'V','W','F','AMLI'}
        Recursively called cycling function.  The
        Defines the cycling used:
        cycle = 'V',    V-cycle
        cycle = 'W',    W-cycle
    cycles_per_level : int, default 1
        Number of V-cycles on each level of an F-cycle
    """
    A = A_list[level_idx]

    n = b.shape[0]
    if n <= n_max_coarse:
        x = np.linalg.solve(A.toarray(), b)
        return x

    x = smoother(A, b, 1e-14, pre_steps, x0)

    # Load restriction operator and construct interpolation operator
    P = P_list[level_idx]
    R = P.T * PR_coef
    coarse_n = R.shape[0]

    # Compute residual and transfer to coarse grid
    r = b - (A @ x).reshape(-1, 1)
    r_C = (R @ r).reshape(-1, 1)

    x0 = np.zeros(coarse_n)
    e_C = W_cycling(
        level_idx + 1,
        A_list,
        P_list,
        r_C,
        x0,
        n_max_coarse,
        PR_coef,
        smoother,
        pre_steps,
        pos_steps,
    )

    # Transfer error to fine grid and correct
    x = x + (P @ e_C).squeeze()

    # Post-smoothing
    x = smoother(A, b, 1e-14, pos_steps, x)

    # Second W-cycle
    # Compute residual and transfer to coarse grid
    r = b - (A @ x).reshape(-1, 1)
    r_C = (R @ r).reshape(-1, 1)

    x0 = np.zeros(coarse_n)
    e_C = W_cycling(
        level_idx + 1,
        A_list,
        P_list,
        r_C,
        x0,
        n_max_coarse,
        PR_coef,
        smoother,
        pre_steps,
        pos_steps,
    )

    # Transfer error to fine grid and correct
    x = x + (P @ e_C).squeeze()

    # Post-smoothing
    x = smoother(A, b, 1e-14, pos_steps, x)

    return x


def F_cycling(
    level_idx,
    A_list,
    P_list,
    b,
    x0,
    n_max_coarse,
    PR_coef,
    smoother,
    pre_steps,
    pos_steps,
):
    A = A_list[level_idx]

    n = b.shape[0]
    if n <= n_max_coarse:
        x = np.linalg.solve(A.toarray(), b)
        return x

    x = smoother(A, b, 1e-14, pre_steps, x0)

    # Load restriction operator and construct interpolation operator
    P = P_list[level_idx]
    R = P.T * PR_coef
    coarse_n = R.shape[0]

    # Compute residual and transfer to coarse grid
    r = b - (A @ x).reshape(-1, 1)
    r_C = (R @ r).reshape(-1, 1)

    x0 = np.zeros(coarse_n)
    e_C = F_cycling(
        level_idx + 1,
        A_list,
        P_list,
        r_C,
        x0,
        n_max_coarse,
        PR_coef,
        smoother,
        pre_steps,
        pos_steps,
    )

    # Transfer error to fine grid and correct
    x = x + (P @ e_C).squeeze()

    # Post-smoothing
    x = smoother(A, b, 1e-14, pos_steps, x)

    # Compute residual and transfer to coarse grid
    r = b - (A @ x).reshape(-1, 1)
    r_C = (R @ r).reshape(-1, 1)

    x0 = np.zeros(coarse_n)
    e_C = V_cycling(
        level_idx + 1,
        A_list,
        P_list,
        r_C,
        x0,
        n_max_coarse,
        PR_coef,
        smoother,
        pre_steps,
        pos_steps,
    )

    # Transfer error to fine grid and correct
    x = x + (P @ e_C).squeeze()

    # Post-smoothing
    x = smoother(A, b, 1e-14, pos_steps, x)

    return x
