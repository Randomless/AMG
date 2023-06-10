import numpy as np
import time
from scipy import sparse

from cycle import get_cycling


def generate_strong_dependency_matrix(A, theta):
    """
    Generate a strong dependency matrix based on the matrix A and threshold theta.

    Args:
        A (array): Input matrix
        theta (float): Threshold value for strong dependency

    Returns:
        sd_mat (sparse matrix): Strong dependency matrix
    """
    n = A.shape[0]
    sd_mat = sparse.lil_matrix((n, n), dtype=int)

    for row in range(n):
        threshold = theta * np.max(-A[row, :])
        if threshold == 0:
            continue
        nnz_in_row = np.any(A[row, :] != 0).nonzero()[1]
        nnz_in_row = np.setdiff1d(nnz_in_row, [row])  # Ignore diagonal element
        sd_in_row = np.any(-A[row, :] >= threshold).nonzero()[1]
        wd_in_row = np.setdiff1d(nnz_in_row, sd_in_row)
        sd_mat[row, sd_in_row] = 1
        sd_mat[row, wd_in_row] = -1

    return sd_mat


def generate_coarse_grid(A):
    """
    Generate a coarse grid identify list from A.

    Args:
        A (array): Fine grid problem matrix

    Returns:
        is_coarse_grid_point (array): Array indicating the coarse grid points, 1 if grid point i is a coarse grid point.
    """

    n = A.shape[0]
    edge_list = (A != 0).astype(int)  # The edge list of the graph
    # np.fill_diagonal(edge_list, 0)  # Remove the diagonal to exclude self-connections.
    edge_list = edge_list - sparse.diags(
        edge_list.diagonal(), 0
    )  # Remove the diagonal to exclude self-connections.

    x_degree = np.sum(
        edge_list, axis=1
    )  # the degree of each grid point, i.e., the number of neighboring points.
    cg_point = -np.ones(n)  # Initialize all grid points as unvisited (marked with -1)

    unvisited_points = np.sum(cg_point == -1)

    while unvisited_points > 0:
        # Find next coarse grid point
        # selects the next coarse grid point with the highest degree (x_degree) and marks it as a coarse grid point
        new_cg_point = np.argmax(x_degree)
        cg_point[new_cg_point] = 1
        x_degree[new_cg_point] = 0

        # Find new fine grid point neighbors
        # and update their degrees.
        neighbors = np.any(edge_list[new_cg_point, :] == 1).nonzero()[1]
        new_neighbors = []

        for neighbor in neighbors:
            if cg_point[neighbor] == -1:
                new_neighbors.append(neighbor)
                # Mark new neighbors
                cg_point[neighbor] = 0
                x_degree[neighbor] = 0

        # Mark new neighbors and add their unvisited neighbors' weight
        for neighbor in new_neighbors:
            nn = np.any(edge_list[neighbor, :] == 1).nonzero()[1]
            nn2 = []

            for i in nn:
                if cg_point[i] == -1:  # Only update those unvisited neighbors
                    nn2.append(i)

            for i in nn2:
                x_degree[i] += 1

        unvisited_points = np.sum(cg_point == -1)

    is_coarse_grid_point = cg_point

    return is_coarse_grid_point


def interpolate_point(curr_pid, A, sd_A, is_cg):
    """
    Perform interpolation for the given point.

    Args:
        curr_pid (int): Current point ID
        A (array): Coefficient matrix
        sd_A (array): Strong dependency matrix
        is_cg (array): Array indicating whether a point is a coarse grid point (1) or not (0)

    Returns:
        ip_id (array): Interpolation point IDs
        ip_coef (array): Interpolation coefficients
    """
    if is_cg[curr_pid] == 1:
        ip_id = np.array([curr_pid])
        ip_coef = np.array([1])
        return ip_id, ip_coef

    neighbors = np.any(A[curr_pid, :] != 0).nonzero()[1]
    neighbors = np.setdiff1d(neighbors, [curr_pid])
    neighbors_cnt = neighbors.size
    ip_cnt = 0
    ip_id = np.zeros(neighbors_cnt, dtype=int)
    ip_coef = np.zeros(neighbors_cnt)

    for i in range(neighbors_cnt):
        curr_neighbor = neighbors[i]
        if is_cg[curr_neighbor] == 1:
            ip_cnt += 1
            ip_id[ip_cnt - 1] = curr_neighbor
            ip_coef[ip_cnt - 1] = A[curr_pid, curr_neighbor]

    ip_id = ip_id[:ip_cnt]
    ip_coef = ip_coef[:ip_cnt]

    a_ii_new = A[curr_pid, curr_pid]

    for i in range(neighbors_cnt):
        curr_neighbor = neighbors[i]
        if is_cg[curr_neighbor] == 0 and sd_A[curr_pid, curr_neighbor] == -1:
            a_ii_new += A[curr_pid, curr_neighbor]

    for i in range(neighbors_cnt):
        curr_neighbor = neighbors[i]
        if is_cg[curr_neighbor] == 0 and sd_A[curr_pid, curr_neighbor] == 1:
            cg_neighbor_list = []
            cg_neighbor_weight_sum = 0

            for j in range(neighbors_cnt):
                curr_check_neighbor = neighbors[j]
                if (
                    is_cg[curr_check_neighbor] == 1
                    and A[curr_neighbor, curr_check_neighbor] != 0
                ):
                    cg_neighbor_weight_sum += A[curr_neighbor, curr_check_neighbor]
                    cg_neighbor_list.append(j)

            for k in range(len(cg_neighbor_list)):
                j = cg_neighbor_list[k]
                curr_check_neighbor = neighbors[j]
                contrib = (
                    A[curr_neighbor, curr_check_neighbor]
                    / cg_neighbor_weight_sum
                    * A[curr_pid, curr_neighbor]
                )

                ip_id_pos = np.where(ip_id == curr_check_neighbor)[0]
                ip_coef[ip_id_pos] += contrib

    ip_coef = ip_coef / -a_ii_new

    return ip_id, ip_coef


def generate_coarse_grid_and_interpolation(A):
    """
    Using A matrix to generate the coarse grid matrix CA and interpolation operator P.

    Args:
        A (array): Fine grid problem matrix

    Returns:
        CA (array): Coarse grid problem coefficient matrix
        P (array): Interpolation operator, CA = P^T * A * P
    """

    is_cg_point = generate_coarse_grid(A)
    cg_point_id = np.cumsum(is_cg_point)
    CA_n = int(np.sum(is_cg_point))
    A_n = A.shape[0]
    sd_A = generate_strong_dependency_matrix(A, 0.5)

    # Construct interpolation operator P
    P_rows = []
    P_cols = []
    P_vals = []
    P_nnz = 0
    n = A.shape[0]

    for curr_pid in range(0, n):
        # P is CA_n cols, A_n rows, each row is an interpolation relationship
        # from coarse grid to fine grid
        ip_id, ip_coef = interpolate_point(curr_pid, A, sd_A, is_cg_point)

        for i in range(ip_id.shape[0]):
            P_nnz += 1
            P_rows.append(curr_pid)
            P_cols.append(int(cg_point_id[ip_id[i]] - 1))
            P_vals.append(ip_coef[i])

    # convert P_cols to int
    # P_cols = map(int, P_cols)
    P = sparse.coo_matrix((P_vals, (P_rows, P_cols)), shape=(A_n, CA_n)).tocsc()

    # Construct coarse grid matrix
    CA = P.transpose() * A * P

    return CA, P


def generate_level_structure(A, n_max_coarse):
    """
    Generate the level info(coefficient matrices) for AMG.

    Args:
        A (array): Original coefficient matrix
        n_max_coarse (int): Maximum number of unknowns could be solved at the coarsest level

    Returns:
        A_list (list): The array of coefficient matrices on each level
        P_list (list): The array of interpolation (prolongation) operators on each level
        max_level (int): Maximum level for V-cycle, initially set to 1
    """
    n = A.shape[0]
    A_list = []
    P_list = []
    level = 1
    A_list.append(A)

    while n > n_max_coarse:
        CA, P = generate_coarse_grid_and_interpolation(A)
        P_list.append(P)
        A_list.append(CA)
        level += 1
        A = CA
        n = A.shape[0]

    max_level = level

    # print A_list and P_list shape with table format
    print("Level structure generated.")
    print("Level\t A shape\t P shape \t A Non-zero percentage")
    for i in range(max_level):
        nnz_ratio = np.count_nonzero(A_list[i].toarray()) / (
            A_list[i].shape[0] * A_list[i].shape[1]
        )
        # last i only print A_list[i] shape , P_list[i] is empty
        if i == max_level - 1:
            print(f"{i}\t {A_list[i].shape}\t \t\t {nnz_ratio * 100.0:.4f}")
        else:
            print(
                f"{i}\t {A_list[i].shape}\t {P_list[i].shape}\t {nnz_ratio * 100.0:.4f}"
            )

    return A_list, P_list, max_level


def vcycle_recursive(
    level, A_list, P_list, b, x0, n_max_coarse, PR_coef, smoother, pre_steps, pos_steps
):
    """
    Perform a recursive V-cycle in one iteration in multigrid_solver.

    Args:
        level (int): Current level
        A_list (list): List of coefficient matrices on each level
        P_list (list): List of interpolation operators on each level
        b (array): Right-hand side vector
        x0 (array): Initial solution vector
        n_max_coarse (int): Threshold for solving the problem directly
        PR_coef (array): Coefficient matrix for restriction operator
        smoother (function): Function handle for an iterative method as a smoother
        pre_steps (int): Number of iterations in the pre-smoothing
        pos_steps (int): Number of iterations in the post-smoothing

    Returns:
        x (array): Solution vector
    """
    # Load coefficient matrix
    A = A_list[level]

    # If the problem is small enough, solve it directly
    n = b.shape[0]
    if n <= n_max_coarse:
        x = np.linalg.solve(A.toarray(), b)
        return x

    # Pre-smoothing
    x = smoother(A, b, 1e-14, pre_steps, x0)

    # Load restriction operator and construct interpolation operator
    P = P_list[level]
    R = P.T * PR_coef
    coarse_n = R.shape[0]

    # Compute residual and transfer to coarse grid
    r = b - (A @ x).reshape(-1, 1)
    r_C = (R @ r).reshape(-1, 1)

    # Solve coarse grid problem recursively
    x0 = np.zeros(coarse_n)
    e_C = vcycle_recursive(
        level + 1,
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


def multigrid_solver(
    A,
    b,
    smoother=None,
    pre_steps=1,
    pos_steps=1,
    rn_tol=1e-10,
    max_cycle=100,
    n_max_coarse=16,
    cycle_type="V",
):
    """
    Multigrid solver for A * x = b using AMG.

    Args:
        A (array): The initial coefficient matrix
        b (array): The right hand side
        smoother (function): Function handle of smoother function
        pre_steps (int): Number of iterations of pre-smoothing
        pos_steps (int): Number of iterations of post-smoothing
        rn_tol (float): The tolerance of the relative residual norm
        max_cycle (int): Maximum number of cycles to perform
        n_max_coarse (int): Maximum number of unknowns could be solved at the coarsest level

    Returns:
        x (array): The solution array
        cycle_cnt (int): Number of cycles performed
        rel_res (array): Relative residual norm after each iteration
    """
    print("-"*10)
    n = A.shape[0]
    x = np.zeros(n)
    rn = np.linalg.norm(b)
    cycle_cnt = 0
    rel_res = [rn]
    rn_stop = rn * rn_tol
    PR_coef = 1

    # Generate coefficient matrices and interpolation operators of each level at once
    tic = time.time()
    A_list, P_list, max_level = generate_level_structure(A, n_max_coarse)
    setup_time = time.time() - tic

    # Repeat V-cycle until convergence
    tic = time.time()
    cycling_fn = get_cycling(cycle_type=cycle_type)
    while rn > rn_stop and cycle_cnt <= max_cycle:
        x = cycling_fn(
            0,
            A_list,
            P_list,
            b,
            x,
            n_max_coarse,
            PR_coef,
            smoother,
            pre_steps,
            pos_steps,
        )
        r = b - (A @ x).reshape(-1, 1)
        rn = np.linalg.norm(r, 2)
        cycle_cnt += 1
        rel_res.append(rn / rel_res[0])

    rel_res = rel_res[1:]
    solve_time = time.time() - tic

    print("Setup time =", setup_time, "(s)")
    print("Solve  time =", solve_time, "(s)")
    print("Performed Cycle =", cycle_cnt)
    print("Residual Norm ||b-A*x||_2    =", rn)

    print("\n")
    return x, cycle_cnt, rel_res
