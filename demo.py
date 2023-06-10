import numpy as np
from scipy import sparse
import scipy.io
from multigrid import multigrid_solver
from smoother import gauss_seidel, jacobi
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # read mat  from local file
    mat = scipy.io.loadmat("./matrices.mat")
    A = mat["A"]
    b = mat["b"]

    # print shape and data type
    print("Load matrix from local file, data :")
    print("A.shape: ", A.shape)
    print("b.shape: ", b.shape)
    print("A.dtype: ", A.dtype)
    print("b.dtype: ", b.dtype)

    cycle_types = ["V", "W", "F"]
    rel_res_dict = {}
    for cycle_type in cycle_types:
        x, cycle_cnt, rel_res = multigrid_solver(
            A,
            b,
            smoother=gauss_seidel,
            pre_steps=1,
            pos_steps=1,
            rn_tol=1e-10,
            max_cycle=100,
            cycle_type=cycle_type,
        )
        rel_res_dict[cycle_type] = rel_res
        print("len(rel_res): ", len(rel_res))

    # plot relative residual dict of different cycle types for comparison
    plt.figure(figsize=(8, 6))
    for cycle_type in cycle_types:
        plt.semilogy(rel_res_dict[cycle_type], label=cycle_type)
    plt.xlabel("cycle")
    plt.ylabel("relative residual")
    plt.legend()
    plt.savefig("./relative_residual.png")
