import numpy as np
from numba import njit

@njit
def fast_l2_norm(v):
    s = 0.0
    for x in v:
        s += x * x
    return s ** 0.5

# Gram-Schmidt QR decomposition
def qr_decomposition(A):
    n = A.shape[0]
    q_stack = []
    a_per = []

    q0 = A[:, 0].astype(np.float64)
    norm = fast_l2_norm(q0)
    q0 = q0 / norm
    q_stack.append(q0)
    a_per.append(A[:, 0])

    for i in range(1, n):
        proj = A[:, i].copy().astype(np.float64)
        for j in range(i):
            residue = np.dot(proj, q_stack[j])
            proj = proj - residue * q_stack[j]
        a_per.append(proj)
        q_stack.append(proj / fast_l2_norm(proj))

    r_stack = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i > j:
                r_stack[i][j] = 0.0
            elif i == j:
                r_stack[i][j] = fast_l2_norm(a_per[i])
            else:
                r_stack[i][j] = float(np.dot(A[:, j], q_stack[i]))

    return q_stack, r_stack

def make_similar(A):
    q_stack, r_stack = qr_decomposition(A)
    Q = np.column_stack(q_stack)
    R = np.array(r_stack)
    return R @ Q

# Off-diagonal Frobenius norm
def off_diag_norm(A):
    n = A.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                total += A[i, j] * A[i, j]
    return total ** 0.5

def QR_algorithm(A, tol=1e-10, max_iter=1000):
    A_next = A.copy().astype(np.float64)
    iters = 0
    diff = 1.0

    while diff > tol and iters < max_iter:
        A_next = make_similar(A_next)
        diff = off_diag_norm(A_next)
        iters += 1

    eigen_values = [float(A_next[i, i]) for i in range(A_next.shape[0])]
    return eigen_values, iters

A_verify = np.array([[1, 2], [3, 4]], dtype=np.float64)
A_test = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=np.float64)

eigens, steps = QR_algorithm(A_verify)
print("A_verify Eigenvalues: ", [round(eigen, 2) for eigen in eigens])
print("Iterations: ", steps)

eigens, steps = QR_algorithm(A_test)
print("A_test Eigenvalues: ", [round(eigen, 2) for eigen in eigens])
print("Iterations: ", steps)

