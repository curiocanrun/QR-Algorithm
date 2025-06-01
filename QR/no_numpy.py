def multiply(A, B):
  rows, cols, dim = len(A), len(B[0]), len(B)
  return [[sum(A[i][k] * B[k][j] for k in range(dim)) for j in range(cols)] for i in range(rows)]

def dot(u, v):
  return sum(u[i] * v[i] for i in range(len(u)))

def norm(v):
  return (sum(x*x for x in v))**0.5

def subtract(u, v):
  return [u[i]-v[i] for i in range(len(u))]

def scalar_multiply(v, c):
  return [c*v[i] for i in range(len(v))]

def make_similar(A):
  n = len(A)
  Q, R = qr_decomposition(A)
  return multiply(R, Q)

def qr_decomposition(A):
    n = len(A)
    m = len(A[0])
    Q = [[0.0]*n for _ in range(m)]
    R = [[0.0]*m for _ in range(m)]

    for i in range(m):
        vi = [A[j][i] for j in range(n)]
        for j in range(i):
            qj = [Q[j][k] for k in range(n)]
            R[j][i] = dot(qj, vi)
            vi = subtract(vi, scalar_multiply(qj, R[j][i]))
        R[i][i] = norm(vi)
        Q[i] = scalar_multiply(vi, 1 / R[i][i])

    Q_cols = list(map(list, zip(*Q)))
    return Q_cols, R

# Frobenius norm of off-diagonal elements
def off_diag_norm(A):
    total = 0.0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i != j:
                total += A[i][j] ** 2
    return total ** 0.5

def QR_algorithm(A, tol=1e-10, max_iter=1000):
    n = len(A)
    A_next = A.copy()
    diff = 1.0
    iters = 0

    while diff > tol and iters < max_iter:
        A_next = make_similar(A_next)
        diff = off_diag_norm(A_next)
        iters += 1

    eigen_values = [A_next[i][i] for i in range(n)]
    return eigen_values, iters

A_verify = [[1, 2], [3, 4]]

eigens, steps = QR_algorithm(A_verify)
print("Eigenvalues:", [round(eigen, 2) for eigen in eigens])
print("Iterations:", steps)

A_test = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]

eigens, steps = QR_algorithm(A_test)
print("Eigenvalues:", [round(eigen, 2) for eigen in eigens])
print("Iterations:", steps)

