
import numpy as np
 
 
def jacobi(A, b, x0=None, tol=1e-10, max_iter=1000):
    """Método de Jacobi:  x_i = ( b_i - sum_{j!=i} a_ij x_j ) / a_ii.
 
    Retorna (x, iteraciones).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.size
 
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    x_new = np.empty(n)
 
    for k in range(1, max_iter + 1):
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i, j] * x[j]
            x_new[i] = (b[i] - s) / A[i, i]
 
        err = np.linalg.norm(x_new - x, np.inf)
        x[:] = x_new
        if err < tol:
            return x, k
 
    return x, max_iter
 
 
if __name__ == "__main__":
    A = np.array([[4.0, -1.0,  0.0],
                  [-1.0, 4.0, -1.0],
                  [0.0, -1.0,  4.0]])
    b = np.array([3.0, 2.0, 3.0])
 
    x, k = jacobi(A, b)
    print("x =", x)
    print("iteraciones =", k)