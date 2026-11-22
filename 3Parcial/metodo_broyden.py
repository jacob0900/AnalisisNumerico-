
import math
import numpy as np
 
 
def jacobiano_fd(f, x, h=1e-6):
    """Aproximación del Jacobiano por diferencias finitas hacia adelante."""
    n = len(x)
    fx = np.atleast_1d(f(x))
    J = np.zeros((len(fx), n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        J[:, j] = (np.atleast_1d(f(x + e)) - fx) / h
    return J
 
 
def broyden(f, x0, H0=None, tol=1e-10, max_iter=100):
    x = np.atleast_1d(np.asarray(x0, dtype=float))
    # H aproxima la INVERSA del Jacobiano
    H = np.linalg.inv(jacobiano_fd(f, x)) if H0 is None else np.asarray(H0, dtype=float)
    fx = np.atleast_1d(f(x))
 
    print(f"{'k':>3} | {'x_k':^32} | {'||f(x_k)||':>12} | {'||d_k||':>12}")
    print("-" * 68)
    print(f"{0:>3} | {np.array2string(x, precision=10):^32} | {np.linalg.norm(fx):>12.4e} | {'-':>12}")
 
    for k in range(max_iter):
        x_new = x - H @ fx                           # paso 2
        d = x_new - x                                # paso 3
        f_new = np.atleast_1d(f(x_new))
        v = f_new - fx                               # paso 4
        Hv = H @ v
        H = H + np.outer(d - Hv, d @ H) / (d @ Hv)   # paso 5
 
        x, fx = x_new, f_new
        print(f"{k+1:>3} | {np.array2string(x, precision=10):^32} | "
              f"{np.linalg.norm(fx):>12.4e} | {np.linalg.norm(d):>12.4e}")
 
        if np.linalg.norm(d) < tol or np.linalg.norm(fx) < tol:
            return x, k + 1
    return x, max_iter
 
 
if __name__ == "__main__":
    f = lambda x: np.array([x[0]**2 + x[1]**2 - 4,
                            math.exp(x[0]) + x[1] - 1])
    sol, it = broyden(f, [1.0, 0])
    print("\nSolución:", sol)
    print("Iteraciones:", it)