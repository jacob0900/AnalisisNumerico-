import numpy as np
 
 
def matriz_jor(A, w):
    """B_Jw = (1-w)I + w*B_J, matriz de iteracion del JOR."""
    D = np.diag(np.diag(A))
    BJ = np.eye(A.shape[0]) - np.linalg.solve(D, A)
    return (1 - w) * np.eye(A.shape[0]) + w * BJ
 
 
def omega_opt(A):
    """omega optimo del JOR: 2 / (2 - lam_max - lam_min) sobre el espectro de B_J."""
    D = np.diag(np.diag(A))
    BJ = np.eye(A.shape[0]) - np.linalg.solve(D, A)
    lam = np.linalg.eigvals(BJ).real
    return 2.0 / (2.0 - lam.max() - lam.min()), lam.min(), lam.max()
 
 
def jor(A, b, w, ks, xstar, tol=1e-8, x0=None):
    """Itera JOR hasta max(ks). Detecta convergencia con la estimacion (2) de Banach:
       ||x_k - x*|| <= alpha/(1-alpha) * ||x_k - x_{k-1}||,  alpha = ||B_Jw||_2."""
    d = np.diag(A).astype(float)
    x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, float)
    alpha = np.linalg.norm(matriz_jor(A, w), 2)
    if alpha >= 1:
        raise ValueError(f"alpha = {alpha:.6f} >= 1: T no es contraccion en ||.||_2")
    c = alpha / (1 - alpha)
 
    reg, k_conv = {}, None
    for k in range(1, max(ks) + 1):
        x_new = x + w * (b - A @ x) / d
        cota = c * np.linalg.norm(x_new - x, 2)
        x = x_new
        if k_conv is None and cota < tol:
            k_conv = k
        if k in ks:
            reg[k] = (np.linalg.norm(x - xstar, 2), cota)
    return reg, k_conv, alpha
 
 
if __name__ == "__main__":
    n, tol = 50, 1e-8
    A = np.diag(2.0 * np.ones(n)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)
    b = np.zeros(n)
    b[0] = b[-1] = 1.0
    xstar = np.ones(n)
 
    w, lmin, lmax = omega_opt(A)
    reg, k_conv, alpha = jor(A, b, w, [20, 40, 80, 160, 320], xstar, tol)
 
    print(f"n = {n}   lam_min = {lmin:.6f}   lam_max = {lmax:.6f}   w_opt = {w:.6f}")
    print(f"alpha = ||B_Jw||_2 = {alpha:.6f}   alpha/(1-alpha) = {alpha/(1-alpha):.4f}   tol = {tol:g}\n")
 
    print(f"{'k_max':>7} {'||x_k - x*||_2':>18} {'cota a posteriori':>20}")
    print("-" * 47)
    for k, (e, cota) in reg.items():
        print(f"{k:>7} {e:>18.6e} {cota:>20.6e}")
 
    print("\nConvergencia detectada en k =", k_conv if k_conv else
          f"ninguna (la cota nunca bajo de {tol:g} en {max(reg)} iteraciones)")