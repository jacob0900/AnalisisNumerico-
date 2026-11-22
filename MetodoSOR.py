import numpy as np
 
 
def es_spd_tridiagonal(A, tol=1e-12):
    """Verifica simetria, definida positiva y estructura tridiagonal."""
    sim = np.allclose(A, A.T, atol=tol)
    tri = np.allclose(A - np.tril(np.triu(A, -1), 1), 0, atol=tol)
    dp = sim and np.all(np.linalg.eigvalsh(A) > tol)
    return sim and tri and dp
 
 
def omega_opt_sor(A, rng=None):
    """w = 2/(1+sqrt(1-rho(B_J)^2)) si A es SPD tridiagonal; si no, w aleatorio en (0,2)."""
    if es_spd_tridiagonal(A):
        D = np.diag(np.diag(A))
        BJ = np.eye(A.shape[0]) - np.linalg.solve(D, A)
        rho = np.abs(np.linalg.eigvals(BJ).real).max()
        return 2.0 / (1.0 + np.sqrt(1.0 - rho ** 2)), rho, True
    rng = np.random.default_rng() if rng is None else rng
    return rng.uniform(0, 2), None, False
 
 
def matriz_sor(A, w):
    """B_SOR = (D + wL)^{-1} [(1-w)D - wU]."""
    D, L, U = np.diag(np.diag(A)), np.tril(A, -1), np.triu(A, 1)
    return np.linalg.solve(D + w * L, (1 - w) * D - w * U)
 
 
def alpha_contraccion(B):
    """Constante de contractilidad: ||B||_2 si es < 1; si no, se usa rho(B) (tasa asintotica)."""
    n2 = np.linalg.norm(B, 2)
    if n2 < 1:
        return n2, "||B_SOR||_2"
    return np.abs(np.linalg.eigvals(B)).max(), "rho(B_SOR)"
 
 
def sor(A, b, w, ks, xstar, tol=1e-8, x0=None):
    """Itera SOR hasta max(ks). Detecta convergencia con la estimacion (2) de Banach:
       ||x_k - x*|| <= alpha/(1-alpha) * ||x_k - x_{k-1}||."""
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, float)
    alpha, etiqueta = alpha_contraccion(matriz_sor(A, w))
    if alpha >= 1:
        raise ValueError(f"alpha = {alpha:.6f} >= 1: el metodo no converge con w = {w:.6f}")
    c = alpha / (1 - alpha)
 
    reg, k_conv = {}, None
    for k in range(1, max(ks) + 1):
        x_ant = x.copy()
        for i in range(n):
            s = A[i] @ x - A[i, i] * x[i]
            x[i] = (1 - w) * x[i] + w * (b[i] - s) / A[i, i]
        cota = c * np.linalg.norm(x - x_ant, 2)
        if k_conv is None and cota < tol:
            k_conv = k
            print(f"Convergencia detectada en k = {k_conv} con cota = {cota:.6e} < tol = {tol:g}")
        if k in ks:
            reg[k] = (np.linalg.norm(x - xstar, 2), cota)
    return reg, k_conv, alpha, etiqueta
 
 
if __name__ == "__main__":
    n, tol = 3, 1e-6
    A = np.array([[38.0, 36.0,  27.0],
                      [36.0, 47.0, 18.0],
                      [27.0, 18.0,  26.0]])
    b = np.array([101.0, 101.0, 71.0])
    xstar = np.array([1.0, 1.0, 1.0])

    w, rho_J, cumple = omega_opt_sor(A, rng=np.random.default_rng(0))
    reg, k_conv, alpha, etiqueta = sor(A, b, w, [20, 40, 80, 160, 201], xstar, tol)
 
    print(f"n = {n}   SPD y tridiagonal: {cumple}"
          + (f"   rho(B_J) = {rho_J:.6f}" if cumple else "   -> w aleatorio en (0,2)"))
    print(f"w = {w:.6f}   alpha = {etiqueta} = {alpha:.6f}"
          f"   alpha/(1-alpha) = {alpha/(1-alpha):.4f}   tol = {tol:g}\n")
 
    print(f"{'k_max':>7} {'||x_k - x*||_2':>18} {'cota a posteriori':>20}")
    print("-" * 47)
    for k, (e, cota) in reg.items():
        print(f"{k:>7} {e:>18.6e} {cota:>20.6e}")
 
    print("\nConvergencia detectada en k =", k_conv if k_conv else
          f"ninguna (la cota nunca bajo de {tol:g} en {max(reg)} iteraciones)")