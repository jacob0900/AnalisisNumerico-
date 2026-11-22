import numpy as np

def newton_nd(F, J, x0, max_iter=100, tol=1e-10):
    """
    Método de Newton multivariado: F(x) = 0,  F: R^n -> R^n

    Parámetros:
    -----------
    F       : función que retorna un array de tamaño n
    J       : jacobiano, retorna una matriz n x n
    x0      : vector inicial
    max_iter: número máximo de iteraciones
    tol     : tolerancia para ||F(x)||_2
    """
    x = np.array(x0, dtype=float)
    n = len(x)

    encabezado = " | ".join(f"{f'x{i+1}':>16}" for i in range(n))
    print(f"{'Iter':>5} | {encabezado} | {'||F(x)||':>12}")
    print("-" * (10 + 19 * n + 12))

    for k in range(max_iter):
        Fx = np.array(F(x), dtype=float)
        norm_f = np.linalg.norm(Fx)

        valores = " | ".join(f"{xi:>16.10f}" for xi in x)
        print(f"{k:>5} | {valores} | {norm_f:>12.4e}")

        # Criterio 1: ||F(x)|| < tol
        if norm_f < tol:
            print(f"\n✓ Convergió en {k} iteraciones  ||F(x)|| = {norm_f:.2e} < {tol}")
            return x, {"converged": True, "iterations": k, "norm_f": norm_f, "reason": "norm_f"}

        # Resolver J(x) Δx = -F(x)
        try:
            delta = np.linalg.solve(np.array(J(x), dtype=float), -Fx)
        except np.linalg.LinAlgError:
            print("\n✗ Jacobiano singular, el método falla.")
            return x, {"converged": False, "iterations": k, "norm_f": norm_f, "reason": "singular_jacobian"}

        x = x + delta

    # Criterio 2: máximo de iteraciones
    norm_f = np.linalg.norm(F(x))
    print(f"\n✗ Máximo de iteraciones ({max_iter}) alcanzado.  ||F(x)|| = {norm_f:.2e}")
    return x, {"converged": False, "iterations": max_iter, "norm_f": norm_f, "reason": "max_iter"}


# ── Sistema 2x2 ───────────────────────────────────────────────
#   x1^2 + 4 x2^2 - 4 = 0
#   x1 - x2           = 0
F = lambda x: np.array([x[0]**2 + 4*x[1]**2 - 4,
                        x[0] - x[1]])

J = lambda x: np.array([[2*x[0], 8*x[1]],
                        [1.0,   -1.0  ]])

for x0 in [(1, 0), (-2, 0)]:
    print("=" * 60)
    print(f"Dato inicial x0 = {x0}")
    print("=" * 60)
    sol, info = newton_nd(F, J, x0)
    print(f"Iteraciones: {info['iterations']}")
    print(f"Solución: x* = ({sol[0]:.12f}, {sol[1]:.12f})\n")