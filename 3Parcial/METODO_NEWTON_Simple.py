def newton_1d(f, df, x0, max_iter=100, tol=1e-6):
    """
    Método de Newton unidimensional: f(x) = 0
    
    Parámetros:
    -----------
    f      : función f(x)
    df     : derivada f'(x)
    x0     : valor inicial
    max_iter: número máximo de iteraciones
    tol    : tolerancia para |f(x)|
    """
    x = float(x0)

    print(f"{'Iter':>5} | {'x':>20} | {'f(x)':>15} | {'|f(x)|':>15}")
    print("-" * 62)

    for k in range(max_iter):
        fx  = f(x)
        norm_f = abs(fx)

        print(f"{k:>5} | {x:>20.10f} | {fx:>15.6e} | {norm_f:>15.6e}")

        # Criterio 1: |f(x)| < tol
        if norm_f < tol:
            print(f"\n✓ Convergió en {k} iteraciones  |f(x)| = {norm_f:.2e} < {tol}")
            return x, {"converged": True, "iterations": k, "norm_f": norm_f, "reason": "norm_f"}

        dfx = df(x)
        if dfx == 0:
            print("\n✗ Derivada nula, el método falla.")
            return x, {"converged": False, "iterations": k, "norm_f": norm_f, "reason": "zero_derivative"}

        x = x - fx / dfx

    # Criterio 2: máximo de iteraciones
    norm_f = abs(f(x))
    print(f"\n✗ Máximo de iteraciones ({max_iter}) alcanzado.  |f(x)| = {norm_f:.2e}")
    return x, {"converged": False, "iterations": max_iter, "norm_f": norm_f, "reason": "max_iter"}


# ── Ejemplo 1:  x**2 - 2x" = 0  (raíz ≈ 1.5214) ────────────────────────────
print("=" * 62)
print("Ejemplo 1:  f(x) = x**3- 2*x + 2")
print("=" * 62)
sol, info = newton_1d(
    f  = lambda x: x**3 - 5*x ,
    df = lambda x: 3*x**2 - 5,
    x0 = 2
)

print(f"Número de iteraciones para converger: {info['iterations']}")
print(f"Solución: x* = {sol:.12f}")





