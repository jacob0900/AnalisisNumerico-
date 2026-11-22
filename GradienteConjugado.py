"""
Método del Gradiente (Descenso Más Rápido) para resolver Ax = b.

Minimiza la función cuadrática
    Phi(x) = 1/2 x^T A x - x^T b,
cuyo gradiente es  grad Phi(x) = A x - b = -r.  Si A es simétrica y
definida positiva, el único minimizador de Phi es la solución de Ax = b.
"""

import numpy as np


def gradiente_descendente(A, b, x0=None, tol=1e-8, max_iter=1000):
    """
    Resuelve Ax = b con el método de descenso más rápido.

    Parámetros
    ----------
    A : ndarray (n, n)
        Matriz simétrica definida positiva.
    b : ndarray (n,)
        Vector del lado derecho.
    x0 : ndarray (n,), opcional
        Aproximación inicial. Si es None se usa el vector cero.
    tol : float
        Tolerancia sobre ||r_k||_2.
    max_iter : int
        Número máximo de iteraciones.

    Retorna
    -------
    x : ndarray (n,)
        Solución aproximada.
    k : int
        Número de iteraciones realizadas.
    historial : list[float]
        Normas del residuo [||r_0||_2, ||r_1||_2, ...].
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()

    r = b - A @ x                      # r_0
    norma_r = np.linalg.norm(r)
    historial = [norma_r]

    k = 0
    while norma_r >= tol and k < max_iter:
        Ar = A @ r                     # único producto matriz-vector por iteración
        alpha = (norma_r ** 2) / (r @ Ar)

        x = x + alpha * r              # x_{k+1} = x_k + alpha_k r_k
        r = r - alpha * Ar             # r_{k+1} = b - A x_{k+1}, reutilizando A r_k

        norma_r = np.linalg.norm(r)
        historial.append(norma_r)
        k += 1

    return x, k, historial


def gradiente_conjugado(A, b, x0=None, tol=1e-8, max_iter=1000):
    """
    Resuelve Ax = b con el método del Gradiente Conjugado.

    Mismos parámetros que gradiente_descendente.

    Retorna
    -------
    x : ndarray (n,)
        Solución aproximada.
    k : int
        Número de iteraciones realizadas.
    historial : list[float]
        Normas del residuo [||r_0||_2, ||r_1||_2, ...].
    R : ndarray (k+1, n)
        Residuos r_0, ..., r_k (deben ser mutuamente ortogonales).
    P : ndarray (k, n)
        Direcciones p_0, ..., p_{k-1} (deben ser A-conjugadas).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()

    r = b - A @ x
    p = r.copy()
    rr = r @ r
    historial = [np.sqrt(rr)]
    residuos, direcciones = [r.copy()], []

    k = 0
    while historial[-1] >= tol and k < max_iter:
        Ap = A @ p
        alpha = rr / (p @ Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        rr_nuevo = r @ r
        beta = rr_nuevo / rr
        direcciones.append(p.copy())
        p = r + beta * p
        rr = rr_nuevo

        historial.append(np.sqrt(rr))
        residuos.append(r.copy())
        k += 1

    return x, k, historial, np.array(residuos), np.array(direcciones)


def max_fuera_diagonal(G):
    """Máximo |G_ij| con i != j de una matriz de Gram normalizada."""
    d = np.sqrt(np.abs(np.diag(G)))
    G = G / np.outer(d, d)
    np.fill_diagonal(G, 0.0)
    return np.max(np.abs(G))


def matriz_tridiagonal(n):
    """Matriz tridiagonal SPD con 2 en la diagonal y -1 en las subdiagonales."""
    return 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


if __name__ == "__main__":
    n = 50
    A = np.array([[38.0, 36.0,  27.0],
                              [36.0, 47.0, 18.0],
                              [27.0, 18.0,  26.0]])
    b = np.array([101.0, 101.0, 71.0])
    xstar = np.array([1.0, 1.0, 1.0])

    x_exacta = np.linalg.solve(A, b)

    for max_iter in (1000, 20000):
        x_num, iters, hist = gradiente_descendente(A, b, tol=1e-6, max_iter=max_iter)
        error = np.linalg.norm(x_num - x_exacta)
        print(f"max_iter = {max_iter}")
        print(f"  Iteraciones realizadas : {iters}")
        print(f"  ||r_final||_2          : {hist[-1]:.3e}")
        print(f"  ||x_num - x_exacta||_2 : {error:.3e}")

    print(f"\nNúmero de condición de A : {np.linalg.cond(A):.1f}")

    # --- Gradiente Conjugado ---
    x_gc, iters, hist, R, P = gradiente_conjugado(A, b, tol=1e-8)
    print("\nGradiente Conjugado")
    print(f"  Iteraciones realizadas : {iters}")
    print(f"  ||r_final||_2          : {hist[-1]:.3e}")
    print(f"  ||x_num - x_exacta||_2 : {np.linalg.norm(x_gc - x_exacta):.3e}")

    # Ortogonalidad: r_i^T r_j = 0  y  p_i^T A p_j = 0  para i != j
    Rk = R[:-1]  # se excluye el último residuo (~0) para evitar dividir por ~0
    print(f"  max |r_i^T r_j| (norm.)   : {max_fuera_diagonal(Rk @ Rk.T):.3e}")
    print(f"  max |p_i^T A p_j| (norm.) : {max_fuera_diagonal(P @ A @ P.T):.3e}")