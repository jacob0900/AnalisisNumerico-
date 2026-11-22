import numpy as np

def plu(A, tol=1e-12):
    """PA = LU con pivoteo parcial (Doolittle)."""
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("La matriz debe ser cuadrada")

    U = A.copy()
    L = np.zeros((n, n))
    P = np.eye(n)

    for k in range(n):
        # pivoteo parcial: fila con el mayor |valor| en la columna k
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[p, k]) < tol:
            raise ValueError(f"Matriz singular (pivote ~0 en la columna {k})")

        if p != k:                       # intercambio de filas
            U[[k, p], k:] = U[[p, k], k:]
            L[[k, p], :k] = L[[p, k], :k]
            P[[k, p], :] = P[[p, k], :]

        L[k, k] = 1.0
        for i in range(k + 1, n):        # eliminación
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0

    return P, L, U