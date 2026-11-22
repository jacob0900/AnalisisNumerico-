"""
Método de Jacobi: versión básica (densa) y versión optimizada con formato CSR.

Sistema: A x = b,  con A de diagonal no nula.
Iteración:  x_i^(k+1) = ( b_i - sum_{j != i} a_ij x_j^(k) ) / a_ii
"""

import numpy as np


# ---------------------------------------------------------------- #
# 1. Versión básica (matriz densa)
# ---------------------------------------------------------------- #
def jacobi(A, b, x0=None, tol=1e-10, max_iter=1000):
    """Método de Jacobi estándar sobre una matriz densa.

    Retorna (x, k, err) con k = iteraciones realizadas.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.size

    if np.any(np.abs(np.diag(A)) < 1e-15):
        raise ValueError("Hay elementos nulos en la diagonal de A.")

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
            return x, k, err

    return x, max_iter, err


# ---------------------------------------------------------------- #
# 2. Formato CSR (Compressed Sparse Row)
# ---------------------------------------------------------------- #
class CSR:
    """Almacenamiento CSR de las entradas no nulas de una matriz.

    val     : valores no nulos, recorridos por filas
    col_ind : columna de cada valor
    row_ptr : row_ptr[i] es el inicio de la fila i en val (tamaño n+1)
    """

    def __init__(self, val, col_ind, row_ptr, shape):
        self.val = np.asarray(val, dtype=float)
        self.col_ind = np.asarray(col_ind, dtype=int)
        self.row_ptr = np.asarray(row_ptr, dtype=int)
        self.shape = shape

    @classmethod
    def from_dense(cls, A, tol=0.0):
        A = np.asarray(A, dtype=float)
        n, m = A.shape
        val, col_ind, row_ptr = [], [], [0]
        for i in range(n):
            for j in range(m):
                if abs(A[i, j]) > tol:
                    val.append(A[i, j])
                    col_ind.append(j)
            row_ptr.append(len(val))
        return cls(val, col_ind, row_ptr, (n, m))

    def matvec(self, x):
        """Producto matriz-vector: solo recorre las entradas no nulas."""
        n = self.shape[0]
        y = np.zeros(n)
        for i in range(n):
            s = 0.0
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                s += self.val[k] * x[self.col_ind[k]]
            y[i] = s
        return y

    def diagonal(self):
        n = self.shape[0]
        d = np.zeros(n)
        for i in range(n):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                if self.col_ind[k] == i:
                    d[i] = self.val[k]
                    break
        return d

    def to_dense(self):
        A = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                A[i, self.col_ind[k]] = self.val[k]
        return A

    @property
    def nnz(self):
        return self.val.size


def jacobi_csr(A_csr, b, x0=None, tol=1e-10, max_iter=1000):
    """Jacobi sobre una matriz en formato CSR.

    Usa la forma equivalente  x^(k+1) = x^(k) + D^{-1} ( b - A x^(k) ),
    que requiere un solo producto matriz-vector por iteración.
    """
    b = np.asarray(b, dtype=float)
    n = b.size
    d = A_csr.diagonal()

    if np.any(np.abs(d) < 1e-15):
        raise ValueError("Hay elementos nulos en la diagonal de A.")

    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    for k in range(1, max_iter + 1):
        r = b - A_csr.matvec(x)      # residuo
        dx = r / d                   # corrección
        x = x + dx

        err = np.linalg.norm(dx, np.inf)
        if err < tol:
            return x, k, err

    return x, max_iter, err


# ---------------------------------------------------------------- #
# 3. Prueba: matriz tridiagonal
# ---------------------------------------------------------------- #
def tridiagonal_densa(n, a=-1.0, d=4.0, c=-1.0):
    """Tridiagonal con subdiagonal a, diagonal d y superdiagonal c."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = d
        if i > 0:
            A[i, i - 1] = a
        if i < n - 1:
            A[i, i + 1] = c
    return A


def tridiagonal_csr(n, a=-1.0, d=4.0, c=-1.0):
    """Construye la tridiagonal directamente en CSR, sin pasar por la densa."""
    val, col_ind, row_ptr = [], [], [0]
    for i in range(n):
        if i > 0:
            val.append(a)
            col_ind.append(i - 1)
        val.append(d)
        col_ind.append(i)
        if i < n - 1:
            val.append(c)
            col_ind.append(i + 1)
        row_ptr.append(len(val))
    return CSR(val, col_ind, row_ptr, (n, n))


if __name__ == "__main__":
    n = 100
    A = tridiagonal_densa(n)
    A_csr = tridiagonal_csr(n)

    x_exacta = np.ones(n)
    b = A @ x_exacta

    x1, k1, e1 = jacobi(A, b, tol=1e-12)
    x2, k2, e2 = jacobi_csr(A_csr, b, tol=1e-12)

    print(f"n = {n}")
    print(f"Almacenamiento denso : {A.size} entradas")
    print(f"Almacenamiento CSR   : {A_csr.nnz} entradas no nulas")
    print()
    print(f"{'método':<14}{'iter':>6}{'err. iter':>14}{'err. real':>14}")
    print(f"{'básico':<14}{k1:>6}{e1:>14.2e}"
          f"{np.linalg.norm(x1 - x_exacta, np.inf):>14.2e}")
    print(f"{'CSR':<14}{k2:>6}{e2:>14.2e}"
          f"{np.linalg.norm(x2 - x_exacta, np.inf):>14.2e}")
    print()
    print(f"||x_basico - x_csr||_inf = {np.linalg.norm(x1 - x2, np.inf):.2e}")