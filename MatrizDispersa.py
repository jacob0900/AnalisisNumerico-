

import time
import numpy as np


def tridiagonal_densa(n=50, sub=-1.0, diag=2.0, sup=-1.0):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = diag
        if i + 1 < n:
            A[i + 1, i] = sub
            A[i, i + 1] = sup
    return A


def densa_a_csr(A):
    """Comprime una matriz densa a CSR y devuelve (values, col_idx, row_ptr).

    Recorre la matriz por filas; cada valor distinto de cero se agrega a values
    junto con su columna en col_idx. Al terminar cada fila se anota en row_ptr
    cuantos no nulos se llevan acumulados, que es justo donde empezara la
    siguiente fila.
    """
    values, col_idx, row_ptr = [], [], [0]
    for fila in A:
        for j, v in enumerate(fila):
            if v != 0.0:               # los ceros no se guardan: ahi esta el ahorro
                values.append(v)
                col_idx.append(j)
        row_ptr.append(len(values))    # frontera entre esta fila y la siguiente
    return values, col_idx, row_ptr


def matvec_csr(values, col_idx, row_ptr, x):
    """Calcula y = A*x con A en formato CSR. Costo: nnz multiplicaciones."""
    n = len(row_ptr) - 1               # row_ptr tiene n+1 entradas
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        # solo los no nulos de la fila i; si la fila esta vacia el rango tampoco
        # itera y y[i] queda en 0 sin trabajo extra
        for k in range(row_ptr[i], row_ptr[i + 1]):
            s += values[k] * x[col_idx[k]]   # col_idx[k] dice contra que x[j] va
        y[i] = s
    return y


def matvec_denso(A, x):
    """Calcula y = A*x recorriendo toda la matriz: n*m multiplicaciones.

    Sirve de referencia: hace el mismo trabajo util que matvec_csr pero ademas
    multiplica por todos los ceros.
    """
    y = [0.0] * len(A)
    for i, fila in enumerate(A):
        s = 0.0
        for j, a in enumerate(fila):   # recorre las m columnas, sean cero o no
            s += a * x[j]
        y[i] = s
    return y


def cronometrar(f, reps):
    """Ejecuta f() reps veces y devuelve (tiempo promedio en segundos, resultado)."""
    t0 = time.perf_counter()
    for _ in range(reps):
        y = f()
    return (time.perf_counter() - t0) / reps, y


if __name__ == "__main__":
    # Prueba pedida: tridiagonal de n = 100 (nnz = 3n-2 = 298 sobre 10 000 entradas).
    n, reps = 100, 200
    A = tridiagonal_densa(n)
    values, col_idx, row_ptr = densa_a_csr(A)
    nnz = row_ptr[-1]

    x = list(np.random.default_rng(0).standard_normal(n))  # semilla fija: reproducible
    A_list = A.tolist()                # listas puras para comparar contra el CSR
                                       # en igualdad de condiciones (sin vectorizar)

    t_csr, y_csr = cronometrar(lambda: matvec_csr(values, col_idx, row_ptr, x), reps)
    t_den, _ = cronometrar(lambda: matvec_denso(A_list, x), reps)

    # Validacion: el resultado debe coincidir con el producto de numpy salvo
    # error de redondeo de punto flotante (~1e-16).
    err = np.max(np.abs(np.array(y_csr) - A @ np.array(x)))

    print(f"n = {n}, nnz = {nnz} de {n*n} ({100*nnz/n**2:.2f} % densidad)")
    print(f"memoria     denso {n*n:>8}   CSR {2*nnz+n+1:>8}")
    print(f"multiplic.  denso {n*n:>8}   CSR {nnz:>8}   ({n*n/nnz:.1f}x menos)")
    print(f"tiempo (ms) denso {t_den*1e3:>8.3f}   CSR {t_csr*1e3:>8.3f}   ({t_den/t_csr:.1f}x)")
    print(f"error max vs A@x  {err:.2e}")