import numpy as np

def gauss_total_pivot(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)
    perm_col = np.arange(n)
    perm_row = np.arange(n)
    
    for i in range(n):
        max_val = 0
        max_row = i
        max_col = i
        
        for k in range(i, n):
            for j in range(i, n):
                if abs(A[k, j]) > max_val:
                    max_val = abs(A[k, j])
                    max_row = k
                    max_col = j
        
        if max_val < 1e-10:
            raise ValueError(f"Matriz singular o muy mal condicionada")
        
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        perm_row[[i, max_row]] = perm_row[[max_row, i]]
        
        A[:, [i, max_col]] = A[:, [max_col, i]]
        perm_col[[i, max_col]] = perm_col[[max_col, i]]
        
        for k in range(i + 1, n):
            factor = A[k, i] / A[i, i]
            A[k, i:] = A[k, i:] - factor * A[i, i:]
            b[k] = b[k] - factor * b[i]
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(A[i, j] * x[j] for j in range(i + 1, n))) / A[i, i]
    
    x_perm = np.zeros(n)
    for i in range(n):
        x_perm[perm_col[i]] = x[i]
    
    return x_perm, A, perm_row, perm_col


def gauss_total_pivot_det(A):
    A = np.array(A, dtype=float)
    n = len(A)
    det = 1.0
    swaps = 0
    
    for i in range(n):
        max_val = 0
        max_row = i
        max_col = i
        
        for k in range(i, n):
            for j in range(i, n):
                if abs(A[k, j]) > max_val:
                    max_val = abs(A[k, j])
                    max_row = k
                    max_col = j
        
        if max_val < 1e-10:
            return 0.0
        
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            swaps += 1
        
        if max_col != i:
            A[:, [i, max_col]] = A[:, [max_col, i]]
            swaps += 1
        
        det *= A[i, i]
        
        for k in range(i + 1, n):
            factor = A[k, i] / A[i, i]
            A[k, i:] = A[k, i:] - factor * A[i, i:]
    
    return det * ((-1) ** swaps)


if __name__ == "__main__":
    A = np.array([
        [4, 3, 2],
        [6, 3, 5],
        [2, 1, 3]
    ], dtype=float)
    
    b = np.array([1, 2, 3], dtype=float)
    
    print("=" * 70)
    print("ELIMINACIÓN GAUSSIANA CON PIVOTEO TOTAL")
    print("=" * 70)
    
    A_copy = A.copy()
    b_copy = b.copy()
    x, A_upper, perm_row, perm_col = gauss_total_pivot(A_copy, b_copy)
    
    print("\nMatriz triangular superior después de pivoteo:")
    print(A_upper)
    
    print("\nPermutación de filas:")
    print(perm_row)
    
    print("\nPermutación de columnas:")
    print(perm_col)
    
    print("\nSolución x:")
    print(x)
    
    print("\nVerificación A·x:")
    print(A @ x)
    
    print("\nVector b:")
    print(b)
    
    print("\nError ||A·x - b||:")
    print(np.linalg.norm(A @ x - b))
    
    print("\n" + "=" * 70)
    print("DETERMINANTE CON PIVOTEO TOTAL")
    print("=" * 70)
    
    det_gauss = gauss_total_pivot_det(A.copy())
    det_numpy = np.linalg.det(A)
    
    print(f"\nDeterminante (Gaussiana): {det_gauss:.10f}")
    print(f"Determinante (NumPy):     {det_numpy:.10f}")
    print(f"Diferencia: {abs(det_gauss - det_numpy):.2e}")
    
    print("\n" + "=" * 70)
    print("VERIFICACIÓN CON NumPy")
    print("=" * 70)
    
    x_numpy = np.linalg.solve(A, b)
    print("\nSolución (NumPy):")
    print(x_numpy)
    
    print("\nDiferencia |x - x_numpy|:")
    print(np.linalg.norm(x - x_numpy))
    
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"✓ Error en solución: {np.linalg.norm(A @ x - b):.2e}")
    print(f"✓ Error en determinante: {abs(det_gauss - det_numpy):.2e}")
    print(f"✓ Pivoteo total garantiza estabilidad numérica máxima")