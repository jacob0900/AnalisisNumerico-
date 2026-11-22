import numpy as np

A = np.array([[1, -2, 3],
              [4,  5, -6],
              [-7, 8, 9]], dtype=float)

print("Norma 1   :", np.linalg.norm(A, 1))         # máx. suma de columnas
print("Norma 2   :", np.linalg.norm(A, 2))         # mayor valor singular
print("Frobenius :", np.linalg.norm(A, 'fro'))     # raíz de la suma de cuadrados
print("Infinito  :", np.linalg.norm(A, np.inf))    # máx. suma de filas