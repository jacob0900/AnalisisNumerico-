# Análisis Numérico

Implementaciones en Python de los métodos vistos en el curso de **Análisis Numérico**: solución de sistemas lineales (métodos directos e iterativos), almacenamiento de matrices dispersas, normas matriciales y búsqueda de raíces de ecuaciones y sistemas no lineales.

Cada script es autocontenido: define el método como una función y, al ejecutarlo, corre un ejemplo que imprime la tabla de iteraciones, la solución y una verificación contra NumPy o SciPy.

## Contenido

### Sistemas lineales: métodos directos

| Archivo | Descripción |
|---|---|
| `EliminacionGaussiana.py` | Eliminación gaussiana con pivoteo total; resuelve `Ax = b` y calcula el determinante, comparando con `numpy.linalg`. |
| `FactorizacionLU` | Factorización `A = LU` (Doolittle) con sustitución hacia adelante y hacia atrás. |
| `PA=LU.py` | Factorización `PA = LU` con pivoteo parcial. |
| `Factorizacion_Cholonsky` | Factorización de Cholesky `A = LLᵀ` para matrices simétricas definidas positivas, verificada con `numpy.linalg.cholesky`. |

### Sistemas lineales: métodos iterativos

| Archivo | Descripción |
|---|---|
| `Jacobi.py` | Método de Jacobi sobre una matriz densa. |
| `JacobiMatrizDispersa.py` | Jacobi en versión densa y en versión optimizada con almacenamiento CSR; compara iteraciones, error y memoria en una matriz tridiagonal de n = 100. |
| `MetodoJOR.py` | Método JOR (Jacobi sobrerrelajado) con ω óptimo calculado a partir del espectro de la matriz de Jacobi y criterio de parada por la cota a posteriori de Banach. |
| `MetodoSOR.py` | Método SOR; usa el ω óptimo cuando la matriz es SPD y tridiagonal, y reporta el error y la cota a posteriori por iteración. |
| `Gradiente.py` | Método del gradiente (descenso más rápido) para matrices SPD, con historial de normas del residuo. |
| `GradienteConjugado.py` | Gradiente descendente y gradiente conjugado; verifica la ortogonalidad de los residuos y la A-conjugación de las direcciones. |

### Matrices dispersas y normas

| Archivo | Descripción |
|---|---|
| `MatrizDispersa.py` | Conversión de matriz densa a formato CSR y producto matriz-vector; compara memoria, número de multiplicaciones y tiempo frente a la versión densa. |
| `csr_documento.pdf` | Documento que explica el formato CSR, su implementación en Python y la prueba con una matriz tridiagonal. |
| `NormasMatriciales.py` | Cálculo de las normas 1, 2, infinito y de Frobenius de una matriz. |

### Ecuaciones no lineales

| Archivo | Descripción |
|---|---|
| `RegulaFasi.py` | Comparación entre bisección y regula falsi para `f(x) = 2x³ + x − 2`, con la raíz de referencia obtenida por `scipy.optimize.brentq`. |
| `PuntoFijo.py` | Iteración de punto fijo con verificación numérica de las hipótesis del teorema de Banach (automapeo y contractividad). |
| `punti.py` | Versión compacta de la iteración de punto fijo con tabla de iteraciones y criterio de parada. |

### `3Parcial/`

| Archivo | Descripción |
|---|---|
| `MetodoBiseccion.py` | Bisección con verificación del teorema de Bolzano y número de iteraciones calculado a priori. |
| `METODO_NEWTON_Simple.py` | Método de Newton en una variable. |
| `Metodo_Newton_Multivariable.py` | Método de Newton para sistemas `F(x) = 0` en ℝⁿ, probado con un sistema 2×2 desde dos datos iniciales. |
| `metodo_broyden.py` | Método de Broyden (cuasi-Newton) actualizando la inversa del jacobiano, con jacobiano inicial por diferencias finitas. |

## Requisitos

- Python 3.9 o superior
- NumPy
- SciPy y pandas (solo para `RegulaFasi.py`)

```bash
pip install numpy scipy pandas
```

## Uso

Cada archivo se ejecuta directamente y muestra su ejemplo:

```bash
python MetodoSOR.py
python 3Parcial/metodo_broyden.py
python FactorizacionLU
```

Las funciones también se pueden importar para resolver otros problemas, por ejemplo:

```python
import numpy as np
from GradienteConjugado import gradiente_conjugado

A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
x, iteraciones, historial, R, P = gradiente_conjugado(A, b)
```

## Autor

- **Jacobo Londoño** — [jacob0900](https://github.com/jacob0900) · [LinkedIn](https://www.linkedin.com/in/jacobo-londo%C3%B1o-baquero-a2b86733a/)
