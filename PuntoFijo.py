"""
Método de Iteración de Punto Fijo
=================================

Resuelve f(x) = 0 reescribiendo el problema como x = g(x) e iterando
x_{k+1} = g(x_k).

Fundamento (Teorema del Punto Fijo de Banach):
    Si g: [a, b] -> [a, b] (automapeo) y existe alpha < 1 tal que
    |g'(x)| <= alpha para todo x en [a, b] (contractividad), entonces:
      1. g tiene un único punto fijo x* en [a, b].
      2. La sucesión x_{k+1} = g(x_k) converge a x* para todo x0 en [a, b].

Problema de prueba:
    f(x) = e^x - 2x - 1 = 0 en [1, 2]
    g(x) = ln(2x + 1),  g'(x) = 2 / (2x + 1),  alpha = g'(1) = 2/3

    Se descarta g(x) = (e^x - 1)/2 porque g'(x) = e^x / 2 > 1 en [1, 2].
"""

import math
from typing import Callable, Optional

import numpy as np


# ----------------------------------------------------------------------
# Verificación de hipótesis
# ----------------------------------------------------------------------
def derivada_numerica(g: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """Aproxima g'(x) por diferencias centradas (error O(h^2))."""
    return (g(x + h) - g(x - h)) / (2 * h)


def verificar_banach(
    g: Callable[[float], float],
    a: float,
    b: float,
    dg: Optional[Callable[[float], float]] = None,
    n_puntos: int = 10_001,
) -> dict:
    """
    Verifica numéricamente las hipótesis de Banach sobre una malla de [a, b].

    Parámetros
    ----------
    g : función de iteración.
    a, b : extremos del intervalo.
    dg : derivada analítica de g (opcional). Si es None se usa diferencias centradas.
    n_puntos : número de puntos de la malla.

    Retorna
    -------
    dict con: rango de g, cumplimiento del automapeo, alpha y cumplimiento
    de la contractividad.
    """
    xs = np.linspace(a, b, n_puntos)
    gx = np.array([g(x) for x in xs])

    if dg is None:
        # Diferencias centradas en el interior; en los bordes se reduce h
        # para no salir del dominio de g.
        h = min(1e-6, (b - a) / 10)
        dgx = np.array([derivada_numerica(g, x, h) for x in xs])
    else:
        dgx = np.array([dg(x) for x in xs])

    g_min, g_max = gx.min(), gx.max()
    alpha = np.abs(dgx).max()

    return {
        "g_min": g_min,
        "g_max": g_max,
        "automapeo": a <= g_min and g_max <= b,
        "alpha": alpha,
        "x_alpha": xs[np.abs(dgx).argmax()],
        "contractiva": alpha < 1,
    }


def imprimir_verificacion(res: dict, a: float, b: float) -> None:
    """Muestra el resultado de verificar_banach."""
    print("Verificación de hipótesis de Banach")
    print("-" * 50)
    print(f"g([{a}, {b}]) = [{res['g_min']:.6f}, {res['g_max']:.6f}]"
          f"  ⊆ [{a}, {b}]? {'Sí' if res['automapeo'] else 'NO'}")
    print(f"alpha = max|g'(x)| = {res['alpha']:.6f} (en x = {res['x_alpha']:.4f})"
          f"  < 1? {'Sí' if res['contractiva'] else 'NO'}")
    print()


# ----------------------------------------------------------------------
# Método de punto fijo
# ----------------------------------------------------------------------
def punto_fijo(g, x0, tol=1e-8, max_iter=100, f=None):
    historial = [{"k": 0, "x": x0, "gx": None, "error": None, "fx": f(x0) if f else None}]
    x_k = x0
    motivo = f"se alcanzó max_iter = {max_iter} sin converger"

    for k in range(1, max_iter + 1):
        x_next = g(x_k)
        historial[-1]["gx"] = x_next          # g(x_k) se guarda en la fila k
        error = abs(x_next - x_k)
        fx = f(x_next) if f else None
        historial.append({"k": k, "x": x_next, "gx": None, "error": error, "fx": fx})
        x_k = x_next

        if error < tol:
            motivo = f"|x_k - x_(k-1)| < tol = {tol:g}"
            break
        if f is not None and abs(fx) < tol:
            motivo = f"|f(x_k)| < tol = {tol:g}"
            break

    historial[-1]["gx"] = g(x_k)              # g en la última fila
    imprimir_tabla(historial)
    print(f"\nCriterio de parada: {motivo}")
    return x_k, len(historial) - 1, historial


def imprimir_tabla(historial: list) -> None:
    """Imprime la tabla k | x_k | |x_k - x_{k-1}| | f(x_k)."""
    print(f"{'k':>3} | {'x_k':>18} | {'|x_k - x_(k-1)|':>16} | {'f(x_k)':>14}")
    print("-" * 61)
    for fila in historial:
        err = "-" if fila["error"] is None else f"{fila['error']:.6e}"
        fx = "-" if fila["fx"] is None else f"{fila['fx']:.6e}"
        print(f"{fila['k']:>3} | {fila['x']:>18.12f} | {err:>16} | {fx:>14}")


# ----------------------------------------------------------------------
# Script de prueba
# ----------------------------------------------------------------------
def main() -> None:
    # Problema de clase
    f = lambda x: math.exp(2*x) -  x - 2
    g = lambda x: math.log( x + 2) / 2
    dg = lambda x: 1 / ( x + 2)
    a, b = 0 , 1.0
    x0, tol, max_iter = 1.5, 1e-8, 100

    # Hipótesis con derivada analítica y numérica
    res = verificar_banach(g, a, b, dg=dg)
    imprimir_verificacion(res, a, b)
    res_num = verificar_banach(g, a, b)
    print(f"alpha con derivada numérica: {res_num['alpha']:.6f}  (teórico 2/3 = {2/3:.6f})\n")

    # Contraejemplo descartado
    g_mala = lambda x: (math.exp(x) - 1) / 2
    res_mala = verificar_banach(g_mala, a, b, dg=lambda x: math.exp(x) / 2)
    print("g(x) = (e^x - 1)/2 (descartada):")
    imprimir_verificacion(res_mala, a, b)

    if not (res["automapeo"] and res["contractiva"]):
        print("g no cumple las hipótesis de Banach; no se garantiza convergencia.")
        return

    # Iteración
    print(f"Iteración de punto fijo: x0 = {x0}, tol = {tol:g}\n")
    x_star, n_iter, _ = punto_fijo(g, x0, tol=tol, max_iter=max_iter, f=f)

    # Resultado final
    print("\nResultado")
    print("-" * 50)
    print(f"x*            = {x_star:.12f}")
    print(f"Iteraciones   = {n_iter}")
    print(f"f(x*)         = {f(x_star):.3e}")
    print(f"|g(x*) - x*|  = {abs(g(x_star) - x_star):.3e}")


if __name__ == "__main__":
    main()