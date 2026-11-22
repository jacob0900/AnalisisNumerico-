import math

def biseccion(f, a, b, eps):
    # Teorema de Bolzano: debe haber cambio de signo en [a, b]
    if f(a) * f(b) >= 0:
        raise ValueError("No se cumple Bolzano: f(a)·f(b) debe ser < 0")

    # Menor entero k que cumple log2((b - a) / eps) < k
    k = math.floor(math.log2((b - a) / eps)) + 1
    print(f"Número de iteraciones: k = {k}\n")
    print(f"{'i':>3} {'a':>12} {'b':>12} {'c':>12} {'f(c)':>14}")

    for i in range(1, k + 1):
        c = (a + b) / 2
        print(f"{i:>3} {a:>12.8f} {b:>12.8f} {c:>12.8f} {f(c):>14.6e}")
        if f(c) == 0:
            return c, i
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2, k


raiz, k = biseccion(lambda x: x**3 + x -1 , 0, 1, 1e-3)
print(f"\nRaíz ≈ {raiz:.8f} en {k} iteraciones")