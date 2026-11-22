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


def imprimir_tabla(historial):
    print(f"{'k':>3} | {'x_k':>18} | {'g(x_k)':>18} | {'|x_k - x_(k-1)|':>16} | {'f(x_k)':>14}")
    print("-" * 82)
    for fila in historial:
        gx  = "-" if fila["gx"] is None else f"{fila['gx']:.12f}"
        err = "-" if fila["error"] is None else f"{fila['error']:.6e}"
        fx  = "-" if fila["fx"] is None else f"{fila['fx']:.6e}"
        print(f"{fila['k']:>3} | {fila['x']:>18.12f} | {gx:>18} | {err:>16} | {fx:>14}")