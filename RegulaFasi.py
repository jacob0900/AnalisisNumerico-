import numpy as np
from scipy.optimize import brentq
import pandas as pd

f = lambda x: 2*x**3 + x - 2
N = 20

# Bisección
a, b = 0.0, 1.0
xb = []
for n in range(N):
    c = (a + b) / 2
    if f(a) * f(c) < 0:
        b = c
    else:
        a = c
    xb.append(c)

# Regula falsi
a, b = 0.0, 1.0
xr = []
for n in range(N):
    c = (a*f(b) - b*f(a)) / (f(b) - f(a))
    if f(a) * f(c) < 0:
        b = c
    else:
        a = c
    xr.append(c)

# Raíz de referencia y tabla
r = brentq(f, 0, 1, xtol=1e-15)
xb, xr = np.array(xb), np.array(xr)

T = pd.DataFrame({
    "n": np.arange(1, N + 1),
    "Biseccion": xb,
    "ErrorBis": np.abs(xb - r),
    "ReguaFalsi": xr,
    "ErrorRF": np.abs(xr - r),
})   
pd.options.display.float_format = "{:.20f}".format
print(T.to_string(index=False))