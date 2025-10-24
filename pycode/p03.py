"""
Band-limited interpolation
"""

import numpy as np
import matplotlib.pyplot as plt

h = 0.5
xmax = 10
x = np.arange(-xmax, xmax+h, h)
xx = np.arange(-xmax-h/20, xmax+h/20, h/20)

plt.figure(figsize=(12, 7))
for iplt in range(3):
    plt.subplot(3, 1, iplt+1)
    if iplt == 0:
        v = np.array(x == 0, dtype=np.float64)                       # delta function
    elif iplt == 1:
        v = np.array(np.abs(x) <= 3, dtype=np.float64)               # square wave
    elif iplt == 2:
        v = np.maximum(0, 1 - np.abs(x)/3)                      # hat function
        
    plt.plot(x, v, 'k.', markersize=10)
    p = np.zeros_like(xx)
    for i in range(len(x)):
        p = p + v[i] * np.sin(np.pi * (xx - x[i])/h) / (np.pi * (xx - x[i]) /h)

    plt.plot(xx, p, linewidth=2)
    plt.xticks(x[::2])
    plt.yticks(np.linspace(0, 1, 3))
    plt.grid(True)
    
plt.tight_layout()
plt.show()
