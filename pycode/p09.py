"""
Polynomial interpolation in equispaced and Chebyshev pts

@author: Khiem Nguyen
"""

import numpy as np
import matplotlib.pyplot as plt

N = 16
xx = np.linspace(-1.01, 1.01, 202)
plt.figure(figsize=(10, 6), dpi=80)
for i in range(2):
    if i == 0:
        s = 'equispaced points'
        x = -1 + 2.0 * np.arange(0, N+1) / N
    elif i == 1:
        s = 'Chebyshev points'
        x = np.cos(np.pi * np.arange(0, N+1) / N)
    plt.subplot(2,2,i+1)
    u = 1./(1 + 16 * x**2)
    uu = 1./(1 + 16 * xx**2)
    p = np.polyfit(x, u, N)
    pp = np.polyval(p, xx)
    plt.plot(x, u, 'o', markersize=4)
    plt.plot(xx, pp, 'k-', linewidth=2)
    plt.axis([-1.1, 1.1, -1, 1.5])
    plt.title(s)
    error = np.linalg.norm(uu - pp, np.inf)
    plt.text(0, -0.5, 'max error = ' + str(np.round(error,6)), ha='center',
             fontsize=16)
    plt.show()
