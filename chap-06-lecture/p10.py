"""
Polynomials and corresponding equipotential curve

@author: Khiem Nguyen
"""

import numpy as np
import matplotlib.pyplot as plt

N = 16
plt.close('all'); plt.figure(figsize=(10, 8))
for i in range(2):
    if i == 0:
        s = 'equispaced points'
        x = -1 + 2*np.linspace(0, 1, N+1)
    elif i == 1:
        s = 'Chebyshev points'
        x = np.cos(np.pi * np.linspace(0, 1, N+1))
    p = np.poly(x)
    # Plot p(x) over [-1, 1]
    xx = np.linspace(-1, 1, 1001)
    pp = np.polyval(p, xx)
    plt.subplot(2, 2, 2*i + 1)
    plt.plot(x, np.zeros(x.shape), 'o', markersize=4)
    plt.plot(xx, pp, 'k-', linewidth=1.2)
    plt.title(s); plt.grid(True)
    
    # Plot equipotential curves
    plt.subplot(2, 2, 2*i + 2)
    plt.plot(np.real(x), np.imag(x), 'o', markersize=4)
    xg = np.linspace(-1.4, 1.4, 201)
    yg = np.linspace(-1.12, 1.12, 201)
    xx, yy = np.meshgrid(xg, yg)
    pp = np.polyval(p, xx + 1j*yy)
    levels = 10.0 ** np.arange(-4, 1)
    plt.contour(xx, yy, np.abs(pp), levels)
    plt.title(s)