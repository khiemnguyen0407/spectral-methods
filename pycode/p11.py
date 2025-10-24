"""
Chebyshev differentiation of a smooth function

@author: Khiem Nguyen
"""

import numpy as np
import matplotlib.pyplot as plt
from cheb import cheb

xx = np.arange(-1.0, 1.01, 0.01)
uu = np.exp(xx) * np.sin(5*xx)
for N in [10, 20]:
    D, x = cheb(N)
    u = np.exp(x) * np.sin(5*x)
    plt.figure(figsize=(10,8))
    plt.subplot(position=[0.15, 0.66 - 0.4*(N==20), 0.31, 0.28])
    plt.plot(x, u, 'bo', xx, uu, 'k-', markersize=4, linewidth=1); plt.grid(True)
    plt.title(r"$u(x), N = " + str(N) + "$")
    
    uprime = np.exp(x) * (np.sin(5*x) + 5*np.cos(5*x))
    error = np.dot(D, u) - uprime
    plt.subplot(position=[0.55, 0.66 -0.4*(N==20), 0.31, 0.28])
    plt.plot(x, error, 'k-o', linewidth=1, markersize=4); plt.grid(True)
    plt.title(r"$\mathbf{D}_{N} \mathbf{u} - u^{\prime}(\mathbf{x}), N = " 
              + str(N) + "$");

