# -*- coding: utf-8 -*-
"""
Accuracy of Chebyshev spectral differentiation
Compare the results with p7.py

@author: Khiem Nguyen
"""

import numpy as np
import matplotlib.pyplot as plt
from cheb import cheb

# Compute derivatives for various values of N
Nmax = 50; E = np.zeros((4, Nmax))

for N in range(1,Nmax+1):
    D, x = cheb(N)
    v = np.abs(x) ** 3            # 3rd deriv in BV
    vprime = 3 * x * np.abs(x)
    E[0, N-1] = np.linalg.norm(np.dot(D, v) - vprime, np.inf)
    
    v = np.exp(-x**(-2))
    vprime = 2 * v / (x ** 3)   # C-infinity
    E[1, N-1] = np.linalg.norm(np.dot(D, v) - vprime, np.inf)
    
    v = 1/(1 + x**2)            # analytic in [-1, 1]
    vprime = -2 * x * (v ** 2)
    E[2, N-1] = np.linalg.norm(np.dot(D, v) - vprime, np.inf)
    
    v = x**10;                  # polynomial
    vprime = 10 * x**9
    E[3, N-1] = np.linalg.norm(np.dot(D, v) - vprime, np.inf)
    
# Plot results
titles = [r"$x^{3}$", r"$\exp(-x^{-2})$", r"$1/(1+x^2)$", r"$x^{10}$"]
plt.figure(figsize=(7,5), dpi=80)
for iplot in range(4):
    plt.subplot(position=[0.05 + 0.45*(iplot % 2), 0.95 - 0.45*(iplot//2), 0.32, 0.28])
    plt.semilogy(np.linspace(1,Nmax,Nmax), E[iplot,:], 'k-o', markersize=4, linewidth=1.2)
    plt.axis([0, Nmax, 1e-16, 1e3]); plt.grid(True)
    plt.xlabel(r"$N$")
    plt.ylabel(r"$\|\| \mathbf{D}_{N}\mathbf{u} - u(\mathbf{x}) \|\|$")
    plt.title(titles[iplot])    
    