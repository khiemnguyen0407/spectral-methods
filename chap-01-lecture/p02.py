"""
Convergence of periodic spectral method
"""
# For various N, set up grid in [-pi, pi] and function u(x)

from scipy.linalg import toeplitz
import numpy as np
import matplotlib.pyplot as plt

Nvec = np.arange(2, 101, 2)
error = np.zeros(Nvec.shape)
for i in range(len(Nvec)):
    N = Nvec[i]
    h = 2*np.pi/N
    x = -np.pi + np.arange(1, N+1) * h
    u = np.exp(np.sin(x))
    uprime = np.cos(x) * u
    
    # Construct spectral differentiation matrix
    v = np.arange(1, N)
    col = np.append(np.array([0]), 0.5 * (-1.0)**v / np.tan(0.5 * h * v))
    row = np.zeros(N)
    row[1:] = col[N-1:0:-1]
    D = toeplitz(col, row)
    # Compute max(abs(D*u - uprime))
    error[i] = np.linalg.norm(np.dot(D, u) - uprime, np.inf)
    
plt.loglog(Nvec, error, 'k.', markersize=10)
plt.grid(True)
plt.xlabel(r'$N$', fontsize=18)
plt.ylabel(r'$\|\mathbf{D}_{N} \mathbf{u} - u^{\prime}\|$', fontsize=18)
plt.title('Convergence of spectral differentiation', fontsize=12)
plt.show()