"""
Eigenvalues of harmonic oscillator

@author: Khiem Nguyen
"""

import numpy as np
from scipy.linalg import toeplitz

L = 8       # domain is [-L, L], periodic
Nmin = 6; Nmax = 36; Nstep = 6
for N in range(Nmin, Nmax+1, Nstep):
    h = 2*np.pi/N
    x = np.linspace(h, 2*np.pi, N)
    x = L/np.pi * (x - np.pi)
    v = np.arange(1, N)
    col = np.append(np.array([-np.pi**2/(3*h*h) - 1/6]), 
                    -0.5*(-1)**v / np.sin(h * v/2)**2 )
    D2 = (np.pi/L)**2 * toeplitz(col)   # 2nd-order differentiation
    
    w = np.linalg.eigvals(-D2 + np.diag(x**2))
    w = np.sort(w)
    
    print('N =', N, ':', '\n--> w =', w[0:3])
    
    
    
