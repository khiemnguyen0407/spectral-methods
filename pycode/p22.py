"""
5th eigenvector of Airy equation u_xx = lambda*x*u

@author: Khiem Nguyen
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import special
from cheb import cheb

plt.close('all')

plt.figure(figsize=(7, 7), dpi=80)
iplot = 1
Nvec = np.arange(12, 49, 12)
for j in range(4):
    N = Nvec[j]
    D, x = cheb(N)
    D2 = np.dot(D, D)
    D2 = D2[1:N, 1:N]
    lam, V = sp.linalg.eig(D2, np.diag(x[1:N]))
    lam = np.real(lam)
    idx_positive = np.argwhere(lam > 0).ravel()
    V = V[:, idx_positive]
    lam = lam[idx_positive]
    
    idx_sort = np.argsort(lam)
    ii = idx_sort[4]
    lam = lam[ii]
    v = np.append(0, np.append(V[:,ii], 0))
    Ai, Aip, Bi, Bip = special.airy(0)
    v = v / v[int(N//2)] * Ai
    xx = np.linspace(-1, 1, 101)
    vv = np.polyval(np.polyfit(x, v, N), xx)
    plt.subplot(2, 2, j+1);
    plt.plot(xx, vv); plt.grid(True)
    plt.title(r'$N = %d,   \mathrm{eig} = %15.10f$' %(N, lam))

plt.show()


    
    
    
    
    
