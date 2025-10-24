# -*- coding: utf-8 -*-
"""
Solve eigenvalue BVP u_xx = lambda * u
                     u(-1) = u(1) = 0

@author: Khiem Nguyen
"""

import numpy as np
import matplotlib.pyplot as plt
from cheb import cheb

N = 36
D, x = cheb(N)
D2 = np.dot(D, D)
D2 = D2[1:N, 1:N]
lam, V = np.linalg.eig(D2)
ii = np.argsort(-lam)
lam = lam[ii]
V = V[:, ii]

xx = np.linspace(-1, 1, 201)
plt.figure(figsize = (9, 9), dpi=80)
for j in range(4, 30, 5):
    u = np.append(0, np.append(V[:, j], 0))
    uu = np.polyval(np.polyfit(x, u, N), xx)
    plt.subplot(position=[0.1, 0.98 - (j-4)/5*0.16, 0.9, 0.13])
    plt.plot(x, u, 'o', xx, uu, 'k-', markersize=4, linewidth=1.2)

    s = r"eig %d = %17.13f $\times 4/\pi^2$" %(j+1, lam[j]*4/np.pi**2)
    s = s + "\t %4.1f ppw" %(4*N/(np.pi * (j+1)))
    plt.title(s)
    plt.box(on=True); 
    plt.xticks([])

