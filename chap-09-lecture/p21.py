"""
eigenvalues of Mathieu operator -u_xx + 2 q cos(2x) u

@author: Khiem Nguyen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

N = 42
h = 2*np.pi/N
x = np.linspace(h, 2*np.pi, N)

v = np.arange(1, N)
col = np.append(-np.pi**2/(3*h**2) - 1/6, -0.5*(-1)**v / np.sin(h * v / 2)**2)
D2 = toeplitz(col)

qq = np.arange(0, 15.2, 0.2)
data = np.zeros((len(qq), 11))
for j in range(len(qq)):
    lam = np.linalg.eigvals(-D2 + 2 * qq[j] * np.diag(np.cos(2*x)))
    lam = np.sort(lam)
    data[j, :] = lam[0:11]

plt.subplot(1,2,1)    
plt.plot(qq, data, '-')