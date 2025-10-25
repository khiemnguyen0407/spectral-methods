"""
Helmholtz equation u_xx + u_yy + k^2 u = f

@author: Khiem Nguyen
"""

import numpy as np
import matplotlib.pyplot as plt
from chebPy import cheb
from scipy.interpolate import interp2d
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')
# Set up spectral grid and tensor product Helmholtz operator:
N = 24;
D, x = cheb(N); y = x
xx, yy = np.meshgrid(x[1:N], y[1:N])
xx = xx.ravel()
yy = yy.ravel()
f = np.exp(-10 * ((yy - 1)**2 + (xx - 0.5)**2) )
D2 = np.dot(D, D)
D2 = D2[1:N, 1:N]
I = np.eye(N-1)
k = 9
L = np.kron(I, D2) + np.kron(D2, I) + k**2 * np.eye((N-1)**2)

# Solve for solution u, reshape to 2D grid, and plot
u = np.linalg.solve(L, f)
uu = np.zeros((N+1, N+1))
uu[1:N, 1:N] = np.reshape(u, newshape=(N-1, N-1))
u_val = uu[N//2, N//2]

xx, yy = np.meshgrid(x, y)
u_interp = interp2d(x, y, uu, kind='cubic')

xxx = np.linspace(-1, 1, 41)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(xxx, xxx)
ax.plot_surface(X, Y, u_interp(xxx, xxx), rstride=1, cstride=1)
ax.set_zlim(-0.03, 0.03)
plt.title(r'$u(0, 0) = %13.11f$' %u_val)

plt.figure(figsize=(6, 6))
plt.contour(X, Y, u_interp(xxx, xxx), alpha=0.9, cmap=plt.cm.hot)
