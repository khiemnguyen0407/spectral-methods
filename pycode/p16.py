# -*- coding: utf-8 -*-
"""
Poisson eq. on [-1,1]x[-1,1] with u = 0 on boundary

@author: nguyenl
"""

import numpy as np
import matplotlib.pyplot as plt
from chebPy import cheb
from scipy.interpolate import interp2d
# from mpl_toolkits.mplot3d import Axes3D

plt.close('all')

# Set up grids and tensor product Laplacian and solve for

N = 24
D, x = cheb(N)
y = x
xx, yy = np.meshgrid(x[1:N], y[1:N])
# xx = np.reshape(xx,(N-1)**2)
xx = xx.ravel()
yy = yy.ravel()
f = 10 * np.sin(8*xx * (yy - 1))

D2 = np.dot(D, D)
D2 = D2[1:N, 1:N]
IMatrix = np.eye(N-1)
L = np.kron(IMatrix, D2) + np.kron(D2, IMatrix)
fig = plt.figure(figsize=(10,8), dpi=80)
plt.subplot(1,2,1)
plt.spy(L)

u = np.linalg.solve(L, f)
# Reshape long 1D results onto 2D grid
uu = np.zeros((N+1, N+1))
uu[1:N, 1:N] = np.reshape(u, newshape=(N-1,N-1))
value = uu[N//4, N//4]

# Interpolate to finer grid and plot
xxx = np.linspace(-1, 1, 41)
f = interp2d(x, y, uu, kind='cubic')
uuu = f(xxx, xxx)
X, Y = np.meshgrid(xxx, xxx)
# fig = plt.figure(figsize=(8, 8), dpi=80)
ax = fig.add_subplot(1,2,2, projection='3d')
ax.plot_surface(X, Y, uuu, rstride=1, cstride=1, cmap=plt.cm.jet)

plt.title(r"$u(2^{-1/2},2^{-1/2})$="+str(value))