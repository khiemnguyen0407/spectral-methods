# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 12:45:06 2020

@author: nguyenl
"""

import numpy as np
import matplotlib.pyplot as plt
from cheb import chebfft
from mpl_toolkits.mplot3d import axes3d
from matplotlib.collections import LineCollection#


plt.close('all') # close all existed figures

# Time-stepping by leap forg formula
N = 140
x = np.cos(np.pi * np.linspace(0, 1, N+1))
dt = 8.0/N**2
v = np.exp(-200*x**2)
vold = np.exp(-200 * (x - dt)**2)
tmax = 4
tplot = 0.05
plotgap = round(tplot/dt)
dt = tplot / plotgap  # slightly modify dt so that it fits the plotgap
nplots = round(tmax / tplot)

sol_data = []
sol_data.append(list(zip(x, v)))

tdata = np.zeros(nplots+1)

for i in range(nplots):
    for n in range(plotgap):
        w = chebfft(chebfft(v)); w[0] = 0; w[-1] = 0
        vnew = 2*v - vold + dt**2 * w; 
        vold = v; v = vnew
    sol_data.append(list(zip(x, v)))
    tdata[i+1] = dt * (i+1) * plotgap

# Plot the solution
plt.figure(figsize=(7,5), dpi=80)
ax = plt.gca(projection = '3d')
sol_line = LineCollection(sol_data)
sol_line.set_alpha(0.7)
ax.add_collection3d(sol_line, zs=tdata, zdir='y')
ax.set_xlim(-1, 1)
ax.set_ylim(0, tmax)
ax.set_zlim(-2, 2)
plt.show()