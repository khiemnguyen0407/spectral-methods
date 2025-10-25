"""
Variable coefficient wave equation

@author: Khiem Nguyen
"""

import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.collections import LineCollection

# Set up grid and initial condition
N = 2**7; h = 2*np.pi/N; x = np.linspace(h, 2*np.pi, N)
t = 0.0; dt = h/4.0
c = 0.2 + np.sin(x - 1) ** 2
v = np.exp(-100 * (x - 1)**2)
vold = np.exp(-100 * (x - 0.2*dt -1)**2)

# Time-stepping by leap-frog formula
tmax = 8.0; tplot = 0.2
plotgap = int(round(tplot/dt)); dt = tplot/plotgap
nplots = int(round(tmax/tplot))
tdata = np.zeros(nplots+1)
data = []
data.append(list(zip(x, v)))

# k = np.append(np.arange(0, N/2), np.append(0, np.arange(-N/2+1, 0)))
k = np.fft.fftfreq(N, 1/N); k[N//2] = 0;
jk = 1j*k
for i in range(nplots):
    for n in range(plotgap):
        t = t + dt
        vprime = np.real(ifft(jk * fft(v)))
        vnew = vold - 2 * dt * c * vprime
        vold = v
        v = vnew
    
    data.append(list(zip(x, v)))
    tdata[i+1] = t  # by default tdata[0] = 0
    
fig = plt.figure(figsize=(7,5), dpi = 80)
ax = fig.add_subplot(111, projection='3d')
poly = LineCollection(data)
poly.set_alpha(0.5)
ax.add_collection3d(poly, zs=tdata, zdir='y')
ax.set_xlabel(r'$x$'); ax.set_xlim3d(0, 2*np.pi)
ax.set_ylabel(r'$t$'); ax.set_ylim3d(0, tmax)
ax.set_zlabel(r'$u$'); ax.set_zlim3d(0, 2.5)
ax.view_init(70, -70)
plt.show()
ax.grid(False)
# ax.set_title('variable coefficient wave'
        
