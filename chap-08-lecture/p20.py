"""
Solve 2nd-order wave equation in 2D via FFT

@author: Khiem Nguyen
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import interp2d
from numpy.fft import fft, ifft


N = 24
x = np.cos(np.pi * np.linspace(0, 1, N+1))
y = x
dt = 6.0/ N**2
xx, yy = np.meshgrid(x, x)

num_snapshots = 4;

plotgap = int((1/(num_snapshots - 1) / dt))
dt = (1/(num_snapshots - 1))/plotgap
vv = np.exp(-40*( (xx - 0.4)**2 + yy**2))
vvold = vv

plt.close('all')

plt.figure(figsize= (10, 10), dpi=80)
# Time-stepping by leap-frog formula
for n in range((num_snapshots-1) * plotgap):
    # t = n* dt
    if n % plotgap == 0 or n == (num_snapshots-1)*plotgap - 1:
        ax = plt.subplot(2,2, (n+1)//plotgap + 1, projection = '3d')
        v_interp = interp2d(xx, yy, vv, kind = 'quintic')
        xxx = np.linspace(-1, 1, 41)
        vvv = v_interp(xxx, xxx)
        xxx, yyy = np.meshgrid(xxx, xxx)
        ax.plot_wireframe(xxx, yyy, vvv, color=[0, 0, 0])
        ax.set_zlim(-0.5, 1)
        
        
    # Compute solution by leap-forg stepping
    uxx = np.zeros((N+1, N+1))
    uyy = np.zeros((N+1, N+1))
    idx = np.arange(1, N)
    k = np.fft.fftfreq(2*N, 1/(2*N)); k[N] = 0
    k2 = np.fft.fftfreq(2*N, 1/(2*N)) ** 2
    jk = 1j*k
    for i in range(1, N):
        v = vv[i, :]
        V = np.append(v, np.flip(v[idx]))
        U = np.real(fft(V))
        
        W1 = np.real(ifft(jk * U))      # 1st-derivative
        W2 = np.real(ifft(-k2 * U))     # 2nd-derivative
        uxx[i, idx] = W2[idx] / (1 - x[idx]**2) - x[idx] * W1[idx]/((1 -x[idx]**2)**1.5)
    
    for j in range(1, N):
        v = vv[:, j]
        V = np.append(v, np.flip(v[idx]))
        U = np.real(fft(V))
        W1 = np.real(ifft(jk * U))
        W2 = np.real(ifft(-k2 * U))
        uyy[idx, j] = W2[idx] / (1 - y[idx]**2) - y[idx] * W1[idx]/((1 - y[idx]**2)**1.5)
    
    vvnew = 2*vv - vvold + dt**2 * (uxx + uyy)
    vvold = vv
    vv = vvnew
    
v_val = vv[N//2, N//2]
plt.show()
