"""
Repetition of p04.py via FFT 
    adapted from Trefethen, Spectral Methods in MATLAB
    
@author: Khiem Nguyen
"""

import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

# Differentiation of hat function
N = 24; h = 2*np.pi/N; x = np.linspace(h, 2*np.pi, N)
v = np.maximum(0, 1 - 0.5*np.abs(x - np.pi))
k = np.append(np.arange(0, N/2), np.append(0, np.arange(-N/2+1, 0)))
v_derv = np.real(ifft( 1j * k * fft(v) ))

plt.figure(figsize=(8, 6), dpi=80)
plt.subplot(3, 2, 1)
plt.plot(x, v, 'k-o', markersize=3)
plt.axis([0, 2*np.pi, -0.5, 1.5]); plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(x, v_derv, 'k-o', markersize=3)
plt.axis([0, 2*np.pi, -1, 1]); plt.grid(True)

# Differentiation of exp(sin(x))
v = np.exp(np.sin(x)); vprime = np.cos(x) * v
v_derv = np.real(ifft( 1j * k * fft(v) ))
plt.subplot(3, 2, 3)
plt.plot(x, v, 'k-o', markersize=3)
plt.axis([0, 2*np.pi, 0, 3]); plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(x, v_derv, 'k-o', markersize=3)
plt.axis([0, 2*np.pi, -2, 2]); plt.grid(True)
error = np.linalg.norm(v_derv - vprime, np.inf)
plt.text(2.2, 1.4, 'max error = ' + str(np.round(error, 16)))
plt.show()
