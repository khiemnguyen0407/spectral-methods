"""
Accuracy of periodic spectral differentiation
"""

import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# Compute derivatives for various values of N
Nmax = 50
E = np.zeros((4, int(Nmax/2) - 2))
for N in range(6, Nmax+1, 2):
    h = 2*np.pi/N
    x = np.linspace(h, 2*np.pi, N)
    k = np.fft.fftfreq(N, 1/N); k[int(N/2)] = 0
    jk = 1j*k
    
    M = int(N/2)
    v = np.abs(np.sin(x)) ** 3
    vprime = 3*np.sin(x) * np.cos(x) * np.abs(np.sin(x))
    v_derv = np.real(ifft(jk * fft(v)))     # without np.real, it results in imaginary value 0
    E[0, M-3] = np.linalg.norm(v_derv - vprime, np.inf)
    
    v = np.exp(-np.sin(0.5*x) ** (-2))
    vprime = 0.5 * v * np.sin(x) / np.sin(0.5*x) ** 4
    v_derv = np.real(ifft(jk * fft(v)))     # without np.real, it results in imaginary value 0
    E[1, M-3] = np.linalg.norm(v_derv - vprime, np.inf)
    
    v = 1 / (1 + np.sin(0.5*x) ** 2)
    vprime = -np.sin(0.5*x) * np.cos(0.5*x) * v ** 2
    v_derv = np.real(ifft(jk * fft(v)))     # without np.real, it results in imaginary value 0
    E[2, M-3] = np.linalg.norm(v_derv - vprime, np.inf)
    
    v = np.sin(10*x)
    vprime = 10*np.cos(10*x)
    v_derv = np.real(ifft(jk * fft(v)))     # without np.real, it results in imaginary value 0
    E[3, M-3] = np.linalg.norm(v_derv - vprime, np.inf)
    
# Plot results
titles = [r'$|\sin(x)|^3$', 
          r'$\exp[-\sin^{-2}(x/2)]$', 
          r'$\displaystyle \frac{1}{1 + \sin^2(x/2)}$',
          r'$\sin(10x)$']
plt.figure(figsize=(10,6))
for iplot in range(4):
    plt.subplot(2, 2, iplot+1)
    plt.semilogy(np.arange(6, Nmax+1, 2), E[iplot], '-o', markersize=5)
    plt.axis([0, Nmax, 1e-16, 1e3]); plt.grid(True)
    plt.xticks(np.arange(0, Nmax+1, 10))
    plt.yticks(10**np.linspace(-15, 0, 4))
    plt.title(titles[iplot])
    
plt.show()
    
