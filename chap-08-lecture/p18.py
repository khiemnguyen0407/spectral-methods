# -*- coding: utf-8 -*-
"""
Chebyshev differentiation via FFT

@author: Khiem
"""

import numpy as np
import matplotlib.pyplot as plt
from cheb import chebfft

plt.close('all')
plt.figure(figsize=(7, 5))


xx = np.linspace(-1, 1, 201)
ff = np.exp(xx) * np.sin(5*xx)
for N in [18, 20]:
    x = np.cos(np.pi * np.arange(0,N+1)/N)
    f = np.exp(x) * np.sin(5*x)
    #plt.subplot(2,2,2*(N==20)+1)
    plt.subplot(position=[0.1, 0.6 - 0.5*(N==20), 0.35, 0.3])
    plt.plot(x, f, 'bo', xx, ff, 'k-', markersize=4, linewidth=1.2)
    plt.grid(True)
    plt.title(r'$f(x), N = ' + str(N) + '$', fontsize=12)
    
    error = chebfft(f) - np.exp(x) * (np.sin(5*x) + 5*np.cos(5*x))
    #plt.subplot(2,2,2*(N==20)+2)
    plt.subplot(position=[0.55, 0.6 - 0.5*(N==20), 0.35, 0.3])
    plt.plot(x, error, 'k-o', markersize=4, linewidth=1.2)
    plt.grid(True)
    plt.title(r"error in $f^{\prime}(x)$, $N =" + str(N) + "$", fontsize=12)
    

