"""
Solve linear boundary value problem u_xx = exp(4x)
                                    u(-1) = u(1) = 0

@author: Khiem Nguyen
"""

import numpy as np
import matplotlib.pyplot as plt
from cheb import cheb

N = 16
D, x = cheb(N)
D2 = np.dot(D, D)
D2 = D2[1:N, 1:N]
f = np.exp(4 * x[1:N])
u = np.linalg.solve(D2, f)
u = np.append(0, np.append(u, 0))

xx = np.linspace(-1, 1, 201)
p = np.polyfit(x, u, N)
uu = np.polyval(p, xx)
plt.plot(x, u, 'k.', xx, uu, '-')
plt.grid(True)
exact = (np.exp(4*xx) - np.sinh(4)*xx - np.cosh(4))/16
plt.title('max error = ' + str(round(np.linalg.norm(uu - exact, np.inf), 14)) )