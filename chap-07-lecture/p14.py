"""
Solve nonlinear BPV u_xx = exp(u)
                    u(_1) = u(1) = 0

@author: Khiem Nguyen
"""

import numpy as np
import matplotlib.pyplot as plt
from cheb import cheb

N = 16
D, x = cheb(N)
D2 = np.dot(D, D)
D2 = D2[1:N, 1:N]
u = np.zeros(N-1)
change = 1; it = 0
while change > 1e-15:
    unew = np.linalg.solve(D2, np.exp(u))
    change = np.linalg.norm(unew - u, np.inf)
    u = unew; it = it + 1
u = np.append(0, np.append(u, 0))

# Plot the solution
xx = np.linspace(-1, 1, 201)
uu = np.polyval(np.polyfit(x, u, N), xx)
plt.plot(x, u, 'o', xx, uu, 'k-', markersize=4, linewidth=1.2); plt.grid(True)
plt.title('Number of steps = ' + str(it))

