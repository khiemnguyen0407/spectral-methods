"""
Convergence of second-order and fourth-order differences
"""
# %%
from scipy.sparse import coo_matrix
import numpy as np
import matplotlib.pyplot as plt

# %%
# For various N, set up grid in [-pi, pi] and function u(x)
Nvec = 2**np.arange(3, 13)
fig, ax = plt.subplots()
for N in Nvec:
    h = 2*np.pi / N
    x = -np.pi + np.arange(1, N+1) * h
    u = np.exp(np.sin(x))
    uprime = np.cos(x) * u
    
    # Construct sparse 4th-order differentiation matrix
    e = np.ones(N)
    e1 = np.arange(0, N)
    e2 = np.append(e1[1:], e1[0])
    e3 = np.append(e1[2:], e1[0:2])
    
    D4 = coo_matrix((2*e/3, (e1, e2)), shape=(N,N)) \
        - coo_matrix((e/12, (e1, e3)), shape=(N,N))
    D4 = (D4 - D4.T)/h
    
    D2 = coo_matrix((0.5*e, (e1, e2)), shape=(N,N))
    D2 = (D2 - D2.T)/h
    
    # Plot max(abs(D4*u - uprime))
    error4 = np.linalg.norm(D4.dot(u) - uprime, ord = np.inf)
    error2 = np.linalg.norm(D2.dot(u) - uprime, ord = np.inf)
    ax.loglog(N, error4, 'ro', markersize=4)
    ax.loglog(N, error2, 'bs', markersize=4)

# %%
# ax.set_xlabel(r'$N$', fontsize=18)
# ax.set_ylabel('error', fontsize=18)
# ax.set_title('Convergence of 4th-order finite differences')
plt.loglog(Nvec, Nvec ** (-4.0), 'r--')
# plt.plot(Nvec, Nvec ** (-2.0), 'b--')
# ax.text(105, 5e-8, r'$N^{-4}$', fontsize=18)
plt.show()

# %%
# plt.show()

# %%
