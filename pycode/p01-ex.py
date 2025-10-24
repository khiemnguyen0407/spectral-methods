"""
Convergence of second-order and fourth-order differences
"""

# %%
from scipy.sparse import coo_matrix
import numpy as np
import matplotlib.pyplot as plt

# %%
# For various N, set up grid in [-pi, pi] and function u(x)
Nvec = 2**np.arange(3, 13)  # 8, 16, ..., 4096

error4_list = []
error2_list = []

fig, ax = plt.subplots()

for N in Nvec:
    h = 2*np.pi / N
    x = -np.pi + np.arange(1, N+1) * h
    u = np.exp(np.sin(x))
    uprime = np.cos(x) * u

    # Construct sparse 4th-order differentiation matrix (periodic)
    e = np.ones(N)
    e1 = np.arange(0, N)
    e2 = np.append(e1[1:], e1[0])          # +1 shift (periodic)
    e3 = np.append(e1[2:], e1[0:2])        # +2 shift (periodic)

    D4 = coo_matrix((2*e/3, (e1, e2)), shape=(N, N)) \
       - coo_matrix((e/12, (e1, e3)), shape=(N, N))
    D4 = (D4 - D4.T) / h

    D2 = coo_matrix((0.5*e, (e1, e2)), shape=(N, N))
    D2 = (D2 - D2.T) / h

    # Errors
    error4 = np.linalg.norm(D4.dot(u) - uprime, ord=np.inf)
    error2 = np.linalg.norm(D2.dot(u) - uprime, ord=np.inf)
    error4_list.append(error4)
    error2_list.append(error2)

# Convert to arrays for plotting
error4_arr = np.array(error4_list)
error2_arr = np.array(error2_list)

# --- Plot everything on the SAME axes ---
ax.set_xscale('log')
ax.set_yscale('log')

# Numerical errors
ax.plot(Nvec, error4_arr, 'ro-', markersize=4, label='4th-order FD (error)')
ax.plot(Nvec, error2_arr, 'bs-', markersize=4, label='2nd-order FD (error)')

# Reference slopes (scaled to match the first point for readability)
c4 = error4_arr[0] * (Nvec[0] ** 4)       # so that c4 * N^{-4} passes through the first red point
c2 = error2_arr[0] * (Nvec[0] ** 2)       # so that c2 * N^{-2} passes through the first blue point
ax.plot(Nvec, c4 * Nvec**(-4.0), 'r--', label=r'$N^{-4}$ (ref)')
ax.plot(Nvec, c2 * Nvec**(-2.0), 'b--', label=r'$N^{-2}$ (ref)')

# Labels, grid, legend
ax.set_xlabel(r'$N$', fontsize=12)
ax.set_ylabel('max error (∞-norm)', fontsize=12)
ax.set_title('Convergence of finite-difference derivatives')
ax.grid(True, which='both', ls=':', alpha=0.6)
ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.show()