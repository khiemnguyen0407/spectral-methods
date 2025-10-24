import numpy as np
from numpy.fft import fft, ifft

def cheb(N):
    """
    Compute the diffferentian matrix
    
    Arguments: 
        N -- Total number of grid points is N+1
    Return: 
        D -- Differentiation matrix
        x -- Chebyshev grid points
    """

    if N == 0:
        D = 0
        x = 1
        return D, x
    x = np.cos(np.pi * np.linspace(0, 1, N+1))
    c = np.append(2.0, np.append(np.ones(N-1), 2.0))
    c = c * (-1.0) ** np.arange(0, N+1)
    c = c[:, np.newaxis]
    X = np.tile(x[:, np.newaxis], (1, N+1))
    dX = X - X.T
    D = np.dot(c, 1.0/c.T) / (dX + np.eye(N+1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def chebfft(v):
    """
    Compute derivative of v(x) with Chebyshev differentiation via FFT
    
    Arguments:
        v -- values of function v = v(x) at the Chebyshev points
    Return:
        w -- derivative of v = v(x) at the Chebyshev points
    """
    N = len(v) - 1
    if N == 0:
        return 0
    x = np.cos(np.pi * np.linspace(0, 1, N+1))
    ii = np.arange(0, N)
    V = np.append(v, v[N-1:0:-1])
    
    U = np.real(fft(V))
    kk = np.append(ii, np.append(0, np.arange(1-N, 0)))
    W = np.real(ifft(1j*kk*U))
    
    w = np.zeros(N+1)
    w[1:N] = -W[1:N] / np.sqrt(1 - x[1:N] ** 2)
    w[0] = np.sum(ii**2 * U[ii]) / N + 0.5 * N * U[N]
    w[N] = np.sum((-1)**(ii+1) * ii**2 * U[ii])/N \
        + 0.5 * (-1)**(N+1) * N * U[N]
    return w
    
    
    