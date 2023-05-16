
import numpy as np
from scipy.sparse import diags
import scipy

def norm2(v):
    return np.sqrt(np.sum(v**2)/len(v))

def steepestdescent(A, b, precision=1e-10):
    """Source: https://en.wikipedia.org/wiki/Gradient_descent#Solution_of_a_linear_system
    Returns solution and normalized residuals"""
    residuals = [1]
    bnorm = norm2(b)

    residual = b
    xi = np.zeros(A.shape[0])
    while norm2(residual) > precision:
        
        p = residual
        a = (residual * p)/(p @ A @ p.T)
        xi = xi + a*p
        residual = b - A @ xi
        residuals.append(norm2(residual)/bnorm)
        
    return xi, residuals

def conjugategradient(A, b, precision=1e-10):
    """Source: https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
    Returns solution and normalized residuals"""
    residuals = [1]
    bnorm = norm2(b)

    residual = b
    xi = np.zeros_like(b)
    pdirection = residual
    while norm2(residual) > precision:
        alpha = residual@residual/(pdirection@A@pdirection)
        xi = xi + alpha*pdirection
        residualk = residual - alpha*A@pdirection
        beta = residualk@residualk/(residual@residual)
        pdirection = residualk + beta*pdirection
        residual = residualk
        residuals.append(np.sqrt(np.sum(residual**2)/len(residual))/bnorm)

    return xi, residuals

def conjugategradient_precon(A, b, M, precision=1e-10):
    """Source: https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
    Returns solution and normalized residuals
    NOT WORKING"""
    residuals = []
    bnorm = norm2(b)

    residual = b
    xi = np.zeros_like(b)
    precon = np.linalg.inv(M)@residual
    pdirection = residual
    while norm2(residual) > precision:
        
        alpha = residual@residual/(pdirection@A@pdirection)
        xi = xi + alpha*pdirection
        residualk = residual - alpha*A@pdirection
        preconk = np.linalg.inv(M)@residualk
        beta = residualk@preconk/(residual@precon)
        pdirection = preconk + beta*pdirection
        residual = residualk
        precon = preconk
        residuals.append(np.sqrt(np.sum(residual**2)/len(residual))/bnorm)

    return xi, residuals

def solve_tridiagonal(n, A, B, C, D):
    """Implements the Thomas algorithm for triadiagonal matrices in o(n) time
    A is lower diagonal of coefficient matrix
    B is main  diagonal of coefficient matrix
    C is upper diagonal of coefficient matrix
    D is the known vector"""

    G = np.empty_like(C)
    R = np.empty_like(D)
    X = np.empty_like(D)

    G[0] = C[0]/B[0]
    R[0] = D[0]/B[0]

    for i in range(1,n-1):
        G[i] = C[i]/(B[i] - A[i-1]*G[i-1])
        R[i] = (D[i] - A[i-1]*R[i-1]) / (B[i] - A[i-1]*G[i-1])
    R[i+1] = (D[i+1] - A[i]*R[i]) / (B[i+1] - A[i]*G[i])
    X[i+1] = R[i+1]
    for i in range(n-2, -1, -1):
        X[i] = R[i] - G[i]*X[i+1]

    return X

def main():
    n = 5

    A = np.linspace(1+5, n-1+5, n-1)
    B = np.linspace(1, n, n)
    C = np.linspace(1, n-1, n-1)
    
    D = 5*np.ones(n)

    X = solve_tridiagonal(n, A, B, C, D)
    print(X)


if __name__ == '__main__':
    main()
