
import numpy as np

def norm2(v):
    return np.sqrt(np.sum(v**2)/len(v))

def steepestdescent(A, b, precision=1e-10):
    """Returns solution and residuals normalized to initial 2 norm of b
    
    Parameters
    ----------
    A : matrix
    b : vector
    
    Source: https://en.wikipedia.org/wiki/Gradient_descent#Solution_of_a_linear_system"""
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
    """Returns solution and residuals normalized to initial 2 norm of b

    Parameters
    ----------
    A : matrix
    b : vector

    Source: https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
    """
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

def solve_tridiagonal(n:int, A, B, C, D):
    """Solves tridiagonal matrix using Thomas algorithm in O(n) time
    
    Parameters
    ----------
    n : size of square array
    A : lower diagonal of coefficient matrix
    B : main  diagonal of coefficient matrix
    C : upper diagonal of coefficient matrix
    D : the known vector"""

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