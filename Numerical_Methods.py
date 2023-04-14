
import numpy as np

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
    Returns solution and normalized residuals"""
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