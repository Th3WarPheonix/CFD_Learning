
import numpy as np
import matplotlib.pyplot as plt

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

def runge_kutta4(fun, dt:float, time, y0:float):
    """ODE integration using 4th-order Runge-Kutta

    Parameters
    ----------
    fun  : ODE to be integrated, if function is vector function y0 and function output must be a single row i.e. [1, 2, 3, ...], not [[1], [2], [3], ...]
    dt   : time step
    time : array of evenly spaced time intervals
    y0   : initial condition
    """

    yfinal = np.empty((len(y0), len(time)))
    yfinal[:, 0] = y0
    for i, t in enumerate(time):
        f1 = fun(t, y0)
        f2 = fun(t + dt / 2, y0 + (dt / 2) * f1)
        f3 = fun(t + dt / 2, y0 + (dt / 2) * f2)
        f4 = fun(t + dt, y0 + dt * f3)
        yfinal[:, i] = y0 + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
        y0 = yfinal[:, i]
    return yfinal

def lorenz(t, y):
    """
        This function defines the dynamical equations
        that represent the Lorenz system. 
        
        Normally we would need to pass the values of
        sigma, beta, and rho, but we have already defined them
        globally above.
    """
    sigma = 10
    beta = 8 / 3
    rho = 28
    # y is a three dimensional state-vector
    dy = [
        sigma * (y[1] - y[0]), 
        y[0] * (rho - y[2]) - y[1],
        y[0] * y[1] - beta * y[2]]
    return np.array(dy)

def main():
    # Initial condition
    y0 = [-8, 8, 25]

    # Compute trajectory 
    dt = 0.01
    T = 10
    num_time_pts = int(T / dt)
    t = np.linspace(0, T, num_time_pts)

    yin = y0
    yout = runge_kutta4(lorenz, dt, t, yin)
    ax = plt.figure().add_subplot(projection='3d')  # make a 3D plot
    ax.plot(yout[0, :], yout[1, :], yout[2, :], 'b')
    plt.show()
if __name__ == '__main__':
    main()
