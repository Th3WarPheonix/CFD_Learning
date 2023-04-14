
import numpy as np
import matplotlib.pyplot as plt

# Jacobi iterations: modify u in place
def Jacobi(N, dx, u, f, niter, omega, x):
    rv = np.zeros(niter)
    r = np.zeros(len(u)) # Store residual of all nodes
    drdu = np.zeros(len(u)) # Store derivastive of residual wrt each node of all nodes
    for ii in range(niter):
        uL = u[0]
        for i in range(1, N-1):
            ut = u[i]
            
            r[i] = f[i] + (uL - 2*u[i] + u[i+1])/dx**2 - u[i]**2 # Residual
            drdu[i] = -2*u[i] - 2/dx**2 # Derivative of residual wrt the current node
            u[i] = u[i] - omega*r[i]/drdu[i] # Updates current node for the next time step with Newtons method

            uL = ut

        rv[ii] = np.sqrt(dx)*np.linalg.norm(r)
        # plt.plot(x, u)
        # plt.title('Iteration: {}'.format(ii))
        # plt.show(block=False)
        # plt.draw()
        # time.sleep(.1)
    plt.figure(2)
    plt.semilogy(range(niter), rv, label='Residual')
    plt.title('Iterative Convergence')
    plt.xlabel('Iterations')
    plt.ylabel(r'$|r_h|_{L_2}$')
    plt.legend()
    # rv[niter] = np.sqrt(dxh)*np.linalg.norm(rh)  # L2 residual norm


def main():
    """Implementing the Jacobi iteration method on a 1D domain.
    The residual r = f-Au, from the basic matrix equation Au = f, an extension of this is that r = Ae, where is the error"""
    num_iter = 10000 # Number of iterations
    omega = 1 # Overrelaxation factor; 1 is no overrelaxation
    N = 16 # number of nodes
    L = 1 # Length of domain
    dx = L/N # Distance between nodes
    x = np.linspace(0, L, N) # Node x values
    u = x + 1
    f = np.zeros(len(x))

    plt.figure(1)
    plt.plot(x, u, label='before')

    Jacobi(N, dx, u, f, num_iter, omega, x)

    plt.figure(1)
    plt.plot(x, u, label='after', marker='o')
    plt.plot(x, x**2+1, label='exact')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()