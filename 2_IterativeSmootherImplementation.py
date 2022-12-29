
import numpy as np
import matplotlib.pyplot as plt

# Jacobi iterations: modify u in place
def Jacobi(N, dx, u, f, niter, omega, x):
    rv = np.zeros(niter)
    r = np.zeros(len(u))
    drdu = np.zeros(len(u))
    for ii in range(niter):
        uL = u[0]
        for i in range(1, N-1):
            ut = u[i]
            
            r[i] = (f[i]+(uL -2*u[i] +u[i+1])/dx**2) -u[i]**2 # Redisual
            drdu[i] = -2*(u[i] + 1/dx**2) # Derivative of residual wrt ui
            u[i] = u[i] + omega*r[i]/-drdu[i] # Updated ui at next time step

            uL = ut

        rv[ii] = np.sqrt(dx)*np.linalg.norm(r)

    plt.semilogy(range(niter), rv, label='Residual')
    plt.title('Iterative Convergence')
    plt.xlabel('Iterations')
    plt.ylabel(r'$|r_h|_{L_2}$')
    plt.legend()
    plt.savefig('H3Q3b.pdf')
    plt.show()
    #rv[niter] = np.sqrt(dxh)*np.linalg.norm(rh)  # L2 residual norm

def main():
    niter = 1000
    omega = 1
    N = 16
    L = 1
    dx = L/N
    x = np.linspace(0, 1, N)
    u = x + 1
    f = np.zeros(len(x))
    Jacobi(N, dx, u, f, niter, omega, x)

if __name__ == '__main__':
    main()