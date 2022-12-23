import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt

def plotsol(X, Y, T):
    """Plkots the solution on a 2D grid"""
    f = plt.figure(figsize=(8,6))
    cont = plt.contourf(X, Y, T, cmap='bwr')
    cbar = f.colorbar(cont)
    cbar.set_label('Temperature (K)')
    plt.xlabel(r'$x$', fontsize =16)
    plt.ylabel(r'$y$', fontsize =16)
    plt.grid()
    plt.tick_params(axis='both', labelsize =12)
    f.tight_layout()
    plt.show()

def source(x, y, Lx, Ly, kappa):
    """Returns the soruce term for the 2D steady heat equation"""
    return np.sin(np.pi*x/Lx)*(-(np.pi/Lx)**2*y/Ly*(1-y/Ly) - 2/Ly**2)*(-kappa)

def heat2d(Lx, Ly, Nx, Ny, kappa):
    """Setting up and solvin the 2D heat equation with matrices AT=q
    (TL-T+TR)/dx^2 + (TU-T+TD)/dx^2 = -q/kappa"""
    dx = float(Lx)/Nx # x spacing
    dy = float(Ly)/Ny # y spacing
    x = np.linspace(0, Lx, Nx+1) # x nodes
    y = np.linspace(0, Ly, Ny+1) # y nodes
    Y, X = np.meshgrid(y, x) # matrices of all nodes
    N = (Nx+1)*(Ny+1) # total number of unknowns
    A = sparse.csr_matrix((N,N)) # empty sparse matrix
    q = np.zeros(N) # source term vector

    # fill in interior contributions
    for iy in range(1, Ny):
        for ix in range(1, Nx):
            i = iy*(Nx+1)+ix
            x = ix*dx
            y = iy*dy
            iL = i-1
            iR = i+1
            iD = i-(Nx+1)
            iU = i+(Nx+1)
            q[i] = source(x,y,Lx,Ly,kappa)/kappa
            # Applying the second order central difference scheme
            A[i, i] = 2/dx**2 + 2/dy**2
            A[i,iL] = A[i,iR] = -1/dx**2
            A[i,iD] = A[i,iU] = -1/dy**2

    # Dirichlet boundary conditions
    for ix in range(Nx+1):
        i = ix
        A[i, i ] = 1
        q[i ] = 0
        i = Ny*(Nx+1)+ix
        A[i,i] = 1
        q[i] = 0
    for iy in range(Ny+1):
        i = iy*(Nx+1)
        A[i,i] = 1
        q[i ] = 0
        i = iy*(Nx+1)+Nx
        A[i,i] = 1
        q[i] = 0

    # Solve system
    Tv = linalg.spsolve(A,q) # solution at all points
    T = np.reshape(Tv, (Nx+1,Ny+1), order='F') # reshape into matrix for plotting

    return X, Y, T

def main():
    Lx, Ly, Nx, Ny, kappa = 2, 1, 40, 20, 0.5
    X, Y, T = heat2d(Lx, Ly, Nx, Ny, kappa)
    plotsol(X, Y, T)

if __name__ == '__main__':
    main()