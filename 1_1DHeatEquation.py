import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg

def heat1d_fd(L: float, kappa: float, N: int, boundary: str='d', T0: float=None, TL: float=None, slope: float=None):
    """
    Sets up and solves the steady one-dimensional heat conduction equation through the use of a spare matrix with a first order central differencing scheme\n
    One dimensional heat equation: -kappa*(dT/dx)^2=q; => -(dT)^2=q(dx)^2/kappa; => TL-2T+TR=q(dx)^2/kappa; TL is node to the left and TR is node to the right\n
    The matrix equation we are solving is AT=q from \n
        \twhere A is a square NxN matrix of the coefficient matrix defined by the equation and discretization we are using\n
        \twhere T is a column Nx1 matrix of the unknowns of the points in order from left to right\n
        \twhere q is a column Nx1 matrix of the heat source terms\n
    Variable Defintions
    T0 is the"""

    dx = L/N # spacing between nodes
    x = np.linspace(0, L, N+1) # x position of all nodes

    data = np.ones((3,N+1)) # diagonals for making a sparse matrix
    data[0,:] *= -1 # the coefficient of the node to the left of the current node
    data[1,:] *= 2  # the coefficient of the current node
    data[2,:] *= -1 # the coefficient of the node to the right of the current node
    diags = np.array([-1, 0, 1]) # specifies what diagonals have non zero numbers, 0 is the primary diagonal, positive is to the right, list doesnt have to be d
    
    A = sparse.spdiags(data, diags, N+1, N+1, "csr") # making the A matrix, csr means compressed sparse row
    q = np.cos(np.pi*x/L)*dx**2/kappa # right-hand side of the equation all of the heat source terms

    '''Establishing boundary conditions'''
    # Only done for end points of a domain (hance the name) because the relationship between the points inside the domain is governed by the heat equation

    match boundary:
        case 'd': # Dirichlet boundary condition modification, dirichlet means it is just a set value
            A[0, 0] = 1 # mod to the coefficient matrix because the discretization does not apply because we set the value ie no contribution from other point
            A[0, 1] = 0 # mod to the coefficient matrix because the discretization does not apply because we set the value ie no contribution from other point
            q[0] = T0  # mod to the known matrix on the left most point

            A[-1, -1] = 1 # mod to the unknown matrix because the discretization does not apply because we set the value ie no contribution from other point
            A[-1, -2] = 0 # mod to the unknown matrix because the discretization does not apply because we set the value ie no contribution from other point
            q[-1] = TL   # mod to the known matrix on the right most point
        
        case 'n': # Nuemann boundary condition, Nuemann means a slope is set
            # Using a two point backward difference (T-TL)/dx = slope; => T-TL = slope*dx
            # A[N, N-1] = -1  # mod to coeff matrix 
            # A[N, N] = 1     # mod to coeff matrix
            # q[N] = slope*dx # Neumann BC on right
            # # Applying dirichlet on the left
            # A[0, 0] = 1
            # A[0, 1] = 0
            # q[0] = T0

            # Using a three point backward difference (3T-4TL+1TLL)/2/dx = slope; => 3T-4TL+1TLL = 2*slope*dx
            A[N, N-2] = 1     # mod to coeff matrix
            A[N, N-1] = -4    # mod to coeff matrix
            A[N, N] = 3       # mod to coeff matrix
            q[N] = 2*slope*dx # Neumann BC on right
            # Applying dirichlet on the left
            A[0, 0] = 1
            A[0, 1] = 0
            q[0] = T0  

        case 'r': # Robin boundary condition, Robin means slope and value set, the temperature and slope can be weighted
            T0 = 1
            A[N, N-2] = 1               # mod to coeff matrix
            A[N, N-1] = -4              # mod to coeff matrix
            A[N, N] = 3 + 2*slope*dx*T0 # mod to coeff matrix
            q[N] = 2*slope*dx           # Robin BC on right

    T = linalg.spsolve(A,q) # solving matrix equation
    return x, T

def plotsol(x, T, L, kappa, T0, TL, slope):
    """Plotting both the finite difference and the analytical solution"""
    f = plt.figure(figsize=(8,3))

    plt.plot(x, T, 'o-', linewidth=2, color='blue', label="Finite Difference Solution")

    xa, Ta = heat1d_analytic(L, kappa, T0, TL, slope)
    plt.plot(xa, Ta, linewidth=2, color='red', label="Analytic Solution")
    
    plt.xlabel(r'Position, $x$', fontsize =16)
    plt.ylabel(r'Temperature, $T$', fontsize=16)
    plt.grid()
    plt.tick_params(axis='both', labelsize =12)
    f.tight_layout()
    plt.legend()
    plt.show()
    # plt.close(f)

def heat1d_analytic(L, kappa, T0, TL, boundary: str='d', slope=None):
    """Returns the points and values for the analytic solution which is obtained after integrating the 1D heat equation twice"""
    x = np.linspace(0, 1, 1000) # number of x points at which to solve the analytical solution does not have to be the same as the finite difference amount

    match boundary:
        case 'd': # Dirichlet boundary conditions at both ends
            d = T0-L**2/kappa/np.pi**2 # helper variable found by applying dirichlet boundary conditions
            c = (TL-d+L**2/kappa/np.pi**2)/L # helper variable found by applying dirichlet boundary conditions
        
        case 'n': # Nuemann boundary condition at one end, dirichlet must be at the other end
            c = slope + L/np.pi/kappa*np.sin(np.pi) # Nuemann on the right
            d = T0-L**2/kappa/np.pi**2 # Dirichlet on the left

            # c = slope # Nuemann on the left
            # d = TL-c*L-L**2/kappa/np.pi**2 # Dirichlet on the right
        
        case 'r':
            c = slope + L/np.pi/kappa*np.sin(np.pi) # Nuemann on the right
            d = TL-c*L-L**2/kappa/np.pi**2 # Dirichlet on the right

            # c = slope # Nuemann on the left
            # d = T0-L**2/kappa/np.pi**2 # Dirichlet on the left

            
    T = L**2/kappa/np.pi**2*np.cos(np.pi*x/L) + c*x + d # Temperature values

    return x, T
    
def conv_study(L, kappa, N, T0: float=None, TL: float=None, slope: float=None):
    """Performs a convergence study to see the effects of the amount of finite difference points on the final answer in comparison to the analytical answer"""
    error = np.zeros(len(N))
    dx = L/N
    xa, Ta = heat1d_analytic(L, kappa, T0=T0, TL=TL, slope=slope)

    for i, Ni in enumerate(N):
        x, T = heat1d_fd(L, kappa, Ni, T0=T0, TL=TL, slope=slope, boundary='n')
        error[i] = abs(T[-1]-Ta[-1])

    roc = -np.log(error[-1]/error[-2])/np.log(2) # rate of convergance
    plt.loglog(dx, error, 'o-', linewidth=2, color='blue', label="roc = {}".format(round(roc, 4)))
    plt.xlabel('dx')
    plt.ylabel('Error')
    plt.title('Convergence Study')
    plt.legend()
    plt.show()
    return

def main():
    L = 1
    kappa = 0.1
    N = 8

    T0 = -.2
    TL = 5
    slope = 2

    x, T = heat1d_fd(L, kappa, N, T0=T0, TL=TL, slope=slope, boundary='n')
    plotsol(x, T, L, kappa, T0, TL, slope)

    N = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    conv_study(L, kappa, N, T0, TL, slope) # only works with slope involved and not solely dirichlet boundary conditions

if __name__ == '__main__':
    main()