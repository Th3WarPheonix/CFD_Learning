
import numpy as np
import numpy.linalg as npla
from scipy import sparse
import matplotlib.pyplot as plt

def Jacobi(UN):
    """Gets the eigenvalues from the Jacobi iteration matrix. Matrix update equation is un1 = -D**-1*(L+U)*un + D**-1*f"""
    data = np.ones((1,UN)) # diagonals for making a sparse matrix
    data[0,:] *= 2
    diags = np.array([0])
    D2 = sparse.spdiags(data, diags, UN, UN, "csr") # all nodes are Unknowns

    data = np.ones((1,UN)) # diagonals for making a sparse matrix
    data[0,:] *= -1
    diags = np.array([-1])
    L2 = sparse.spdiags(data, diags, UN, UN, "csr") # all nodes are Unknowns

    data = np.ones((1,UN)) # diagonals for making a sparse matrix
    data[0,:] *= -1
    diags = np.array([1])
    U2 = sparse.spdiags(data, diags, UN, UN, "csr") # all nodes are Unknowns

    SJ = np.matmul(npla.inv(D2.toarray()), (L2.toarray()+U2.toarray()))

    values, vectors = npla.eig(SJ)
    return values

def Gauss_Seidel(UN):
    """Gets the eigenvalues from the Gauss-Seidel iteration matrix. Matrix update equation is un1 = -(D+L)**-1*U*un + (D+L)**-1*f"""
    data = np.ones((1,UN)) # diagonals for making a sparse matrix
    data[0,:] *= 2
    diags = np.array([0])
    D2 = sparse.spdiags(data, diags, UN, UN, "csr") # all nodes are Unknowns

    data = np.ones((1,UN)) # diagonals for making a sparse matrix
    data[0,:] *= -1
    diags = np.array([-1])
    L2 = sparse.spdiags(data, diags, UN, UN, "csr") # all nodes are Unknowns

    data = np.ones((1,UN)) # diagonals for making a sparse matrix
    data[0,:] *= -1
    diags = np.array([1])
    U2 = sparse.spdiags(data, diags, UN, UN, "csr") # all nodes are Unknowns

    SGS = np.matmul(npla.inv(D2.toarray()+L2.toarray()), (U2.toarray()))

    values, vectors = npla.eig(SGS)
    return values
    
def main():
    """
    Background
    ----------
    This is an exercise to show the rate of convergence of these iterative smoother methods by analysing the eigenvalues of the iteration matrices. In actual use the matrices are never formed because that defeats the purpose of using smoother, which is saving time and space, over driect solvers using matrix operations. We are trying to solve the same matrix equation Au = f. A can be decomposed into A = D + L + U where D is only the main diagonal, L is everything below the main diagonal, lower triangular portion, U is everything above the main diagonal, upper triangular portion. D can be thought of the current node in the iteration, L contains all previously iterated upon nodes, U is all nodes not already iterated upon. un1 is the state of the node at the next time step and un is the state of the node at the current time step. Eigenvalues below 1 mean a faster convergence, 1 is no convergence, and above 1 means the solutiuon will never converge and the error will grow exponentially fast away from the solution. Solution convergence rate is dominated by the largest eigenvalue. Smoothers by themselves only become better than direct solvers in 3D and are comparable in 2D and are slower in 1D.
    
    Current Example
    ---------------
    Working on the 1D heat conduction equation where AT=q; => -TL+2T-TR=q(dx)^2/kappa; A=-TL+2T-TR, f=q(dx)^2/kappa. Therefore A = D + L + U; D = 2 on the main diagonal, L = -1 on the first lower diagonal, U = -1 on the first upper diagonal"""
    unknowns = [4, 8, 16, 32, 64, 128]
    max_eigGS = np.empty(len(unknowns))
    max_eigJ = np.empty(len(unknowns))

    for i, UN in enumerate(unknowns):
        w = Jacobi(UN)
        max_eigJ[i] = max(abs(w))

        w = Gauss_Seidel(UN)
        max_eigGS[i] = max(abs(w))
    
    plt.figure(1)
    
    plt.semilogx(unknowns, max_eigGS, marker='o', label='Guass-Siedel')
    plt.semilogx(unknowns, max_eigJ, marker='o', label='Jacobian')
    
    plt.xlabel('Unknowns')
    plt.ylabel('Max Eigenvalue Magnitude')
    plt.title('Maximum Eigenvalue Magnitude vs. Unknowns')
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
    main()