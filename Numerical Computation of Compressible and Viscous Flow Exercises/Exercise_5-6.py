import numpy as np
import matplotlib.pyplot as plt
from Method_DB import solve_tridiagonal

def parabolic_soln(value_array, time_steps, alpha, nu, dx):
    """Model parabolic solution equation

    un1 = un + nu*dt*(D+/dx)(D-/dx)*((1-alpha)*un + alpha*un1)

    alpha = 0 full explicit
    alpha = 1/2 crank nicholson
    alpha = 1 fully implicit

    """
    if alpha == 0: # fully explicit
        dt = dx**2/2/nu
    elif alpha == 1/2: # crank nicholson
        dt = 1/8 # chosen from book
    elif alpha == 1: # fully implicit
        dt = 10**9 # chosen from book

    uns  = np.copy(value_array)
    n = len(uns)

    # General Setup
    A = np.ones_like(uns)
    B = np.ones_like(uns)
    C = np.ones_like(uns)
    D = np.ones_like(uns)

    A = A * -(alpha*nu*dt/dx**2)
    B = B * (1 + 2*alpha*nu*dt/dx**2)
    C = C * -(alpha*nu*dt/dx**2)

    # Boundary Conditions
    A[n-2] = 0 # left
    B[n-1] = 1 # left
    B[0] = 1 # right
    C[0] = 0 # right

    for t in range(time_steps):
        for i, uni in enumerate(uns[:-1]):
            D[i] = uni + nu*dt/dx**2*(1-alpha)*(uns[i+1] -2*uni + uns[i-1])
        D[0] = 0 # left boundary condition
        D[n-1] = 1 # right boundary condition
        uns = solve_tridiagonal(n, A, B, C, D)

    return uns

def main():
    x = np.linspace(0, 1, 41)
    dx = 1/40
    nu = 1
    alphas = [0, 1/2, 1]
    timesteps = 100
    values = np.zeros_like(x)
    values[-1] = 1
    

    # Explicit Plot
    for t in range(timesteps):
        final = parabolic_soln(values, t, 0, nu, dx)
        if t % 10 == 0 and t != 0:
            plt.plot(x, final, label="t={}".format(t))

    # Crank Nicholson Plot
    for t in range(6):
        final = parabolic_soln(values, t, 1/2, nu, dx)
        plt.plot(x, final, label="t={}".format(t))

    # Parabolic plot
    for t in range(2):
        final = parabolic_soln(values, t, 1, nu, dx)
        if t != 0:
            plt.plot(x, final, label="t={}".format(t))

    plt.plot(x, final, label="t=100")
    plt.plot(x, values, label="Initial Condition")
    plt.legend()
    plt.show()


if  __name__ == '__main__':
    main()