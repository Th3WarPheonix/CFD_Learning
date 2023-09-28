
import numpy as np
import matplotlib.pyplot as plt
from Method_DB import solve_tridiagonal




def implicit_central(value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    n = len(uns)

    # Equations
    # a*CFL/2* (un1[i+1]) + (un1[i]) - a*CFL/2* (un1[i-1]) = un[i]
    # left  boundary un1[i] = 1
    # right boundary un1[i] = 1 + CFL*(un1[i]) - CFL*(un1[i-1]) = un[i]

    # General Setup
    A = -CFL_num/2*np.ones_like(uns)
    B = np.ones_like(uns)
    C = -A

    # Boundary Conditions
    A[n-2] = -CFL_num # left
    B[n-1] = 1 + CFL_num # left
    B[0] = 1 # right
    C[0] = 0 # right

    for t in range(time_steps):
        D = uns
        uns = solve_tridiagonal(n, A, B, C, D)

    return uns


def main():
    pass


if __name__ == '__main__':
    main()