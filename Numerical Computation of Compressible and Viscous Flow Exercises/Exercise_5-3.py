
import numpy as np
import matplotlib.pyplot as plt

def solve_tridiagonal(n:int, A, B, C, D):
    """Solves tridiagonal matrix using the Thomas algorithm in O(n) time
    All vectors should be the same length. The last element of A and C
    are not used.
    
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

    for i in range(1, n-1):
        G[i] = C[i]/(B[i] - A[i-1]*G[i-1])
        R[i] = (D[i] - A[i-1]*R[i-1]) / (B[i] - A[i-1]*G[i-1])
    R[i+1] = (D[i+1] - A[i]*R[i]) / (B[i+1] - A[i]*G[i])
    X[i+1] = R[i+1]
    for i in range(n-2, -1, -1):
        X[i] = R[i] - G[i]*X[i+1]

    return X

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

def crank_nicholson(value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    n = len(uns)

    # Equations
    # CFL/2* (un1[i+1]) + (un1[i]) - CFL/2* (un1[i-1]) = un[i]
    # left  boundary un1[i] = 1
    # right boundary un1[i] = 1 + CFL*(un1[i]) - CFL*(un1[i-1]) = un[i]

    # General Setup
    A = -0.5*CFL_num/2*np.ones_like(uns)
    B = np.ones_like(uns)
    C = -A
    D = np.empty_like(uns)

    # Boundary Conditions
    A[n-2] = -CFL_num # left
    B[n-1] = 1 + CFL_num # left
    B[0] = 1 # right
    C[0] = 0 # right

    for t in range(time_steps):
        for i, uni in enumerate(uns[:-1]):
            D[i] = uni - 0.5*CFL_num*0.5*(uns[i+1] - uns[i-1])
        D[0] = uns[0]
        D[n-1] = uns[n-1]
        uns = solve_tridiagonal(n, A, B, C, D)

    return uns

def initial_condition(value1, value2, x_sep, x_start, x_end, num_nodes):
    """Can only produce a two level step"""
    domain = np.linspace(x_start, x_end, num_nodes)
    values1 = np.linspace(value1, value1, int(np.ceil(x_sep/(x_end-x_start)*num_nodes)))
    values2 = np.linspace(value2, value2, int(np.ceil((1-x_sep/(x_end-x_start))*num_nodes))-1)
    return domain, np.concatenate([values1, values2])

def main():
    CFL_num = 0.9
    advection_coeff = 1
    num_points = 40
    x_start = 0
    x_end = 2
    deltax = (x_end-x_start)/num_points
    deltat = CFL_num*deltax*advection_coeff # The expression (domain[0] - domain[1])/advection_coeff should really be a minimum over the spacing of all points
    time_steps = 10
    domain, values_ic = initial_condition(1, .5, .5, 0, 2, num_points+1) # set up inital conditions
    domain, exact_sol = initial_condition(1, .5, 1-deltax, 0, 2, num_points+1) # set up final conditions
    domain, exact_sol2 = initial_condition(1, .5, 1.5, 0, 2, num_points+1) # set up final conditions

    # Running the finite difference methods CFL = 0.9
    values_imc = implicit_central(values_ic, CFL_num, time_steps)
    values_cn = crank_nicholson(values_ic, CFL_num, time_steps)

    CFL_num = 2
    values_imc2 = implicit_central(values_ic, CFL_num, time_steps)
    values_cn2 = crank_nicholson(values_ic, CFL_num, time_steps)

    # Plotting the finite difference methods
    fig, axis = plt.subplots(2, 2, figsize=(16, 13.25))
    plt.subplots_adjust(hspace=.302, bottom=.05, top=.955)
    fig.suptitle('Excerise 5.3', fontsize=16)
    axis[0,0].plot(domain, values_imc, label='Implicit Central', marker='.')
    axis[0,1].plot(domain, values_cn, label='Crank-Nicholson', marker='.')
    
    axis[1,0].plot(domain, values_imc2, label='Implicit Central CFL=2', marker='.')
    axis[1,1].plot(domain, values_cn2, label='Crank Nicholson CFL=2', marker='.')

    for i, plot_row in enumerate(axis):
        if i == 0:
            for plot in plot_row:
                plot.plot(domain, exact_sol, label='Exact Solution', linestyle='-.', zorder=0)
                plot.plot(domain, values_ic, label='Initial Condition', linestyle='--', zorder=0)
                plot.set_xlabel('x')
                plot.set_ylabel('u')
                plot.legend()
                plot.set_ylim([0.25, 1.25])
        else:
            for plot in plot_row:
                plot.plot(domain, exact_sol2, label='Exact Solution CFL=2', linestyle='-.', zorder=0)
                plot.plot(domain, values_ic, label='Initial Condition', linestyle='--', zorder=0)
                plot.set_xlabel('x')
                plot.set_ylabel('u')
                plot.legend()
                plot.set_ylim([0.25, 1.25])

    plt.show()

if __name__ == '__main__':
    main()