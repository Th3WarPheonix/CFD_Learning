
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

def stretch_mesh(dymin, ymax, numpts):
    """
    Implements Rakich Stretching function on pg 78 (section 5.3.1).
    Function returns points assuming a stretch from 0 to ymax

    Parameters
    ----------
    dymin : smallest y2-y1 distance from which point spacing will only
    increase
    ymax : distnace over which the stretch occurs
    numpts : number of points in stretch
    """
    ys = np.zeros(numpts)

    kold = 0
    knew = 1
    a = 1/(numpts-1)
    while abs(kold - knew) > 1e-6:
        kold = knew
        fcn = dymin - ymax*(np.exp(kold*a)-1)/(np.exp(kold)-1)
        dfcn = -ymax*((a-1)*np.exp(a*kold+kold)-a*np.exp(a*kold)+np.exp(kold))/(np.exp(kold)-1)**2
        knew = kold - fcn/dfcn

    for j in range(1, numpts):
        a = (j+1-1)/(numpts-1)
        ys[j] = ys[0] + ymax*(np.exp(kold*a)-1)/(np.exp(kold)-1)

    return ys

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

def explicit_backward(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            un1s[i] = uns[i] - CFL_num*(CURRENT_VALUE-LAST_VALUE)

        uns = np.copy(un1s)
    return un1s

def explicit_forward(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]
            un1s[i] = CURRENT_VALUE - CFL_num*(NEXT_VALUE-CURRENT_VALUE)

        uns = np.copy(un1s)
    return un1s

def explicit_central(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]
            un1s[i] = CURRENT_VALUE - 0.5*CFL_num*(NEXT_VALUE-LAST_VALUE)
        uns = np.copy(un1s)

    return un1s

def lax_method(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            NEXT_VALUE    = uns[i+1]
            un1s[i] = (LAST_VALUE + NEXT_VALUE)/2 - 0.5*CFL_num*(NEXT_VALUE-LAST_VALUE)

        uns = np.copy(un1s)
    return un1s

def lax_wendroff(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]
            un1s[i] = CURRENT_VALUE - 0.5*CFL_num*(NEXT_VALUE-LAST_VALUE) + 0.5*CFL_num**2*(NEXT_VALUE-2*CURRENT_VALUE+LAST_VALUE)

        uns = np.copy(un1s)
    return un1s

def maccormack_method(domain_array, value_array, CFL_num, time_steps):
    uns    = np.copy(value_array)
    un1s_2 = np.copy(value_array)
    un1s   = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]
            un1s_2[i] = CURRENT_VALUE - CFL_num*(NEXT_VALUE-CURRENT_VALUE)
            un1s[i] = 0.5*(CURRENT_VALUE + un1s_2[i] - CFL_num*(un1s_2[i]-un1s_2[i-1]))

        uns = np.copy(un1s)
    return un1s

def jameson_method(domain_array, value_array, CFL_num, time_steps):
    un1s = np.copy(value_array)
    uk = np.zeros((5, len(value_array)))
    alphak = [1/4, 1/3, 1/2, 1]
    
    for t in range(time_steps): # cycle through time
        uk[0] = np.copy(un1s)
        uk[:, 0] = uk[0, 0]
        uk[:,-1] = uk[0,-1]
        for i in range(1, 5): # cycle through uk
            for j in range(1, len(domain_array)-1): # cycle through domain
                uk[i,j] = uk[0,j]-alphak[i-1]*CFL_num*0.5*(uk[i-1,j+1]-uk[i-1,j-1])
        
        un1s = np.copy(uk[i])
    return un1s

def warming_beam(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(2, len(domain_array)-1):
            LAST_VALUE2   = uns[i-2]
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]

            un1s[i] = CURRENT_VALUE - 0.5*CFL_num*(3*CURRENT_VALUE-4*LAST_VALUE+LAST_VALUE2) + 0.5*(CFL_num)**2*(LAST_VALUE2-2*LAST_VALUE+CURRENT_VALUE)
        uns = np.copy(un1s)
    return un1s

def upwind_method(domain_array, value_array, CFL_num, time_steps):
    uns  = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(2, len(domain_array)-1):
            LAST_VALUE    = uns[i-1]
            CURRENT_VALUE = uns[i]
            NEXT_VALUE    = uns[i+1]

            # The expression abs(CFL_num) should really be abs(advection_coeff)*deltat/deltax
            un1s[i] = CURRENT_VALUE - 0.5*CFL_num*(NEXT_VALUE-LAST_VALUE) + 0.5*abs(CFL_num)*(NEXT_VALUE-2*CURRENT_VALUE+LAST_VALUE)
        uns = np.copy(un1s)
    return un1s

def initial_condition(value1, value2, x_sep, x_start, x_end, num_nodes):
    """Can only produce a two level step"""
    domain = np.linspace(x_start, x_end, num_nodes)
    values1 = np.linspace(value1, value1, int(np.ceil(x_sep/(x_end-x_start)*num_nodes)))
    values2 = np.linspace(value2, value2, int(np.ceil((1-x_sep/(x_end-x_start))*num_nodes))-1)
    return domain, np.concatenate([values1, values2])