# Page 64
import numpy as np
import matplotlib.pyplot as plt

""" CODE NOTE:
un1 is the value of the node at the next time step
un  is the value of the node at the current time step
CFL = cdt/dx"""

def explicit_backward(domain_array, value_array, advection_coeff, deltat, time_steps):
    uns = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)):
            un1s[i] = uns[i] - advection_coeff*deltat*(uns[i]-uns[i-1])/(domain_array[i]-domain_array[i-1])

        uns = np.copy(un1s)
    return un1s

def explicit_forward(domain_array, value_array, advection_coeff, deltat, time_steps):
    uns = np.copy(value_array)
    un1s = np.copy(value_array)

    for t in range(time_steps):
        for i in range(1, len(domain_array)):
            un1s[i] = uns[i] - advection_coeff*deltat*(uns[i+1]-uns[i])/(domain_array[i]-domain_array[i-1])
        uns = np.copy(un1s)
    return un1s

def explicit_central():
    pass

def implicit_central():
    pass

def crank_nicolson():
    pass

def lax_method():
    pass

def lax_wendroff():
    pass

def maccormack_method():
    pass

def jameson_method():
    pass

def warming_beam():
    pass

def upwind_method():
    pass

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
    deltat = CFL_num*(x_end-x_start)/num_points*advection_coeff # The expression (domain[0] - domain[1])/advection_coeff should really be a minimum over the spacing of all points
    
    domain, values_ic = initial_condition(1, .5, .5, 0, 2, num_points+1) # set up inital conditions

    values_eb = explicit_backward(domain, values_ic, advection_coeff, deltat, 10)
    values_ef = explicit_forward(domain, values_ic, advection_coeff, deltat, 10)


    plt.plot(domain, values_eb, label='Explicit Backward', marker='.')
    plt.plot(domain, values_ef, label='Explicit Forward', marker='.')
    plt.plot(domain, values_ic, label='Initial Condition', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()
